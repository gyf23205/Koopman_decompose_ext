"""Portable cumulative-KAE mixture-of-experts components.

Each external KAE contributes one frozen block using a
16-dimensional padded input, a 64-unit encoder, and a linear decoder. Ordered
blocks form a runtime stack with mode counts inferred from their checkpoints.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F



MAJORUPDATE_OBSERVATION_CONTRACT = "crazyflie_rl_majorupdate_f7e73b4_waypoint_v1"
OBSERVATION_DIM = 15
ACTION_DIM = 3
PADDED_DIM = 16
KAE_HIDDEN_SIZE = 64
KAE_STEPS = 1
KAE_PAD_VALUE = 1.0
FIXED_EXPLORATION_LOG_STD = (-2.7, -2.6, -1.6)


def _validate_fixed_log_std(
    value: Sequence[float] | torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    log_std = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    expected = torch.as_tensor(FIXED_EXPLORATION_LOG_STD, dtype=torch.float32)
    if action_dim != expected.numel():
        raise ValueError(
            "The real-world bounded exploration contract requires exactly "
            f"{expected.numel()} actions, but received {action_dim}."
        )
    if log_std.shape != expected.shape or not torch.equal(log_std, expected):
        raise ValueError(
            "initial_log_std must exactly match the fixed real-world contract "
            f"{FIXED_EXPLORATION_LOG_STD}."
        )
    return log_std


def _elu_network(input_dim, hidden_dims, output_dim):
    layers = []
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class HierarchicalMoEHead(nn.Module):
    """Route within each KAE, route between KAEs, then blend with an MLP."""

    def __init__(self, obs_dim, act_dim, branch_mode_counts):
        super().__init__()
        self.branch_mode_counts = tuple(branch_mode_counts)
        if not self.branch_mode_counts or any(count <= 0 for count in self.branch_mode_counts):
            raise ValueError("MoE requires at least one nonempty KAE branch.")
        self.mlp_network = _elu_network(obs_dim, (256, 128, 64), act_dim)
        self.expert_weight_networks = nn.ModuleList(
            _elu_network(obs_dim, (32,), count) for count in self.branch_mode_counts
        )
        self.kae_router_network = _elu_network(obs_dim, (128,), len(self.branch_mode_counts))
        self.gating_network = _elu_network(obs_dim, (32,), 1)
        with torch.no_grad():
            for network in (*self.expert_weight_networks, self.gating_network):
                network[-1].weight.zero_()
                network[-1].bias.fill_(1.0)

    def expert_weights(self, observations):
        router = torch.softmax(self.kae_router_network(observations), dim=-1)
        weights = []
        for index, network in enumerate(self.expert_weight_networks):
            branch_weights = network(observations)
            branch_weights = branch_weights / (branch_weights.abs().sum(dim=-1, keepdim=True) + 1e-6)
            weights.append(branch_weights * router[:, index:index + 1])
        return torch.cat(weights, dim=-1)

    def forward(self, observations, expert_actions, *, kae_action_scale=1.0, mlp_action_scale=1.0):
        weights = self.expert_weights(observations)
        # Preserve the spectral path's precision until after summing the modes.
        kae_action = kae_action_scale * torch.sum(
            weights.to(expert_actions.dtype).unsqueeze(-1) * expert_actions, dim=1,
        ).to(observations.dtype)
        mlp_action = mlp_action_scale * self.mlp_network(observations)
        gate = torch.sigmoid(self.gating_network(observations))

        return {
            "kae_action": kae_action,
            # Keep the existing telemetry name for the trainable MLP pathway.
            "residual_action": mlp_action,
            "gate": gate,
            "final_mean": gate * kae_action + (1.0 - gate) * mlp_action,
            "expert_weights": weights,
        }



class RunningStandardScalerModule(nn.Module):

    def __init__(
        self,
        running_mean: torch.Tensor,
        running_variance: torch.Tensor,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
    ):
        super().__init__()
        self.register_buffer("running_mean", running_mean.float())
        self.register_buffer("running_variance", running_variance.float())
        self.epsilon = float(epsilon)
        self.clip_threshold = float(clip_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        normalized = (x - self.running_mean) / (torch.sqrt(self.running_variance) + self.epsilon)
        return torch.clamp(normalized, min=-self.clip_threshold, max=self.clip_threshold)


class _PortableKaeEncoder(nn.Module):
    """Architecture-compatible copy of the external notebook encoder."""

    def __init__(self, padded_dim: int, hidden_size: int, mode_count: int) -> None:
        super().__init__()
        self.normalizer = nn.Identity()
        self.encoder = nn.Sequential(
            nn.Linear(padded_dim, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, hidden_size * 3),
            nn.Tanh(),
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, mode_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Normalize only the observations; keep the constant padding unchanged.
        observations = self.normalizer(inputs[..., :OBSERVATION_DIM])
        padded = torch.cat((observations, inputs[..., OBSERVATION_DIM:]), dim=-1)
        return self.encoder(padded)


class _PortableKaeDecoder(nn.Module):
    """Architecture-compatible copy of the external notebook decoder."""

    def __init__(self, mode_count: int, padded_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(mode_count, padded_dim, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.linear(latent)


class PortableKoopmanAutoencoder(nn.Module):
    """One frozen KAE block used inside the cumulative controller.

    ``K`` is a registered buffer.  It therefore follows device transfers and
    is included in ``state_dict``. The class contains no external paths or
    external Python types.
    """

    def __init__(
        self,
        mode_count: int,
        *,
        koopman_matrix: torch.Tensor | None = None,
        padded_dim: int = PADDED_DIM,
        hidden_size: int = KAE_HIDDEN_SIZE,
        spectral_components: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if padded_dim != PADDED_DIM:
            raise ValueError(
                f"Expanding KAE artifacts require padded_dim={PADDED_DIM}, got {padded_dim}."
            )
        if hidden_size != KAE_HIDDEN_SIZE:
            raise ValueError(
                f"Expanding KAE artifacts require hidden_size={KAE_HIDDEN_SIZE}, got {hidden_size}."
            )
        if mode_count <= 0:
            raise ValueError(f"KAE mode_count must be positive, got {mode_count}.")

        self.padded_dim = int(padded_dim)
        self.state_dim = self.padded_dim
        self.hidden_size = int(hidden_size)
        self.hidden_dim = self.hidden_size
        self.mode_count = int(mode_count)
        self.observable_dim = self.mode_count
        # Names intentionally match the external model for direct state copying.
        self.encoder = _PortableKaeEncoder(padded_dim, hidden_size, mode_count)
        self.decoder = _PortableKaeDecoder(mode_count, padded_dim)

        if koopman_matrix is None:
            matrix = torch.eye(mode_count, dtype=torch.float32)
        else:
            matrix = torch.as_tensor(koopman_matrix).detach().to(dtype=torch.float32).clone()
        if matrix.shape != (mode_count, mode_count):
            raise ValueError(
                "Koopman matrix shape must equal (mode_count, mode_count); "
                f"received {tuple(matrix.shape)} for {mode_count} modes."
            )
        if not torch.isfinite(matrix).all():
            raise ValueError("Koopman matrix contains non-finite values.")
        self.register_buffer("K", matrix, persistent=True)
        self._spectral_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # Keep parameters and the supplied Koopman buffer on one device even
        # when normalizing an external CUDA pickle.
        self.to(matrix.device)
        if spectral_components is None:
            eigenvalues, eigenvectors = torch.linalg.eig(self.K.to(torch.float64))
            try:
                eigenvectors_inverse = torch.linalg.inv(eigenvectors)
            except torch.linalg.LinAlgError as exc:
                raise ValueError("Koopman eigenvector matrix is singular.") from exc
            spectral_components = (eigenvalues, eigenvectors, eigenvectors_inverse)
        self.capture_spectral_components(*spectral_components)
        self.freeze()

    @classmethod
    def from_external(cls, model: nn.Module) -> "PortableKoopmanAutoencoder":
        """Copy a notebook KAE into an external-type-free portable module."""

        try:
            matrix = torch.as_tensor(getattr(model, "K")).detach()
            external_encoder = getattr(model, "encoder")
            external_decoder = getattr(model, "decoder")
            encoder_network = getattr(external_encoder, "encoder")
            decoder_linear = getattr(external_decoder, "linear")
            first_linear = encoder_network[0]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("External KAE does not expose the expected encoder/decoder/K layout.") from exc

        if not isinstance(first_linear, nn.Linear) or not isinstance(decoder_linear, nn.Linear):
            raise ValueError("External KAE encoder and decoder must use the expected linear layers.")
        mode_count = int(matrix.shape[0]) if matrix.ndim == 2 else -1
        if matrix.shape != (mode_count, mode_count):
            raise ValueError("External KAE K must be a square matrix.")
        if decoder_linear.in_features != mode_count:
            raise ValueError("External KAE decoder width does not match its Koopman mode count.")
        if first_linear.in_features != PADDED_DIM or first_linear.out_features % 4:
            raise ValueError("External KAE does not use the required 16-dimensional padded encoder.")
        hidden_size = first_linear.out_features // 4
        if len(encoder_network) != 9 or any(
            not isinstance(encoder_network[index], nn.Tanh) for index in (1, 3, 5, 7)
        ):
            raise ValueError("External KAE encoder must use four Tanh hidden activations.")

        portable = cls(
            mode_count,
            koopman_matrix=matrix,
            padded_dim=first_linear.in_features,
            hidden_size=hidden_size,
            spectral_components=model.spectral_components() if isinstance(model, cls) else None,
        )
        if isinstance(model, cls):
            portable.encoder.normalizer = deepcopy(model.encoder.normalizer).to(portable.K.device)
        try:
            portable.encoder.load_state_dict(external_encoder.state_dict(), strict=True)
            portable.decoder.load_state_dict(external_decoder.state_dict(), strict=True)
        except RuntimeError as exc:
            raise ValueError("External KAE architecture does not match the portable KAE contract.") from exc
        if not all(torch.isfinite(value).all() for value in portable.state_dict().values()):
            raise ValueError("External KAE contains non-finite parameters.")
        return portable.freeze()

    def freeze(self) -> "PortableKoopmanAutoencoder":
        self.requires_grad_(False)
        super().train(False)
        return self

    def train(self, mode: bool = True) -> "PortableKoopmanAutoencoder":
        """Remain in evaluation mode when a parent MoE enters training mode."""

        del mode
        return self.freeze()

    def _apply(self, fn: Any, recurse: bool = True) -> "PortableKoopmanAutoencoder":
        self._spectral_cache = None
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self._spectral_cache = None
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.freeze()
        return result

    @property
    def has_captured_spectral_components(self) -> bool:
        return all(
            hasattr(self, name)
            for name in (
                "_captured_eigenvalues",
                "_captured_eigenvectors",
                "_captured_eigenvectors_inverse",
            )
        )

    def capture_spectral_components(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvectors_inverse: torch.Tensor,
    ) -> "PortableKoopmanAutoencoder":
        """Persist the accepted eigenbasis so expert identities cannot reorder.

        A stacked controller never eigendecomposes a synthetic block-diagonal
        operator. Each atomic KAE instead carries the exact basis accepted at
        import, including its ordering.
        """

        values = torch.as_tensor(eigenvalues).detach().to(torch.complex128).clone()
        vectors = torch.as_tensor(eigenvectors).detach().to(torch.complex128).clone()
        inverse = torch.as_tensor(eigenvectors_inverse).detach().to(torch.complex128).clone()
        expected_vector_shape = (self.mode_count, self.mode_count)
        if values.shape != (self.mode_count,):
            raise ValueError(
                f"Captured eigenvalues must have shape [{self.mode_count}], got {tuple(values.shape)}."
            )
        if vectors.shape != expected_vector_shape or inverse.shape != expected_vector_shape:
            raise ValueError(
                "Captured eigenvectors and inverse must both have shape "
                f"{expected_vector_shape}."
            )
        if not all(torch.isfinite(value).all() for value in (values, vectors, inverse)):
            raise ValueError("Captured spectral components contain non-finite values.")
        identity = vectors @ inverse
        if not torch.allclose(
            identity,
            torch.eye(self.mode_count, dtype=torch.complex128, device=identity.device),
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError("Captured eigenvector inverse is inconsistent with the eigenvectors.")

        for name, value in (
            ("_captured_eigenvalues", values),
            ("_captured_eigenvectors", vectors),
            ("_captured_eigenvectors_inverse", inverse),
        ):
            value = value.to(self.K.device)
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                self.register_buffer(name, value, persistent=True)
        self._spectral_cache = None
        return self.freeze()

    def spectral_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return eigenvalues, right eigenvectors, and their inverse."""

        if self._spectral_cache is None:
            if not self.has_captured_spectral_components:
                raise RuntimeError("Cumulative KAE blocks require a captured spectral basis.")
            self._spectral_cache = (
                self._captured_eigenvalues,
                self._captured_eigenvectors,
                self._captured_eigenvectors_inverse,
            )
        return self._spectral_cache

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.shape[-1] != self.padded_dim:
            raise ValueError(
                f"KAE expects {self.padded_dim} input values, got {inputs.shape[-1]}."
            )
        latent = self.encoder(inputs)
        propagated_latent = latent @ self.K.T
        return self.decoder(latent), latent, self.decoder(propagated_latent)


def koopman_expert_actions(
    kae: PortableKoopmanAutoencoder,
    latent: torch.Tensor,
    *,
    action_dim: int = ACTION_DIM,
    steps: int = KAE_STEPS,
) -> torch.Tensor:
    """Decompose a KAE action into one contribution per Koopman mode.

    The result has shape ``[batch, mode_count, action_dim]``.  Summing over the
    mode axis reconstructs ``decoder(latent @ matrix_power(K, steps).T)`` up to
    eigendecomposition roundoff.
    """

    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    if latent.ndim != 2 or latent.shape[-1] != kae.mode_count:
        raise ValueError(
            f"latent must have shape [batch, {kae.mode_count}], got {tuple(latent.shape)}."
        )
    if not 0 < action_dim <= kae.padded_dim:
        raise ValueError(f"action_dim must be in [1, {kae.padded_dim}], got {action_dim}.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    eigenvalues, eigenvectors, eigenvectors_inverse = kae.spectral_components()
    complex_dtype = eigenvalues.dtype
    latent_complex = latent.to(dtype=complex_dtype)
    decoder = kae.decoder.linear.weight[:action_dim].to(dtype=complex_dtype)

    # c = V^-1 z (column convention); batched rows therefore use V^-T.
    modal_coordinates = latent_complex @ eigenvectors_inverse.T
    decoded_modes = decoder @ eigenvectors
    propagated_coordinates = modal_coordinates * eigenvalues.pow(steps).unsqueeze(0)
    contributions = torch.einsum("bm,am->bma", propagated_coordinates, decoded_modes).real
    if not torch.isfinite(contributions).all():
        raise RuntimeError("Koopman expert decomposition produced non-finite contributions.")
    # Keep expert contributions in float64 through dynamic reweighting. The
    # controller casts only the final summed action back to its policy dtype
    # so cancellation between modes does not lose unnecessary precision.
    return contributions


class StackedPortableKAE(nn.Module):
    """Ordered additive stack of frozen KAE blocks.

    Blocks remain independent: each observation is encoded by every retained
    KAE, each block is decomposed in its own captured spectral basis, and the
    resulting expert tensors are concatenated. There is intentionally no
    synthetic global Koopman matrix whose eigendecomposition could mix modes
    from different tasks.
    """

    def __init__(
        self,
        blocks: Sequence[PortableKoopmanAutoencoder],
        block_origins: Sequence[str],
        *,
        mode_slices: Sequence[tuple[int, int]] | None = None,
        require_captured_spectra: bool = True,
    ) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("A stacked KAE requires at least one mode block.")
        if len(blocks) != len(block_origins):
            raise ValueError("KAE block origins must align one-to-one with the blocks.")

        normalized_origins = tuple(str(origin).strip() for origin in block_origins)
        if any(not origin for origin in normalized_origins):
            raise ValueError("KAE block origins cannot be empty.")
        if len(set(normalized_origins)) != len(normalized_origins):
            raise ValueError("KAE block origins must be unique and ordered.")

        normalized_blocks: list[PortableKoopmanAutoencoder] = []
        computed_slices: list[tuple[int, int]] = []
        offset = 0
        for index, block in enumerate(blocks):
            if not isinstance(block, PortableKoopmanAutoencoder):
                raise TypeError(
                    "Stacked KAE blocks must be PortableKoopmanAutoencoder instances; "
                    f"block {index} is {type(block)!r}."
                )
            if block.padded_dim != PADDED_DIM or block.hidden_size != KAE_HIDDEN_SIZE:
                raise ValueError("Every stacked KAE block must use padded_dim=16 and hidden_size=64.")
            if require_captured_spectra and not block.has_captured_spectral_components:
                raise ValueError(
                    f"KAE block {normalized_origins[index]!r} has no captured spectral basis."
                )
            normalized_blocks.append(block.freeze())
            computed_slices.append((offset, offset + block.mode_count))
            offset += block.mode_count

        if mode_slices is not None:
            supplied_slices = tuple((int(start), int(stop)) for start, stop in mode_slices)
            if supplied_slices != tuple(computed_slices):
                raise ValueError(
                    f"Stack mode slices {supplied_slices} do not match {tuple(computed_slices)}."
                )

        self.blocks = nn.ModuleList(normalized_blocks)
        self.block_origins = normalized_origins
        self.mode_slices = tuple(computed_slices)
        self.mode_count = offset
        self.observable_dim = self.mode_count
        self.padded_dim = PADDED_DIM
        self.state_dim = PADDED_DIM
        self.hidden_size = KAE_HIDDEN_SIZE
        self.hidden_dim = KAE_HIDDEN_SIZE
        self.freeze()

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def mode_origins(self) -> tuple[str, ...]:
        return tuple(
            origin
            for origin, block in zip(self.block_origins, self.blocks)
            for _ in range(block.mode_count)
        )

    def freeze(self) -> "StackedPortableKAE":
        for block in self.blocks:
            block.freeze()
        self.requires_grad_(False)
        super().train(False)
        return self

    def train(self, mode: bool = True) -> "StackedPortableKAE":
        del mode
        return self.freeze()

    def expert_actions(
        self,
        inputs: torch.Tensor,
        *,
        action_dim: int = ACTION_DIM,
        steps: int = KAE_STEPS,
    ) -> torch.Tensor:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 2 or inputs.shape[-1] != self.padded_dim:
            raise ValueError(
                f"Stacked KAE expects inputs shaped [batch, {self.padded_dim}], "
                f"got {tuple(inputs.shape)}."
            )

        contributions: list[torch.Tensor] = []
        for block in self.blocks:
            _, latent, _ = block(inputs)
            contributions.append(
                koopman_expert_actions(
                    block,
                    latent,
                    action_dim=action_dim,
                    steps=steps,
                )
            )
        experts = torch.cat(contributions, dim=1)
        expected_shape = (inputs.shape[0], self.mode_count, action_dim)
        if experts.shape != expected_shape:
            raise RuntimeError(
                f"Stacked expert output has shape {tuple(experts.shape)}, expected {expected_shape}."
            )
        return experts

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 2 or inputs.shape[-1] != self.padded_dim:
            raise ValueError(
                f"Stacked KAE expects inputs shaped [batch, {self.padded_dim}], "
                f"got {tuple(inputs.shape)}."
            )

        reconstructed: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []
        propagated: list[torch.Tensor] = []
        for block in self.blocks:
            block_reconstructed, block_latent, block_propagated = block(inputs)
            reconstructed.append(block_reconstructed)
            latents.append(block_latent)
            propagated.append(block_propagated)
        return (
            torch.stack(reconstructed, dim=0).sum(dim=0),
            torch.cat(latents, dim=-1),
            torch.stack(propagated, dim=0).sum(dim=0),
        )


def expert_weight_sensitivity_penalty(
    current_weights: torch.Tensor,
    previous_weights: torch.Tensor | None,
    *,
    scale: float = -0.1,
    step_dt: float = 0.02,
) -> torch.Tensor:
    """Compute the historical temporal expert-weight reward penalty."""

    if previous_weights is None:
        return torch.zeros(
            (*current_weights.shape[:-1], 1),
            dtype=current_weights.dtype,
            device=current_weights.device,
        )
    if previous_weights.shape != current_weights.shape:
        raise ValueError(
            "previous expert weights must match current expert weights; "
            f"got {tuple(previous_weights.shape)} and {tuple(current_weights.shape)}."
        )
    squared_change = torch.square(current_weights - previous_weights).sum(dim=-1, keepdim=True)
    return float(scale) * float(step_dt) * squared_change


@dataclass(frozen=True)
class MoEMechanismOutput:
    """Auditable components of one dynamic-mode MoE forward pass."""

    kae_action: torch.Tensor
    residual_action: torch.Tensor
    gate: torch.Tensor
    final_mean: torch.Tensor
    sensitivity_penalty: torch.Tensor
    expert_weights: torch.Tensor
    expert_actions: torch.Tensor

    def policy_outputs(self) -> dict[str, torch.Tensor]:
        return {
            "kae_action": self.kae_action,
            "residual_action": self.residual_action,
            "gate": self.gate,
            "final_mean": self.final_mean,
            "sensitivity_penalty": self.sensitivity_penalty,
            "expert_weights": self.expert_weights,
            "expert_actions": self.expert_actions,
        }


def mechanism_log_record(
    output: MoEMechanismOutput,
    batch_index: int = 0,
    *,
    mode_origins: Sequence[str] | None = None,
) -> dict[str, float]:
    """Flatten one mechanism result into dynamic, stage-safe log columns."""

    def values(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
        flat = tensor[batch_index].detach().reshape(-1).cpu()
        return {f"{prefix}_{index:02d}": float(value) for index, value in enumerate(flat)}

    record: dict[str, float] = {}
    record.update(values("kae_action", output.kae_action))
    record.update(values("residual_action", output.residual_action))
    record.update(values("gate", output.gate))
    record.update(values("final_mean", output.final_mean))
    record.update(values("sensitivity_penalty", output.sensitivity_penalty))
    record.update(values("expert_weight", output.expert_weights))
    if mode_origins is not None:
        origins = tuple(str(value) for value in mode_origins)
        expert_weights = output.expert_weights[batch_index].detach().reshape(-1).cpu()
        if len(origins) != expert_weights.numel():
            raise ValueError(
                "mode_origins must contain one label per expert weight; "
                f"got {len(origins)} labels and {expert_weights.numel()} weights."
            )
        local_indices: dict[str, int] = {}
        for origin, weight in zip(origins, expert_weights):
            safe_origin = "".join(character if character.isalnum() else "_" for character in origin)
            local_index = local_indices.get(safe_origin, 0)
            local_indices[safe_origin] = local_index + 1
            record[f"expert_weight_{safe_origin}_{local_index:02d}"] = float(weight)
    return record


class ExpandingKaeMoECore(nn.Module):
    """Hierarchical routing and MLP blending around a frozen KAE stack."""

    observation_contract = MAJORUPDATE_OBSERVATION_CONTRACT
    action_bounds = (-1.0, 1.0)

    def __init__(
        self,
        kae: StackedPortableKAE,
        *,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        steps: int = KAE_STEPS,
        pad_value: float = KAE_PAD_VALUE,
        residual_scale: float = 1.0,
        sensitivity_scale: float = -0.1,
        step_dt: float = 0.02,
    ) -> None:
        super().__init__()
        if not isinstance(kae, StackedPortableKAE):
            raise TypeError("The current MoE requires a cumulative StackedPortableKAE.")
        if observation_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                "MajorUpdate MoE requires observation_dim=15 and action_dim=3; "
                f"got {observation_dim} and {action_dim}."
            )
        if kae.padded_dim != PADDED_DIM:
            raise ValueError(f"MajorUpdate MoE requires a {PADDED_DIM}-wide portable KAE.")
        if steps != KAE_STEPS:
            raise ValueError(f"MajorUpdate expanding KAE requires p={KAE_STEPS}.")
        if pad_value != KAE_PAD_VALUE:
            raise ValueError(f"MajorUpdate expanding KAE requires pad_value={KAE_PAD_VALUE}.")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive.")
        if not math.isfinite(sensitivity_scale):
            raise ValueError("sensitivity_scale must be finite.")
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be finite and positive.")

        self.kae = kae.freeze()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.mode_count = kae.mode_count
        self.steps = steps
        self.pad_value = float(pad_value)
        self.residual_scale = float(residual_scale)
        self.sensitivity_scale = float(sensitivity_scale)
        self.step_dt = float(step_dt)

        self.head = HierarchicalMoEHead(
            observation_dim, action_dim, [block.mode_count for block in kae.blocks],
        )
        self.last_mechanism_output: MoEMechanismOutput | None = None

    def expert_weights(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head.expert_weights(observations)

    def mechanism(
        self,
        observations: torch.Tensor,
        *,
        previous_expert_weights: torch.Tensor | None = None,
    ) -> MoEMechanismOutput:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observations must have shape [batch, {self.observation_dim}], "
                f"got {tuple(observations.shape)}."
            )

        padding = torch.full(
            (observations.shape[0], PADDED_DIM - self.observation_dim),
            self.pad_value,
            dtype=observations.dtype,
            device=observations.device,
        )
        padded_observations = torch.cat((observations, padding), dim=-1)
        with torch.no_grad():
            expert_actions = self.kae.expert_actions(
                padded_observations,
                action_dim=self.action_dim,
                steps=self.steps,
            )

        routed = self.head(observations, expert_actions, mlp_action_scale=self.residual_scale)
        # The policy center must be a normalized waypoint before exploration.
        routed["final_mean"] = routed["final_mean"].clamp(*self.action_bounds)
        output = MoEMechanismOutput(
            **routed,
            expert_actions=expert_actions,
            sensitivity_penalty=expert_weight_sensitivity_penalty(
                routed["expert_weights"], previous_expert_weights,
                scale=self.sensitivity_scale, step_dt=self.step_dt,
            ),
        )
        self.last_mechanism_output = output
        return output

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mechanism(observations).final_mean


class SquashedNormal:
    """Tanh-squashed diagonal Normal with stable inverse likelihoods."""

    def __init__(
        self,
        bounded_mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        epsilon: float = 1e-6,
    ) -> None:
        if bounded_mean.shape[-1] != log_std.shape[-1]:
            raise ValueError("bounded_mean and log_std must have the same final dimension.")
        self.epsilon = float(epsilon)
        # Preserve the exact deterministic center (including a legitimate
        # boundary value) while using a numerically safe latent location.
        self.bounded_mean = bounded_mean.clamp(-1.0, 1.0)
        safe_bounded_mean = self.bounded_mean.clamp(-1.0 + epsilon, 1.0 - epsilon)
        self.pre_tanh_loc = torch.atanh(safe_bounded_mean)
        self.scale = log_std.exp().expand_as(bounded_mean)
        self.base_dist = torch.distributions.Normal(self.pre_tanh_loc, self.scale)
        self.has_rsample = True

    @property
    def mean(self) -> torch.Tensor:
        return self.bounded_mean

    @property
    def mode(self) -> torch.Tensor:
        return self.bounded_mean

    @property
    def stddev(self) -> torch.Tensor:
        # Delta-method value used only for PPO diagnostics.
        return (1.0 - self.bounded_mean.square()) * self.scale

    @property
    def variance(self) -> torch.Tensor:
        return self.stddev.square()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        bounded = value.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        pre_tanh = torch.atanh(bounded)
        # log(1 - tanh(x)^2), written in a stable form.
        log_abs_det_jacobian = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return self.base_dist.log_prob(pre_tanh) - log_abs_det_jacobian

    def entropy(self) -> torch.Tensor:
        # The transformed entropy is not analytic.  This deterministic
        # first-order approximation keeps SKRL's entropy interface stable;
        # real experiment configs use zero entropy-loss scale.
        jacobian_at_mode = torch.log1p(-self.bounded_mean.square() + self.epsilon)
        return self.base_dist.entropy() + jacobian_at_mode


def bounded_action_sample(
    bounded_mean: torch.Tensor,
    log_std: Sequence[float] | torch.Tensor = FIXED_EXPLORATION_LOG_STD,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, SquashedNormal]:
    """Sample a normalized action and return its summed corrected log-prob."""

    log_std_tensor = torch.as_tensor(log_std, dtype=bounded_mean.dtype, device=bounded_mean.device)
    log_std_tensor = log_std_tensor.reshape(-1)
    if log_std_tensor.numel() != bounded_mean.shape[-1]:
        raise ValueError("log_std width must match the bounded action mean.")
    distribution = SquashedNormal(bounded_mean, log_std_tensor)
    actions = distribution.mode if deterministic else distribution.rsample()
    log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
    return actions, log_prob, distribution
