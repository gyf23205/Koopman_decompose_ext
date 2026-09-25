"""Tensor-only interchange for KAEs and complete, trained MoE controllers.

Transfer files contain model state, not optimizer or rollout state. Native
training checkpoints remain responsible for resuming an interrupted run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import math
from typing import Any

import torch
from torch import nn

from .models import (
    ACTION_DIM, FIXED_EXPLORATION_LOG_STD, KAE_HIDDEN_SIZE, KAE_PAD_VALUE,
    KAE_STEPS, MAJORUPDATE_OBSERVATION_CONTRACT, OBSERVATION_DIM, PADDED_DIM,
    ExpandingKaeMoECore, PortableKoopmanAutoencoder, RunningStandardScalerModule,
    StackedPortableKAE,
)


KAE_CHECKPOINT_FORMAT = "kae_moe.kae.v1"
MOE_CHECKPOINT_FORMAT = "kae_moe.policy.v1"
CONTROLLER_CONTRACT = {
    "observation_contract": MAJORUPDATE_OBSERVATION_CONTRACT,
    "observation_dim": OBSERVATION_DIM,
    "action_dim": ACTION_DIM,
    "padded_dim": PADDED_DIM,
    "steps": KAE_STEPS,
    "pad_value": KAE_PAD_VALUE,
    "action_type": "absolute_normalized_waypoint",
    "action_bounds": [-1.0, 1.0],
    "xy_bounds": [-4.0, 4.0],
    "z_bounds": [0.25, 2.0],
    "position_error_scale": [2.0, 2.0, 1.0],
    "velocity_scale": 0.5,
    "control_dt": 0.02,
    "fixed_exploration_log_std": list(FIXED_EXPLORATION_LOG_STD),
}


def _cpu_copy(value: Any) -> Any:
    """Detach tensors and reject Python objects requiring pickle imports."""

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Checkpoint tensors must contain only finite values.")
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Transfer checkpoint dictionary keys must be strings.")
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"Unsupported transfer checkpoint value: {type(value).__name__}.")


def _normalizer_payload(normalizer: nn.Module) -> dict[str, Any]:
    if isinstance(normalizer, nn.Identity):
        return {"type": "identity"}
    if isinstance(normalizer, RunningStandardScalerModule):
        return {
            "type": "running_standard_scaler",
            "running_mean": normalizer.running_mean,
            "running_variance": normalizer.running_variance,
            "epsilon": normalizer.epsilon,
            "clip_threshold": normalizer.clip_threshold,
        }
    raise TypeError("A portable KAE normalizer must be identity or RunningStandardScalerModule.")


def _normalizer_from_payload(payload: Mapping[str, Any]) -> nn.Module:
    if payload.get("type") == "identity":
        return nn.Identity()
    if payload.get("type") != "running_standard_scaler":
        raise ValueError("Unknown KAE normalizer type.")
    mean = payload["running_mean"]
    variance = payload["running_variance"]
    if not isinstance(mean, torch.Tensor) or not isinstance(variance, torch.Tensor):
        raise ValueError("KAE normalizer statistics must be tensors.")
    if mean.shape != (OBSERVATION_DIM,) or variance.shape != (OBSERVATION_DIM,):
        raise ValueError("KAE normalizer statistics must each have 15 entries.")
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("KAE normalizer statistics must be finite with nonnegative variance.")
    epsilon = float(payload["epsilon"])
    clip = float(payload["clip_threshold"])
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(clip) or clip <= 0:
        raise ValueError("Normalizer epsilon and clipping threshold must be finite and positive.")
    return RunningStandardScalerModule(mean, variance, epsilon=epsilon, clip_threshold=clip)


def _kae_payload(model: PortableKoopmanAutoencoder) -> dict[str, Any]:
    if not isinstance(model, PortableKoopmanAutoencoder):
        raise TypeError("Convert external KAEs with PortableKoopmanAutoencoder.from_external first.")
    if not model.has_captured_spectral_components:
        raise ValueError("A KAE transfer requires captured spectral components.")
    payload = {
        "config": {
            "mode_count": model.mode_count,
            "padded_dim": model.padded_dim,
            "hidden_size": model.hidden_size,
        },
        "normalizer": _normalizer_payload(model.encoder.normalizer),
        "state_dict": model.state_dict(),
    }
    _normalizer_from_payload(payload["normalizer"])
    return payload


def _kae_from_payload(payload: Mapping[str, Any], device: str | torch.device) -> PortableKoopmanAutoencoder:
    config = payload["config"]
    if config["padded_dim"] != PADDED_DIM or config["hidden_size"] != KAE_HIDDEN_SIZE:
        raise ValueError("KAE architecture must use padded_dim=16 and hidden_size=64.")
    count = config["mode_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("KAE mode_count must be a positive integer.")
    state = payload["state_dict"]
    spectrum = tuple(state[name] for name in (
        "_captured_eigenvalues", "_captured_eigenvectors", "_captured_eigenvectors_inverse",
    ))
    # Supply the stored basis at construction: loading never calls eig().
    model = PortableKoopmanAutoencoder(
        count, koopman_matrix=state["K"],
        padded_dim=config["padded_dim"], hidden_size=config["hidden_size"],
        spectral_components=spectrum,
    )
    model.encoder.normalizer = _normalizer_from_payload(payload["normalizer"])
    for name, expected in model.encoder.normalizer.state_dict().items():
        saved = state.get(f"encoder.normalizer.{name}")
        if not isinstance(saved, torch.Tensor) or not torch.equal(saved, expected):
            raise ValueError("KAE normalizer configuration disagrees with its saved state.")
    model.load_state_dict(state, strict=True)
    eigenvalues, vectors, _ = model.spectral_components()
    if not torch.allclose(
        model.K.to(torch.complex128) @ vectors, vectors * eigenvalues,
        rtol=1e-6, atol=1e-8,
    ):
        raise ValueError("Captured KAE spectral basis does not match its Koopman matrix.")
    return model.to(device).freeze()


def _read(path: str | Path, expected_format: str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != expected_format:
        raise ValueError(f"Expected a {expected_format} transfer checkpoint.")
    if payload.get("contract") != CONTROLLER_CONTRACT:
        raise ValueError("Checkpoint observation/action contract does not match this controller.")
    _cpu_copy(payload)  # Also checks finiteness and the permitted value types.
    return payload


def _write(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_copy(payload), output)
    return output


def save_kae_checkpoint(
    model: PortableKoopmanAutoencoder,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export a frozen block, including K, normalizer, and its exact mode basis."""

    return _write({
        "format": KAE_CHECKPOINT_FORMAT,
        "contract": CONTROLLER_CONTRACT,
        "kae": _kae_payload(model),
        "metadata": dict(metadata or {}),
    }, path)


def load_kae_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> PortableKoopmanAutoencoder:
    """Load a portable KAE without importing any notebook-specific classes."""

    payload = _read(path, KAE_CHECKPOINT_FORMAT)
    return _kae_from_payload(payload["kae"], device)


def save_moe_checkpoint(
    core: ExpandingKaeMoECore,
    path: str | Path,
    *,
    critic_state: Mapping[str, Any] | None = None,
    value_preprocessor_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export a trained policy for another task; omit optimizer/resume state."""

    if not isinstance(core, ExpandingKaeMoECore):
        raise TypeError("A policy transfer requires ExpandingKaeMoECore.")
    if core.observation_contract != MAJORUPDATE_OBSERVATION_CONTRACT or core.action_bounds != (-1.0, 1.0):
        raise ValueError("Controller observation or action bounds are incompatible.")
    return _write({
        "format": MOE_CHECKPOINT_FORMAT,
        "contract": CONTROLLER_CONTRACT,
        "config": {
            "observation_dim": core.observation_dim,
            "action_dim": core.action_dim,
            "steps": core.steps,
            "pad_value": core.pad_value,
            "residual_scale": core.residual_scale,
            "sensitivity_scale": core.sensitivity_scale,
            "step_dt": core.step_dt,
        },
        "blocks": [_kae_payload(block) for block in core.kae.blocks],
        "block_origins": list(core.kae.block_origins),
        "mode_slices": [list(value) for value in core.kae.mode_slices],
        "head_state": core.head.state_dict(),
        "critic_state": critic_state,
        "value_preprocessor_state": value_preprocessor_state,
        "metadata": dict(metadata or {}),
    }, path)


def load_moe_checkpoint(
    path: str | Path, device: str | torch.device = "cpu",
) -> tuple[ExpandingKaeMoECore, dict[str, Any]]:
    """Reconstruct the complete policy with the saved branch and mode order."""

    payload = _read(path, MOE_CHECKPOINT_FORMAT)
    blocks = [_kae_from_payload(block, device) for block in payload["blocks"]]
    stack = StackedPortableKAE(
        blocks, payload["block_origins"], mode_slices=payload["mode_slices"],
    )
    core = ExpandingKaeMoECore(stack, **payload["config"]).to(device)
    core.head.load_state_dict(payload["head_state"], strict=True)
    core.eval()
    return core, {
        "metadata": payload["metadata"],
        "critic_state": payload["critic_state"],
        "value_preprocessor_state": payload["value_preprocessor_state"],
    }
