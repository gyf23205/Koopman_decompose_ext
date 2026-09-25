"""Optional SKRL model adapters for the shared torch-only controller."""
from __future__ import annotations
from typing import Any, Mapping
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model
from .models import (
    ACTION_DIM, OBSERVATION_DIM, KAE_STEPS, KAE_PAD_VALUE,
    FIXED_EXPLORATION_LOG_STD, ExpandingKaeMoECore,
    StackedPortableKAE, SquashedNormal, _validate_fixed_log_std,
)

class SquashedGaussianMixin:
    """SKRL-compatible stochastic model mixin with bounded PPO actions."""

    def __init__(self, *, reduction: str = "sum") -> None:
        if reduction not in ("mean", "sum", "prod", "none"):
            raise ValueError("reduction must be mean, sum, prod, or none.")
        self._squashed_reduction = reduction
        self._squashed_distribution: SquashedNormal | None = None
        self._squashed_log_std: torch.Tensor | None = None
        self._squashed_num_samples = 0

    def act(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        mean_actions, log_std, raw_outputs = self.compute(inputs, role)
        outputs = dict(raw_outputs)
        log_std = log_std.to(device=mean_actions.device, dtype=mean_actions.dtype).reshape(-1)
        if log_std.numel() != mean_actions.shape[-1]:
            raise ValueError("Policy log_std width does not match its action dimension.")

        distribution = SquashedNormal(mean_actions, log_std)
        sampled_actions = distribution.rsample()
        likelihood_actions = inputs.get("taken_actions", sampled_actions)
        component_log_prob = distribution.log_prob(likelihood_actions)
        if self._squashed_reduction == "sum":
            log_prob = component_log_prob.sum(dim=-1, keepdim=True)
        elif self._squashed_reduction == "mean":
            log_prob = component_log_prob.mean(dim=-1, keepdim=True)
        elif self._squashed_reduction == "prod":
            log_prob = component_log_prob.prod(dim=-1, keepdim=True)
        else:
            log_prob = component_log_prob

        self._squashed_distribution = distribution
        self._squashed_log_std = log_std
        self._squashed_num_samples = mean_actions.shape[0]
        outputs["mean_actions"] = mean_actions
        return sampled_actions, log_prob, outputs

    def deterministic_act(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, None, dict[str, Any]]:
        """Return the exact bounded policy center without exploration noise."""

        mean_actions, _, raw_outputs = self.compute(inputs, role)
        outputs = dict(raw_outputs)
        outputs["mean_actions"] = mean_actions
        return mean_actions, None, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        del role
        if self._squashed_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._squashed_distribution.entropy()

    def get_log_std(self, role: str = "") -> torch.Tensor:
        del role
        if self._squashed_log_std is None:
            raise RuntimeError("act must be called before get_log_std.")
        return self._squashed_log_std.repeat(self._squashed_num_samples, 1)

    def distribution(self, role: str = "") -> SquashedNormal | None:
        del role
        return self._squashed_distribution


class ExpandingKaeMoEPolicy(SquashedGaussianMixin, Model):
    """SKRL PPO policy for an expanding-mode KAE MoE stage."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device,
        kae_model: StackedPortableKAE,
        **kwargs: Any,
    ) -> None:
        if "expert_weight_range" in kwargs:
            raise ValueError("expert_weight_range is obsolete; MoE now uses normalized branch weights.")
        Model.__init__(self, observation_space, action_space, device)
        SquashedGaussianMixin.__init__(self, reduction=kwargs.get("reduction", "sum"))
        if self.num_observations != OBSERVATION_DIM or self.num_actions != ACTION_DIM:
            raise ValueError("Expanding KAE MoE policy requires 15 observations and 3 actions.")

        self.moe_core = ExpandingKaeMoECore(
            kae_model,
            observation_dim=self.num_observations,
            action_dim=self.num_actions,
            steps=int(kwargs.get("kae_steps", KAE_STEPS)),
            pad_value=float(kwargs.get("kae_pad_value", KAE_PAD_VALUE)),
            residual_scale=float(kwargs.get("residual_scale", 1.0)),
            sensitivity_scale=float(kwargs.get("sensitivity_scale", -0.1)),
            step_dt=float(kwargs.get("step_dt", 0.02)),
        )
        initial_log_std = _validate_fixed_log_std(
            kwargs.get("initial_log_std", FIXED_EXPLORATION_LOG_STD),
            self.num_actions,
        ).to(device=self.device)
        self.log_std_parameter = nn.Parameter(initial_log_std, requires_grad=False)

        # Names expected by SKRL's model-instantiation/checkpoint machinery.
        self.net_container = nn.Identity()
        self.policy_layer = self.moe_core

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del role
        mechanism = self.moe_core.mechanism(
            inputs["states"],
            previous_expert_weights=inputs.get("previous_expert_weights"),
        )
        return mechanism.final_mean, self.log_std_parameter, mechanism.policy_outputs()


class BoundedScratchPolicy(SquashedGaussianMixin, Model):
    """Fresh fully trainable PPO actor with bounded actions."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device,
        **kwargs: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        SquashedGaussianMixin.__init__(self, reduction=kwargs.get("reduction", "sum"))
        if self.num_observations != OBSERVATION_DIM or self.num_actions != ACTION_DIM:
            raise ValueError("Bounded scratch policy requires 15 observations and 3 actions.")

        # Separate containers make the final deterministic actor directly
        # exportable as Sequential(net_container, policy_layer).
        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.policy_layer = nn.Sequential(nn.Linear(64, self.num_actions), nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=self.device))

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del role
        mean = self.policy_layer(self.net_container(inputs["states"]))
        return mean, self.log_std_parameter, {"final_mean": mean}


class BoundedValue(DeterministicMixin, Model):
    """Separate critic shared by fresh scratch and fresh MoE PPO stages."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device,
        **kwargs: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=bool(kwargs.get("clip_actions", False)))
        if self.num_observations != OBSERVATION_DIM:
            raise ValueError("Bounded real-world value model requires 15 observations.")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del role
        return self.net(inputs["states"]), {}
