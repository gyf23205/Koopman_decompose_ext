"""Load the paired datasets exported by Isaac and the real-flight SDK.

Pickle inputs must be trusted local exports. Training observations are returned
unchanged: Isaac has already normalized them; SDK observations are raw.
"""

from __future__ import annotations

import copy
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .checkpoints import save_kae_checkpoint
from .models import MAJORUPDATE_OBSERVATION_CONTRACT, PortableKoopmanAutoencoder, RunningStandardScalerModule


@dataclass
class KaeTrainingDataset:
    observations: torch.Tensor
    actions: torch.Tensor
    metadata: dict
    runtime_normalizer: nn.Module

    def padded_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [N, 16] tensors, padding both observations and labels with 1."""
        count = len(self.observations)
        return (
            torch.cat((self.observations, torch.ones(count, 1)), dim=1),
            torch.cat((self.actions, torch.ones(count, 13)), dim=1),
        )


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _matrix(value, width: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must be a Tensor shaped [N, {width}].")
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if not len(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain nonempty, finite float32 data.")
    return value


def _pickle_records(path: Path, width: int) -> list[torch.Tensor]:
    records = []
    with path.open("rb") as stream:
        while stream.peek(1):
            # Checking EOF before loading also makes truncated records fail.
            records.append(_matrix(_CpuUnpickler(stream).load(), width, path.name))
    if not records:
        raise ValueError(f"{path.name} has no samples.")
    return records


def _normalizer(directory: Path, metadata: dict, *, isaac_export: bool) -> nn.Module:
    if not isaac_export:
        if metadata.get("action_target") not in {"full_policy_action", "incremental_residual"}:
            raise ValueError("SDK dataset must declare full_policy_action or incremental_residual targets.")
        return nn.Identity()
    if metadata.get("observations") != "policy input after frozen state preprocessing":
        raise ValueError("Isaac dataset must declare its frozen observation preprocessing.")
    state = torch.load(directory / "state_preprocessor.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("state_preprocessor.pt must contain a state dictionary.")
    if not state and metadata.get("preprocessor_class", "").split(".")[-1] == "Identity":
        return nn.Identity()
    if "running_mean" not in state or "running_variance" not in state:
        raise ValueError("Frozen preprocessing requires running_mean and running_variance.")
    mean = torch.as_tensor(state["running_mean"], dtype=torch.float32)
    variance = torch.as_tensor(state["running_variance"], dtype=torch.float32)
    if mean.shape != (15,) or variance.shape != (15,):
        raise ValueError("Frozen observation normalizer must have 15 features.")
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("Frozen normalizer statistics must be finite with nonnegative variance.")
    options = metadata.get("preprocessor_options", {})
    epsilon = float(options.get("epsilon", 1e-8))
    clip_threshold = float(options.get("clip_threshold", 5.0))
    if not (0 < epsilon < float("inf") and 0 < clip_threshold < float("inf")):
        raise ValueError("Normalizer epsilon and clip threshold must be finite and positive.")
    return RunningStandardScalerModule(mean, variance, epsilon=epsilon, clip_threshold=clip_threshold)


def load_kae_dataset(directory: str | Path) -> KaeTrainingDataset:
    """Load SDK .pt data or paired Tensor pickle records without relabeling."""
    directory = Path(directory).expanduser().resolve()
    metadata_paths = [path for path in (directory / "dataset_metadata.json", directory / "metadata.json") if path.exists()]
    if len(metadata_paths) != 1:
        raise ValueError("Dataset needs exactly one dataset_metadata.json (SDK) or metadata.json (Isaac).")
    metadata = json.loads(metadata_paths[0].read_text())
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be a JSON object.")
    if metadata.get("observation_contract", MAJORUPDATE_OBSERVATION_CONTRACT) != MAJORUPDATE_OBSERVATION_CONTRACT:
        raise ValueError("Dataset observation contract does not match the Crazyflie controller.")
    isaac_export = metadata_paths[0].name == "metadata.json"
    if isaac_export and (metadata.get("obs_dim") != 15 or metadata.get("act_dim") != 3):
        raise ValueError("Isaac data must declare 15 observations and 3 actions.")

    tensor_path = directory / "kae_dataset.pt"
    if not isaac_export and tensor_path.exists():
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        observations = _matrix(payload["observations"], 15, "observations")
        actions = _matrix(payload["kae_training_actions"], 3, "kae_training_actions")
    else:
        inputs = _pickle_records(directory / "obs_log.pkl", 15)
        outputs = _pickle_records(directory / "act_log.pkl", 3)
        if len(inputs) != len(outputs) or any(len(x) != len(y) for x, y in zip(inputs, outputs)):
            raise ValueError("Observation/action pickle records must have aligned batches and row counts.")
        observations, actions = torch.cat(inputs), torch.cat(outputs)
    if len(observations) != len(actions):
        raise ValueError("Observations and action targets must have equal row counts.")
    expected_count = metadata.get("samples" if isaac_export else "sample_count")
    if expected_count is not None and expected_count != len(observations):
        raise ValueError("Dataset sample count does not match its metadata.")
    normalizer = _normalizer(directory, metadata, isaac_export=isaac_export).requires_grad_(False).eval()
    metadata = {
        **metadata,
        "dataset_directory": str(directory),
        "action_target": "full_policy_action" if isaac_export else metadata["action_target"],
        "training_observations": "teacher_normalized" if isaac_export else "raw",
        "sample_count": len(observations),
    }
    return KaeTrainingDataset(observations, actions, metadata, normalizer)


def export_trained_kae(model: nn.Module, dataset: KaeTrainingDataset, output: str | Path, *, metadata=None) -> dict:
    """Check conversion parity, then attach runtime-only normalization and save.

    Teacher-normalized observations may have been clipped, so they cannot be
    reliably inverted. Check stored samples before attaching the normalizer,
    then separately check the raw-input path against explicit preprocessing.
    """
    if int(model.observable_dim) != 16:
        raise ValueError("Each new experiment KAE must contain 16 modes.")
    portable = PortableKoopmanAutoencoder.from_external(model).cpu().eval()
    training_device = next(model.parameters()).device
    inputs, _ = dataset.padded_pairs()
    inputs = inputs[:1024]
    with torch.inference_mode():
        expected = model(inputs.to(training_device))[2].cpu()
        converted = portable(inputs)[2]
        torch.testing.assert_close(converted, expected, rtol=1e-5, atol=1e-6)
        portable.encoder.normalizer = copy.deepcopy(dataset.runtime_normalizer).cpu().eval()
        # Synthetic raw observations test the wiring without pretending to
        # reconstruct pre-clipping teacher observations from the dataset.
        raw = torch.linspace(-6, 6, 32 * 15).reshape(32, 15)
        raw_inputs = torch.cat((raw, torch.ones(len(raw), 1)), dim=1)
        preprocessed = torch.cat((portable.encoder.normalizer(raw), torch.ones(len(raw), 1)), dim=1)
        expected_raw = model(preprocessed.to(training_device))[2].cpu()
        actual_raw = portable(raw_inputs)[2]
        torch.testing.assert_close(actual_raw, expected_raw, rtol=1e-5, atol=1e-6)
    report = {
        "training_space_max_error": float((converted - expected).abs().max()),
        "raw_input_max_error": float((actual_raw - expected_raw).abs().max()),
    }
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_kae_checkpoint(portable, output, metadata={**dataset.metadata, **(metadata or {}), "export_validation": report})
    return report
