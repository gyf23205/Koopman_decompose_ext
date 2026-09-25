import importlib.util
import json
import pickle
from pathlib import Path

import tempfile
import unittest
import torch
from torch import nn

from kae_moe.checkpoints import load_kae_checkpoint
from kae_moe.dataset import export_trained_kae, load_kae_dataset
from kae_moe.models import RunningStandardScalerModule


def write_records(path, records):
    with path.open("wb") as stream:
        for record in records:
            pickle.dump(record, stream)


def sdk_dataset(path, count=7, target="incremental_residual"):
    observations = torch.randn(count, 15)
    actions = torch.full((count, 3), 4.25)  # Corrections must not be clipped.
    (path / "dataset_metadata.json").write_text(json.dumps({
        "action_target": target, "sample_count": count,
    }))
    write_records(path / "obs_log.pkl", [observations[:3], observations[3:]])
    write_records(path / "act_log.pkl", [actions[:3], actions[3:]])
    return observations, actions


def isaac_dataset(path):
    normalized = torch.full((9, 15), 2.0)
    actions = torch.randn(9, 3)
    (path / "metadata.json").write_text(json.dumps({
        "obs_dim": 15, "act_dim": 3, "samples": 9,
        "observations": "policy input after frozen state preprocessing",
        "preprocessor_class": "skrl.resources.preprocessors.torch.RunningStandardScaler",
        "preprocessor_options": {"epsilon": 1e-7, "clip_threshold": 3.0},
    }))
    torch.save({"running_mean": torch.ones(15), "running_variance": torch.full((15,), 4.0)}, path / "state_preprocessor.pt")
    write_records(path / "obs_log.pkl", [normalized])
    write_records(path / "act_log.pkl", [actions])
    return normalized, actions


class KaeDatasetTests(unittest.TestCase):
    def test_sdk_raw_inputs_and_unclipped_labels(self):
        for target in ['full_policy_action', 'incremental_residual']:
            with self.subTest(target=target):
                with tempfile.TemporaryDirectory() as directory:
                    tmp_path = Path(directory)
                    observations, actions = sdk_dataset(tmp_path, target=target)
                    data = load_kae_dataset(tmp_path)
                    torch.testing.assert_close(data.observations, observations)
                    torch.testing.assert_close(data.actions, actions)
                    assert isinstance(data.runtime_normalizer, nn.Identity)
                    assert data.metadata["action_target"] == target
                    inputs, targets = data.padded_pairs()
                    assert inputs.shape == targets.shape == (7, 16)
                    torch.testing.assert_close(inputs[:, -1], torch.ones(7))
                    torch.testing.assert_close(targets[:, 3:], torch.ones(7, 13))

    def test_isaac_inputs_not_normalized_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            observations, actions = isaac_dataset(tmp_path)
            data = load_kae_dataset(tmp_path)
            torch.testing.assert_close(data.observations, observations)
            torch.testing.assert_close(data.actions, actions)
            assert isinstance(data.runtime_normalizer, RunningStandardScalerModule)
            torch.testing.assert_close(data.runtime_normalizer(torch.ones(2, 15) * 5), torch.ones(2, 15) * 2)
            assert data.runtime_normalizer.epsilon == 1e-7
            assert data.metadata["training_observations"] == "teacher_normalized"

    def test_sdk_tensor_file_uses_correction_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            observations, actions = sdk_dataset(tmp_path)
            torch.save({"observations": observations, "actions": torch.zeros_like(actions), "kae_training_actions": actions}, tmp_path / "kae_dataset.pt")
            data = load_kae_dataset(tmp_path)
            torch.testing.assert_close(data.actions, actions)

    def test_rejects_invalid_pairs(self):
        for bad in ['count', 'batch', 'shape', 'nan', 'empty', 'metadata']:
            with self.subTest(bad=bad):
                with tempfile.TemporaryDirectory() as directory:
                    tmp_path = Path(directory)
                    observations, actions = sdk_dataset(tmp_path)
                    if bad == "count":
                        write_records(tmp_path / "act_log.pkl", [actions[:3]])
                    elif bad == "batch":
                        write_records(tmp_path / "act_log.pkl", [actions[:4], actions[4:]])
                    elif bad == "shape":
                        write_records(tmp_path / "obs_log.pkl", [torch.ones(7, 14)])
                    elif bad == "nan":
                        observations[0, 0] = float("nan")
                        write_records(tmp_path / "obs_log.pkl", [observations[:3], observations[3:]])
                    elif bad == "empty":
                        write_records(tmp_path / "obs_log.pkl", [])
                    else:
                        (tmp_path / "dataset_metadata.json").write_text(json.dumps({"action_target": "incremental_residual", "sample_count": 100}))
                    with self.assertRaises(ValueError):
                        load_kae_dataset(tmp_path)

    def test_rejects_missing_or_invalid_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            isaac_dataset(tmp_path)
            torch.save({}, tmp_path / "state_preprocessor.pt")
            with self.assertRaisesRegex(ValueError, "running_mean"):
                load_kae_dataset(tmp_path)
            torch.save({"running_mean": torch.ones(15), "running_variance": -torch.ones(15)}, tmp_path / "state_preprocessor.pt")
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                load_kae_dataset(tmp_path)

    def test_portable_export_includes_k_and_runtime_normalizer(self):
        for normalized in [False, True]:
            with self.subTest(normalized=normalized):
                with tempfile.TemporaryDirectory() as directory:
                    tmp_path = Path(directory)
                    (isaac_dataset if normalized else sdk_dataset)(tmp_path)
                    data = load_kae_dataset(tmp_path)
                    source_path = Path(__file__).parents[1] / "KAE" / "Autoencoder.py"
                    spec = importlib.util.spec_from_file_location("notebook_autoencoder", source_path)
                    source = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(source)
                    model = source.KoopmanAutoencoder_walk(16, 64, 16, "cpu")
                    # Fitted K can carry an autograd graph; exporting must not deepcopy it.
                    model.K = torch.diag(torch.linspace(0.5, 1.5, 16)).requires_grad_(True) * 1.0
                    assert "K" not in model.state_dict()
                    output = tmp_path / "portable.pt"
                    report = export_trained_kae(model, data, output)
                    portable = load_kae_checkpoint(output)
                    torch.testing.assert_close(portable.K, model.K)
                    assert "K" in portable.state_dict()
                    assert report["raw_input_max_error"] < 1e-6
                    assert report["training_space_max_error"] < 1e-6
                    assert not any(parameter.requires_grad for parameter in portable.parameters())
                    assert isinstance(model.encoder, source.Encoder_walk)
                    assert not hasattr(model.encoder, "normalizer")
                    observations = torch.randn(5, 15)
                    padded = torch.cat((observations, torch.ones(5, 1)), dim=1)
                    training_space = torch.cat((data.runtime_normalizer(observations), torch.ones(5, 1)), dim=1)
                    torch.testing.assert_close(portable(padded)[2], model(training_space)[2])


if __name__ == "__main__":
    unittest.main()
