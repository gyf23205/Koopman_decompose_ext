"""CPU checks for this repository's local KAE MoE runtime."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from kae_moe import (
    ExpandingKaeMoECore, PortableKoopmanAutoencoder, RunningStandardScalerModule,
    StackedPortableKAE, load_kae_checkpoint, load_moe_checkpoint,
    save_kae_checkpoint, save_moe_checkpoint,
)
from kae_moe.models import (
    FIXED_EXPLORATION_LOG_STD, HierarchicalMoEHead, bounded_action_sample,
    koopman_expert_actions,
)


class SharedRuntimeTests(unittest.TestCase):
    def test_runtime_imports_from_this_repository(self):
        import kae_moe
        import kae_moe.checkpoints
        import kae_moe.models

        package = Path(__file__).resolve().parents[1] / "kae_moe"
        for module in (kae_moe, kae_moe.checkpoints, kae_moe.models):
            self.assertEqual(Path(module.__file__).resolve().parent, package)

    def setUp(self):
        torch.manual_seed(31)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def block(self, offset=0, normalized=False):
        block = PortableKoopmanAutoencoder(
            16, koopman_matrix=torch.diag(torch.linspace(0.7, 1.05, 16) + offset * 0.01),
        )
        if normalized:
            block.encoder.normalizer = RunningStandardScalerModule(
                torch.linspace(-1, 1, 15), torch.linspace(0.1, 2, 15),
            )
        return block.freeze()

    def core(self, count):
        return ExpandingKaeMoECore(StackedPortableKAE(
            [self.block(index, normalized=index == 0) for index in range(count)],
            [f"task-{index}" for index in range(count)],
        ))

    def test_pi_initialization_routing_blend_and_gradients(self):
        for count in (1, 2, 4):
            with self.subTest(branches=count):
                head = HierarchicalMoEHead(15, 3, [16] * count)
                observations = torch.randn(8, 15)
                experts = torch.randn(8, 16 * count, 3, dtype=torch.float64)
                router = torch.softmax(head.kae_router_network(observations), dim=1)
                expected_weights = torch.cat([
                    router[:, index:index + 1].expand(-1, 16) / (16 + 1e-6)
                    for index in range(count)
                ], dim=1)
                actual = head(observations, experts)
                torch.testing.assert_close(actual["expert_weights"], expected_weights)
                torch.testing.assert_close(actual["gate"], torch.full((8, 1), torch.sigmoid(torch.tensor(1.0))))
                expected_kae = (expected_weights.double().unsqueeze(-1) * experts).sum(1).float()
                torch.testing.assert_close(actual["kae_action"], expected_kae)
                torch.testing.assert_close(actual["final_mean"],
                    actual["gate"] * expected_kae + (1 - actual["gate"]) * head.mlp_network(observations))
                actual["final_mean"].square().sum().backward()
                for network in [head.mlp_network, head.gating_network, *head.expert_weight_networks]:
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in network.parameters()), 0)
                router_grad = sum(p.grad.abs().sum().item() for p in head.kae_router_network.parameters())
                self.assertEqual(router_grad == 0, count == 1)

    def test_signed_l1_weights_and_action_scales(self):
        head = HierarchicalMoEHead(15, 3, [16])
        with torch.no_grad():
            head.expert_weight_networks[0][-1].bias.copy_(torch.linspace(-2, 1, 16))
        obs = torch.randn(4, 15)
        experts = torch.randn(4, 16, 3, dtype=torch.float64)
        raw = head.expert_weight_networks[0](obs)
        weights = raw / (raw.abs().sum(dim=1, keepdim=True) + 1e-6)
        out = head(obs, experts, kae_action_scale=2, mlp_action_scale=0.3)
        torch.testing.assert_close(out["expert_weights"], weights)
        torch.testing.assert_close(out["kae_action"], 2 * (weights.double().unsqueeze(-1) * experts).sum(1).float())
        torch.testing.assert_close(out["residual_action"], 0.3 * head.mlp_network(obs))
        self.assertTrue((weights < 0).any())

    def test_normalizer_once_padding_untouched_full_sum(self):
        block = self.block(normalized=True)
        obs = torch.randn(5, 15) * 100
        padded = torch.cat((obs, torch.ones(5, 1)), dim=1)
        normalized = ((obs - block.encoder.normalizer.running_mean) /
                      (block.encoder.normalizer.running_variance.sqrt() + 1e-8)).clamp(-5, 5)
        expected = block.encoder.encoder(torch.cat((normalized, torch.ones(5, 1)), dim=1))
        actual = block.encoder(padded)
        torch.testing.assert_close(actual, expected)
        modes = koopman_expert_actions(block, actual)
        torch.testing.assert_close(modes.sum(1).float(), block(padded)[2][:, :3], rtol=1e-5, atol=1e-6)

    def test_policy_roundtrip_preserves_trained_head_basis_order_and_freezing(self):
        obs = torch.randn(11, 15)
        for count in (1, 2, 4):
            with self.subTest(branches=count):
                core = self.core(count)
                with torch.no_grad():
                    for parameter in core.head.parameters():
                        parameter.add_(0.05 * torch.randn_like(parameter))
                core.residual_scale = 0.8
                path = self.path / f"moe-{count}.pt"
                expected = core(obs)
                save_moe_checkpoint(core, path, critic_state={"weight": torch.randn(3, 3)},
                                    value_preprocessor_state={"running_mean": torch.tensor(0.2)},
                                    metadata={"source": "square", "inputs": list(core.kae.block_origins)})
                with patch("torch.linalg.eig", side_effect=AssertionError("loading must not recompute modes")):
                    loaded, extras = load_moe_checkpoint(path)
                torch.testing.assert_close(loaded(obs), expected, rtol=0, atol=0)
                self.assertEqual(loaded.kae.block_origins, core.kae.block_origins)
                self.assertEqual(loaded.kae.mode_slices, core.kae.mode_slices)
                self.assertEqual(extras["metadata"]["source"], "square")
                self.assertIsNotNone(extras["critic_state"])
                self.assertIsNotNone(extras["value_preprocessor_state"])
                for original, restored in zip(core.kae.blocks, loaded.kae.blocks):
                    for a, b in zip(original.spectral_components(), restored.spectral_components()):
                        self.assertTrue(torch.equal(a, b))
                loaded.train()
                loaded(obs).square().sum().backward()
                self.assertTrue(all(parameter.grad is None and not parameter.requires_grad
                                    for parameter in loaded.kae.parameters()))
                self.assertFalse(loaded.kae.training)
                self.assertTrue(all(not block.training for block in loaded.kae.blocks))
                self.assertTrue(any(parameter.grad is not None for parameter in loaded.head.parameters()))

    def test_kae_roundtrip_and_external_conversion_keep_basis(self):
        block = self.block(normalized=True)
        path = self.path / "kae.pt"
        save_kae_checkpoint(block, path, metadata={"target": "full_policy"})
        with patch("torch.linalg.eig", side_effect=AssertionError("must preserve captured basis")):
            loaded = load_kae_checkpoint(path)
            copied = PortableKoopmanAutoencoder.from_external(loaded)
        inputs = torch.randn(8, 16)
        for expected, actual in zip(block(inputs), copied(inputs)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIn("K", loaded.state_dict())
        self.assertEqual(loaded.encoder.normalizer.clip_threshold, 5)
        payload = torch.load(path, weights_only=True)
        self.assertEqual(payload["format"], "kae_moe.kae.v1")
        self.assertTrue(all(value.device.type == "cpu" for value in payload["kae"]["state_dict"].values()))

    def test_reject_bad_contract_state_normalizer_and_spectra(self):
        path = self.path / "good.pt"
        save_kae_checkpoint(self.block(normalized=True), path)
        invalid = self.path / "invalid.pt"
        for field in ("contract", "variance", "nan", "basis", "shape"):
            with self.subTest(field=field):
                payload = torch.load(path, weights_only=True)
                if field == "contract":
                    payload["contract"]["action_type"] = "relative"
                elif field == "variance":
                    payload["kae"]["normalizer"]["running_variance"][0] = -1
                elif field == "nan":
                    payload["kae"]["state_dict"]["K"][0, 0] = torch.nan
                elif field == "basis":
                    payload["kae"]["state_dict"]["_captured_eigenvalues"][0] += 1
                else:
                    payload["kae"]["state_dict"]["decoder.linear.weight"] = torch.zeros(3, 3)
                torch.save(payload, invalid)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_kae_checkpoint(invalid)

    def test_bounded_distribution_and_core_boundary(self):
        core = self.core(1)
        with torch.no_grad():
            core.head.mlp_network[-1].bias.fill_(100)
        means = core(torch.zeros(4, 15))
        torch.testing.assert_close(means, torch.ones_like(means))
        sample, log_prob, _ = bounded_action_sample(means)
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue((sample.abs() <= 1).all())
        deterministic, _, _ = bounded_action_sample(means, deterministic=True)
        self.assertTrue(torch.equal(deterministic, means))

    @unittest.skipUnless(importlib.util.find_spec("skrl"), "SKRL adapter is optional")
    def test_skrl_policy_reports_weights_and_matches_core(self):
        from kae_moe.skrl_models import ExpandingKaeMoEPolicy
        for count in (1, 2, 4):
            with self.subTest(branches=count):
                core = self.core(count)
                policy = ExpandingKaeMoEPolicy(15, 3, "cpu", core.kae)
                policy.moe_core.load_state_dict(core.state_dict())
                obs = torch.randn(5, 15)
                means, _, outputs = policy.deterministic_act({"states": obs})
                torch.testing.assert_close(means, core(obs), rtol=0, atol=0)
                torch.testing.assert_close(outputs["expert_weights"], core.expert_weights(obs))
                self.assertEqual(outputs["expert_weights"].shape, (5, 16 * count))
                self.assertFalse(policy.log_std_parameter.requires_grad)
                torch.testing.assert_close(policy.log_std_parameter, torch.tensor(FIXED_EXPLORATION_LOG_STD))


if __name__ == "__main__":
    unittest.main()
