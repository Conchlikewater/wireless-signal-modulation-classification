"""Tests for W3 initialization, model scope, and deterministic profiles."""

import unittest

import torch

from signal_modulation.model import (
    GlobalPoolingTemporalCNN1D,
    LSTMTemporalCNN1D,
    TemporalCNN1D,
)
from signal_modulation.w3_ablation import (
    architecture_profile,
    create_w3_model,
    reconstruct_a2t_initial_backbone,
    state_dict_sha256,
    verify_dropout_initialization,
)


class W3AblationTests(unittest.TestCase):
    def test_reconstructed_backbone_is_reproducible_per_seed(self) -> None:
        first, first_hash = reconstruct_a2t_initial_backbone(7, num_classes=11)
        repeated, repeated_hash = reconstruct_a2t_initial_backbone(7, num_classes=11)
        _, changed_hash = reconstruct_a2t_initial_backbone(8, num_classes=11)

        self.assertEqual(first_hash, repeated_hash)
        self.assertNotEqual(first_hash, changed_hash)
        self.assertEqual(state_dict_sha256(first), state_dict_sha256(repeated))

    def test_a2g_loads_the_exact_reconstructed_backbone(self) -> None:
        backbone, expected_hash = reconstruct_a2t_initial_backbone(9, num_classes=11)
        torch.manual_seed(9)
        model = create_w3_model(
            "A2-G",
            num_classes=11,
            initial_backbone=backbone,
        )

        self.assertIsInstance(model, GlobalPoolingTemporalCNN1D)
        self.assertEqual(state_dict_sha256(model.features.state_dict()), expected_hash)

    def test_a3_has_identical_initial_state_to_a1(self) -> None:
        baseline_hash, no_dropout_hash = verify_dropout_initialization(
            10,
            num_classes=11,
        )

        self.assertEqual(baseline_hash, no_dropout_hash)
        self.assertIsInstance(create_w3_model("A3", num_classes=11), TemporalCNN1D)

    def test_profiles_match_frozen_parameter_and_mac_budget(self) -> None:
        a2t = architecture_profile("A2-T")
        a2g = architecture_profile("A2-G")
        a3 = architecture_profile("A3")

        self.assertEqual(a2t["trainable_parameters"], 224_587)
        self.assertEqual(a2g["trainable_parameters"], 224_559)
        self.assertEqual(a2t["head_trainable_parameters"], 132_619)
        self.assertEqual(a2g["head_trainable_parameters"], 132_591)
        self.assertEqual(a2t["conv_linear_macs_per_sample"], 4_441_472)
        self.assertEqual(a2g["conv_linear_macs_per_sample"], 4_440_625)
        self.assertEqual(a3["trainable_parameters"], a2t["trainable_parameters"])
        self.assertEqual(
            a3["conv_linear_macs_per_sample"],
            a2t["conv_linear_macs_per_sample"],
        )

    def test_a2l_loads_shared_backbone_and_matches_capacity_profile(self) -> None:
        backbone, expected_hash = reconstruct_a2t_initial_backbone(9, num_classes=11)
        model = create_w3_model(
            "A2-L",
            num_classes=11,
            initial_backbone=backbone,
        )
        profile = architecture_profile("A2-L")

        self.assertIsInstance(model, LSTMTemporalCNN1D)
        self.assertEqual(state_dict_sha256(model.features.state_dict()), expected_hash)
        self.assertEqual(profile["trainable_parameters"], 223_932)
        self.assertEqual(profile["head_trainable_parameters"], 131_964)
        self.assertEqual(profile["lstm_matrix_macs_per_sample"], 4_145_280)
        self.assertEqual(profile["total_estimated_macs_per_sample"], 8_455_669)

    def test_unregistered_w3_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported W3"):
            create_w3_model("A0", num_classes=11)


if __name__ == "__main__":
    unittest.main()
