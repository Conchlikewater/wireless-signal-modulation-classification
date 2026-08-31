"""Tests for the frozen W2 model matrix."""

import unittest

import torch

from scripts.run_v2_paired_experiments import W2_MATRIX, create_w2_model
from signal_modulation.model import SimpleCNN1D, TemporalCNN1D


class V2RunnerTests(unittest.TestCase):
    def test_matrix_contains_only_pre_registered_a0_and_a1(self) -> None:
        self.assertEqual(
            W2_MATRIX,
            (("A0", "SimpleCNN1D"), ("A1", "TemporalCNN1D")),
        )

    def test_model_factory_preserves_existing_architectures(self) -> None:
        a0 = create_w2_model("A0", num_classes=11)
        a1 = create_w2_model("A1", num_classes=11)

        self.assertIsInstance(a0, SimpleCNN1D)
        self.assertIsInstance(a1, TemporalCNN1D)
        self.assertEqual(a0(torch.randn(2, 2, 128)).shape, (2, 11))
        self.assertEqual(a1(torch.randn(2, 2, 128)).shape, (2, 11))

    def test_unregistered_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported W2"):
            create_w2_model("A2-G", num_classes=11)


if __name__ == "__main__":
    unittest.main()
