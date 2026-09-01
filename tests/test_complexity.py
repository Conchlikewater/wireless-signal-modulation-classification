"""Tests for the frozen W3 parameter and MAC conventions."""

import unittest

from signal_modulation.complexity import estimate_conv_linear_macs
from signal_modulation.model import (
    GlobalPoolingTemporalCNN1D,
    TemporalCNN1D,
    count_trainable_parameters,
)


class ComplexityTests(unittest.TestCase):
    def test_a2_parameter_counts_and_macs_are_capacity_controlled(self) -> None:
        temporal = TemporalCNN1D(num_classes=11)
        global_pooling = GlobalPoolingTemporalCNN1D(num_classes=11)

        self.assertEqual(count_trainable_parameters(temporal.classifier), 132_619)
        self.assertEqual(count_trainable_parameters(global_pooling.classifier), 132_591)
        self.assertEqual(count_trainable_parameters(temporal), 224_587)
        self.assertEqual(count_trainable_parameters(global_pooling), 224_559)
        self.assertEqual(estimate_conv_linear_macs(temporal), 4_441_472)
        self.assertEqual(estimate_conv_linear_macs(global_pooling), 4_440_625)

    def test_dropout_change_does_not_change_parameters_or_macs(self) -> None:
        with_dropout = TemporalCNN1D(num_classes=11, dropout=0.3)
        without_dropout = TemporalCNN1D(num_classes=11, dropout=0.0)

        self.assertEqual(
            count_trainable_parameters(with_dropout),
            count_trainable_parameters(without_dropout),
        )
        self.assertEqual(
            estimate_conv_linear_macs(with_dropout),
            estimate_conv_linear_macs(without_dropout),
        )

    def test_invalid_input_shape_for_macs_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_conv_linear_macs(TemporalCNN1D(num_classes=11), channels=0)


if __name__ == "__main__":
    unittest.main()
