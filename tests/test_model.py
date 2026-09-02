"""Tests for the first 1D CNN baseline."""

import unittest

import torch

from signal_modulation.model import (
    GlobalPoolingTemporalCNN1D,
    LSTMTemporalCNN1D,
    SimpleCNN1D,
    TemporalCNN1D,
    count_trainable_parameters,
)


class SimpleCNN1DTests(unittest.TestCase):
    def test_output_shape_matches_batch_and_class_count(self) -> None:
        model = SimpleCNN1D(num_classes=11)
        inputs = torch.randn(8, 2, 128)

        logits = model(inputs)

        self.assertEqual(logits.shape, (8, 11))

    def test_model_accepts_a_different_sequence_length(self) -> None:
        model = SimpleCNN1D(num_classes=4)

        logits = model(torch.randn(3, 2, 64))

        self.assertEqual(logits.shape, (3, 4))

    def test_invalid_channel_shape_is_rejected(self) -> None:
        model = SimpleCNN1D(num_classes=3)

        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(4, 1, 128))

    def test_invalid_class_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimpleCNN1D(num_classes=1)

    def test_cross_entropy_backward_produces_gradients(self) -> None:
        torch.manual_seed(42)
        model = SimpleCNN1D(num_classes=3)
        inputs = torch.randn(6, 2, 128)
        labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)

        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertGreater(float(model.classifier.weight.grad.abs().sum()), 0.0)

    def test_parameter_count_is_small_and_non_zero(self) -> None:
        model = SimpleCNN1D(num_classes=11)

        parameter_count = count_trainable_parameters(model)

        self.assertEqual(parameter_count, 11_499)


class TemporalCNN1DTests(unittest.TestCase):
    def test_output_shape_matches_batch_and_class_count(self) -> None:
        model = TemporalCNN1D(num_classes=11)

        logits = model(torch.randn(8, 2, 128))

        self.assertEqual(logits.shape, (8, 11))

    def test_model_rejects_a_different_sequence_length(self) -> None:
        model = TemporalCNN1D(num_classes=4)

        with self.assertRaisesRegex(ValueError, "configured sequence length"):
            model(torch.randn(3, 2, 64))

    def test_invalid_temporal_bin_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TemporalCNN1D(num_classes=3, temporal_bins=0)

    def test_too_short_sequence_is_rejected(self) -> None:
        model = TemporalCNN1D(num_classes=3)

        with self.assertRaisesRegex(ValueError, "length"):
            model(torch.randn(4, 2, 3))

    def test_incompatible_sequence_length_and_bins_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide"):
            TemporalCNN1D(num_classes=3, sequence_length=126, temporal_bins=8)

    def test_cross_entropy_backward_produces_gradients(self) -> None:
        torch.manual_seed(42)
        model = TemporalCNN1D(num_classes=3)
        inputs = torch.randn(6, 2, 128)
        labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)

        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        final_layer = model.classifier[-1]
        self.assertIsInstance(final_layer, torch.nn.Linear)
        self.assertIsNotNone(final_layer.weight.grad)
        self.assertGreater(float(final_layer.weight.grad.abs().sum()), 0.0)

    def test_parameter_count_is_larger_but_still_compact(self) -> None:
        model = TemporalCNN1D(num_classes=11)

        parameter_count = count_trainable_parameters(model)

        self.assertEqual(parameter_count, 224_587)

    def test_dropout_ablation_preserves_parameter_initialization(self) -> None:
        torch.manual_seed(123)
        baseline = TemporalCNN1D(num_classes=11, dropout=0.3)
        torch.manual_seed(123)
        no_dropout = TemporalCNN1D(num_classes=11, dropout=0.0)

        self.assertEqual(baseline.dropout, 0.3)
        self.assertEqual(no_dropout.dropout, 0.0)
        for name, baseline_value in baseline.state_dict().items():
            torch.testing.assert_close(baseline_value, no_dropout.state_dict()[name])

    def test_invalid_dropout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dropout"):
            TemporalCNN1D(num_classes=11, dropout=1.0)


class GlobalPoolingTemporalCNN1DTests(unittest.TestCase):
    def test_output_and_backbone_shapes_match_frozen_a2g_design(self) -> None:
        model = GlobalPoolingTemporalCNN1D(num_classes=11)
        inputs = torch.randn(4, 2, 128)

        backbone = model.features(inputs)
        pooled = model.aggregation(backbone)
        logits = model(inputs)

        self.assertEqual(backbone.shape, (4, 128, 32))
        self.assertEqual(pooled.shape, (4, 128, 1))
        self.assertEqual(logits.shape, (4, 11))

    def test_backbone_initialization_matches_temporal_cnn_for_same_seed(self) -> None:
        torch.manual_seed(456)
        temporal = TemporalCNN1D(num_classes=11)
        torch.manual_seed(456)
        global_pooling = GlobalPoolingTemporalCNN1D(num_classes=11)

        temporal_backbone = temporal.features[:-1].state_dict()
        global_backbone = global_pooling.features.state_dict()
        self.assertEqual(temporal_backbone.keys(), global_backbone.keys())
        for name, temporal_value in temporal_backbone.items():
            torch.testing.assert_close(temporal_value, global_backbone[name])

    def test_parameter_budget_matches_protocol(self) -> None:
        model = GlobalPoolingTemporalCNN1D(num_classes=11)

        self.assertEqual(count_trainable_parameters(model.classifier), 132_591)
        self.assertEqual(count_trainable_parameters(model), 224_559)
        self.assertLess(
            abs(224_587 - count_trainable_parameters(model)) / 224_587,
            0.01,
        )

    def test_non_frozen_sequence_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence_length=128"):
            GlobalPoolingTemporalCNN1D(num_classes=11, sequence_length=64)


class LSTMTemporalCNN1DTests(unittest.TestCase):
    def test_lstm_receives_32_steps_and_returns_class_logits(self) -> None:
        model = LSTMTemporalCNN1D(num_classes=11)
        observed: list[tuple[int, ...]] = []
        handle = model.aggregation.register_forward_pre_hook(
            lambda _module, inputs: observed.append(tuple(inputs[0].shape))
        )
        try:
            logits = model(torch.randn(4, 2, 128))
        finally:
            handle.remove()

        self.assertEqual(observed, [(4, 32, 128)])
        self.assertEqual(logits.shape, (4, 11))

    def test_a2l_matches_preregistered_capacity_and_keeps_backbone_trainable(self) -> None:
        model = LSTMTemporalCNN1D(num_classes=11)
        loss = model(torch.randn(2, 2, 128)).sum()
        loss.backward()

        self.assertEqual(count_trainable_parameters(model.features), 91_968)
        self.assertEqual(count_trainable_parameters(model.aggregation), 130_556)
        self.assertEqual(count_trainable_parameters(model.classifier), 1_408)
        self.assertEqual(count_trainable_parameters(model), 223_932)
        self.assertAlmostEqual(abs(224_587 - 223_932) / 224_587, 0.002916, places=6)
        self.assertIsNotNone(model.features[0].weight.grad)
        self.assertIsNotNone(model.aggregation.weight_ih_l0.grad)
        self.assertEqual(model.classifier[0].p, 0.3)

    def test_non_frozen_a2l_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence_length=128"):
            LSTMTemporalCNN1D(num_classes=11, sequence_length=64)
        with self.assertRaisesRegex(ValueError, "hidden_size=127"):
            LSTMTemporalCNN1D(num_classes=11, hidden_size=128)


if __name__ == "__main__":
    unittest.main()
