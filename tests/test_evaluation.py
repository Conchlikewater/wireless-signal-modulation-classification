"""Tests for dependency-light classification evaluation."""

import unittest

import numpy as np
import torch
from torch import nn

from signal_modulation.dataset import IQSignalDataset, create_data_loader
from signal_modulation.evaluation import (
    calculate_classification_metrics,
    calculate_snr_metrics,
    evaluate_classifier,
)


class _SignalAsLogits(nn.Module):
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return signals[:, 0, :2]


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_matrix_and_macro_metrics(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        predictions = torch.tensor([0, 1, 1, 1])

        metrics = calculate_classification_metrics(
            labels, predictions, num_classes=2
        )

        self.assertEqual(metrics.confusion_matrix, ((1, 1), (0, 2)))
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        self.assertAlmostEqual(metrics.macro_precision, (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(metrics.macro_recall, 0.75)
        self.assertAlmostEqual(metrics.macro_f1, (2.0 / 3.0 + 0.8) / 2.0)

    def test_snr_metrics_separate_signal_conditions(self) -> None:
        labels = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([0, 0, 0, 1])
        snrs = torch.tensor([-10.0, -10.0, 10.0, 10.0])

        results = calculate_snr_metrics(labels, predictions, snrs)

        self.assertEqual([result.snr for result in results], [-10.0, 10.0])
        self.assertEqual([result.sample_count for result in results], [2, 2])
        self.assertEqual([result.accuracy for result in results], [0.5, 1.0])
        self.assertAlmostEqual(results[0].macro_f1, 1.0 / 3.0)
        self.assertAlmostEqual(results[1].macro_f1, 1.0)

    def test_invalid_class_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid class"):
            calculate_classification_metrics(
                torch.tensor([0, 2]),
                torch.tensor([0, 1]),
                num_classes=2,
            )

    def test_model_evaluation_collects_loss_predictions_and_snr(self) -> None:
        signals = np.zeros((4, 2, 2), dtype=np.float32)
        signals[:, 0, :] = np.asarray(
            [[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
        snrs = np.asarray([-10, -10, 10, 10], dtype=np.float32)
        loader = create_data_loader(
            IQSignalDataset(signals, labels, snrs),
            batch_size=2,
            shuffle=False,
        )

        result = evaluate_classifier(
            _SignalAsLogits(),
            loader,
            device=torch.device("cpu"),
            num_classes=2,
        )

        self.assertEqual(result.sample_count, 4)
        self.assertAlmostEqual(result.classification.accuracy, 1.0)
        self.assertAlmostEqual(result.classification.macro_f1, 1.0)
        self.assertEqual(len(result.by_snr), 2)

    def test_empty_evaluation_loader_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no evaluation samples"):
            evaluate_classifier(
                _SignalAsLogits(),
                [],
                device=torch.device("cpu"),
                num_classes=2,
            )


if __name__ == "__main__":
    unittest.main()
