"""Tests for the minimal one-window inference demo."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.predict_single import CLASSES, DEFAULT_CHECKPOINT, load_signal, predict
from signal_modulation.model import TemporalCNN1D


class PredictSingleTests(unittest.TestCase):
    def test_load_predict_and_confidence_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal_path = root / "signal.npy"
            checkpoint_path = root / "checkpoint.pt"
            np.save(signal_path, np.zeros((2, 128), dtype=np.float32))
            model = TemporalCNN1D(num_classes=len(CLASSES))
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            signal = load_signal(signal_path)
            label, confidence = predict(signal, checkpoint_path)

        self.assertEqual(tuple(signal.shape), (1, 2, 128))
        self.assertIn(label, CLASSES)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_invalid_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npy"
            np.save(path, np.zeros((128,), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "shape"):
                load_signal(path)

    def test_repository_default_checkpoint_is_loadable(self) -> None:
        label, confidence = predict(
            torch.zeros((1, 2, 128), dtype=torch.float32),
            DEFAULT_CHECKPOINT,
        )

        self.assertIn(label, CLASSES)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
