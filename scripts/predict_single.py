"""Predict one saved float32 I/Q window with the default A1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from signal_modulation.model import TemporalCNN1D


CLASSES = (
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
)
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "experiments/v2/w2/A1/run_seed_20260901/best_checkpoint.pt"
)


def load_signal(path: Path) -> torch.Tensor:
    signal = np.load(path, allow_pickle=False)
    if signal.shape != (2, 128) or not np.issubdtype(signal.dtype, np.number):
        raise ValueError("input must be a numeric NumPy array with shape (2, 128)")
    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError("input contains NaN or infinity")
    return torch.from_numpy(signal).unsqueeze(0)


def predict(signal: torch.Tensor, checkpoint_path: Path) -> tuple[str, float]:
    model = TemporalCNN1D(num_classes=len(CLASSES))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(signal), dim=1)[0]
    confidence, class_index = probabilities.max(dim=0)
    return CLASSES[int(class_index)], float(confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signal", type=Path, help=".npy file with shape (2, 128)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    label, confidence = predict(load_signal(args.signal), args.checkpoint)
    print(f"predicted_class={label}")
    print(f"confidence={confidence:.6f}")


if __name__ == "__main__":
    main()
