"""Frozen Wireless V2 settings with separate split and run randomness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from signal_modulation.reproducibility import configure_reproducibility
from signal_modulation.split_manifest import (
    DevelopmentLoaders,
    create_development_loaders,
    load_frozen_development_split,
)


V2_PROTOCOL_VERSION = "wireless-v2-w0-v1"
V2_DATASET_SHA256 = (
    "b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c"
)
V2_SAMPLE_COUNT = 220_000
V2_SPLIT_SEED = 20260812
V2_RUN_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)


@dataclass(frozen=True, slots=True)
class V2ExperimentConfig:
    """Pre-registered V2 configuration; split_seed and run_seed have distinct roles."""

    split_seed: int = V2_SPLIT_SEED
    run_seed: int = V2_RUN_SEEDS[0]
    epochs: int = 20
    patience: int = 4
    min_delta: float = 0.0001
    learning_rate: float = 0.001
    train_batch_size: int = 256
    evaluation_batch_size: int = 512
    num_workers: int = 0
    dropout: float = 0.3
    use_amp: bool = False
    use_scheduler: bool = False

    def __post_init__(self) -> None:
        if self.split_seed != V2_SPLIT_SEED:
            raise ValueError(
                f"split_seed is frozen at {V2_SPLIT_SEED} for Wireless V2"
            )
        if self.run_seed not in V2_RUN_SEEDS:
            raise ValueError(f"run_seed must be one of the frozen seeds: {V2_RUN_SEEDS}")
        positive_integers = {
            "epochs": self.epochs,
            "patience": self.patience,
            "train_batch_size": self.train_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and greater than zero")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.use_amp:
            raise ValueError("AMP is disabled by the frozen V2 protocol")
        if self.use_scheduler:
            raise ValueError("a learning-rate scheduler is disabled by the V2 protocol")


def configure_v2_run_randomness(config: V2ExperimentConfig) -> None:
    """Seed model initialization, Dropout and training randomness with run_seed only."""

    configure_reproducibility(config.run_seed)


def create_v2_development_loaders(
    dataset: Dataset,
    manifest_path: str | Path,
    config: V2ExperimentConfig,
    *,
    dataset_sha256: str,
    pin_memory: bool = False,
) -> DevelopmentLoaders:
    """Load the frozen split and expose only train/validation DataLoaders."""

    if dataset_sha256.lower() != V2_DATASET_SHA256:
        raise ValueError("dataset SHA-256 does not match the frozen V2 protocol")
    split = load_frozen_development_split(
        manifest_path,
        expected_dataset_sha256=V2_DATASET_SHA256,
        expected_split_seed=config.split_seed,
        expected_protocol_version=V2_PROTOCOL_VERSION,
        expected_sample_count=V2_SAMPLE_COUNT,
    )
    return create_development_loaders(
        dataset,
        split,
        run_seed=config.run_seed,
        train_batch_size=config.train_batch_size,
        evaluation_batch_size=config.evaluation_batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
