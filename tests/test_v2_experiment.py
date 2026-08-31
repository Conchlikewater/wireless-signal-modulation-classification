"""Tests for the pre-registered Wireless V2 seed roles."""

import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_RUN_SEEDS,
    V2_SPLIT_SEED,
    V2ExperimentConfig,
    configure_v2_run_randomness,
    create_v2_development_loaders,
)


class V2ExperimentTests(unittest.TestCase):
    def test_all_pre_registered_run_seeds_are_accepted(self) -> None:
        self.assertEqual(len(V2_RUN_SEEDS), 5)
        for run_seed in V2_RUN_SEEDS:
            config = V2ExperimentConfig(run_seed=run_seed)
            self.assertEqual(config.split_seed, V2_SPLIT_SEED)
            self.assertEqual(config.run_seed, run_seed)

    def test_split_seed_and_unknown_run_seed_cannot_be_changed(self) -> None:
        with self.assertRaisesRegex(ValueError, "split_seed is frozen"):
            V2ExperimentConfig(split_seed=1)
        with self.assertRaisesRegex(ValueError, "run_seed must be one"):
            V2ExperimentConfig(run_seed=17)

    @patch("signal_modulation.v2_experiment.configure_reproducibility")
    def test_run_randomness_uses_run_seed_not_split_seed(self, configure) -> None:
        config = V2ExperimentConfig(run_seed=V2_RUN_SEEDS[-1])

        configure_v2_run_randomness(config)

        configure.assert_called_once_with(V2_RUN_SEEDS[-1])
        self.assertNotEqual(config.run_seed, config.split_seed)

    @patch("signal_modulation.v2_experiment.create_development_loaders")
    @patch("signal_modulation.v2_experiment.load_frozen_development_split")
    def test_v2_loader_uses_split_seed_for_manifest_and_run_seed_for_order(
        self,
        load_split,
        create_loaders,
    ) -> None:
        load_split.return_value = sentinel.frozen_split
        create_loaders.return_value = sentinel.loaders
        config = V2ExperimentConfig(run_seed=V2_RUN_SEEDS[2])

        result = create_v2_development_loaders(
            sentinel.dataset,
            Path("split_manifest.json"),
            config,
            dataset_sha256=V2_DATASET_SHA256,
        )

        self.assertIs(result, sentinel.loaders)
        load_split.assert_called_once_with(
            Path("split_manifest.json"),
            expected_dataset_sha256=V2_DATASET_SHA256,
            expected_split_seed=V2_SPLIT_SEED,
            expected_protocol_version="wireless-v2-w0-v1",
            expected_sample_count=220_000,
        )
        create_loaders.assert_called_once_with(
            sentinel.dataset,
            sentinel.frozen_split,
            run_seed=V2_RUN_SEEDS[2],
            train_batch_size=256,
            evaluation_batch_size=512,
            num_workers=0,
            pin_memory=False,
        )

    def test_v2_loader_rejects_another_dataset_identity(self) -> None:
        config = V2ExperimentConfig()

        with self.assertRaisesRegex(ValueError, "dataset SHA-256"):
            create_v2_development_loaders(
                sentinel.dataset,
                Path("split_manifest.json"),
                config,
                dataset_sha256="b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
