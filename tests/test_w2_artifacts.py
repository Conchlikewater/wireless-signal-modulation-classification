"""Integrity checks for the committed W2 raw results and checkpoints."""

import json
import unittest
from pathlib import Path

import torch

from signal_modulation.data_integrity import sha256_file
from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_RUN_SEEDS,
    V2_SPLIT_MANIFEST_SHA256,
)
from signal_modulation.v2_statistics import summarize_w2_results


class W2ArtifactTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    artifact_root = repository_root / "experiments" / "v2" / "w2"

    def _raw_results(self) -> list[dict]:
        result_paths = sorted(
            self.artifact_root.glob("A*/run_seed_*/validation_result.json")
        )
        self.assertEqual(len(result_paths), 10)
        return [
            json.loads(path.read_text(encoding="utf-8")) for path in result_paths
        ]

    def test_every_registered_pair_has_a_loadable_hashed_checkpoint(self) -> None:
        results = self._raw_results()
        observed_pairs = set()
        for result in results:
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["scope"], "fixed_validation_only")
            self.assertFalse(result["test_set_used"])
            self.assertNotIn("test", result)
            self.assertEqual(result["dataset"]["pickle_sha256"], V2_DATASET_SHA256)
            self.assertEqual(
                result["protocol"]["split_manifest_sha256"],
                V2_SPLIT_MANIFEST_SHA256,
            )
            self.assertEqual(result["validation"]["sample_count"], 33_000)

            checkpoint_path = self.repository_root / result["checkpoint_file"]
            self.assertTrue(checkpoint_path.is_file())
            self.assertEqual(
                sha256_file(checkpoint_path),
                result["checkpoint_sha256"],
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                int(checkpoint["epoch"]),
                int(result["training"]["best_epoch"]),
            )
            observed_pairs.add((result["configuration"], result["run_seed"]))

        expected_pairs = {
            (configuration, run_seed)
            for configuration in ("A0", "A1")
            for run_seed in V2_RUN_SEEDS
        }
        self.assertEqual(observed_pairs, expected_pairs)

    def test_summary_is_recomputed_from_all_raw_runs(self) -> None:
        results = self._raw_results()
        committed = json.loads(
            (self.artifact_root / "w2_summary.json").read_text(encoding="utf-8")
        )
        recomputed = summarize_w2_results(results, run_seeds=V2_RUN_SEEDS)

        for key, value in recomputed.items():
            self.assertEqual(committed[key], value)
        self.assertEqual(committed["raw_run_count"], 10)
        self.assertEqual(committed["paired_delta"]["n_pairs"], 5)
        self.assertFalse(committed["test_set_used"])

    def test_no_failed_run_record_exists(self) -> None:
        self.assertEqual(list(self.artifact_root.rglob("failure.json")), [])


if __name__ == "__main__":
    unittest.main()
