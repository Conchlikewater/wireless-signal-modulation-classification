"""Integrity checks for the committed W3 ablation results and states."""

import json
import unittest
from pathlib import Path

import torch

from scripts.run_v2_ablation_experiments import load_reused_a2t_records
from signal_modulation.data_integrity import sha256_file
from signal_modulation.model import GlobalPoolingTemporalCNN1D, TemporalCNN1D
from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_RUN_SEEDS,
    V2_SPLIT_MANIFEST_SHA256,
)
from signal_modulation.v2_statistics import summarize_w3_results
from signal_modulation.w3_ablation import state_dict_sha256


class W3ArtifactTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    artifact_root = repository_root / "experiments" / "v2" / "w3"
    w2_artifact_root = repository_root / "experiments" / "v2" / "w2"
    expected_summary_sha256 = (
        "b0edff36b715f68599ceb8076e371a80994ac0af4ae02e4d7af67d357eaeccc8"
    )

    def _new_results(self) -> list[dict]:
        result_paths = sorted(
            self.artifact_root.glob("A*/run_seed_*/validation_result.json")
        )
        self.assertEqual(len(result_paths), 10)
        return [
            json.loads(path.read_text(encoding="utf-8")) for path in result_paths
        ]

    def test_every_new_pair_has_a_loadable_hashed_checkpoint(self) -> None:
        observed_pairs = set()
        for result in self._new_results():
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
            self.assertEqual(sha256_file(checkpoint_path), result["checkpoint_sha256"])
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                int(checkpoint["epoch"]),
                int(result["training"]["best_epoch"]),
            )
            if result["configuration"] == "A2-G":
                model = GlobalPoolingTemporalCNN1D(num_classes=11)
            else:
                model = TemporalCNN1D(num_classes=11, dropout=0.0)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            observed_pairs.add((result["configuration"], result["run_seed"]))

        expected_pairs = {
            (configuration, run_seed)
            for configuration in ("A2-G", "A3")
            for run_seed in V2_RUN_SEEDS
        }
        self.assertEqual(observed_pairs, expected_pairs)

    def test_summary_recomputes_from_reused_and_new_runs(self) -> None:
        summary_path = self.artifact_root / "w3_summary.json"
        self.assertEqual(sha256_file(summary_path), self.expected_summary_sha256)
        committed = json.loads(summary_path.read_text(encoding="utf-8"))
        all_results = load_reused_a2t_records(self.w2_artifact_root) + self._new_results()
        recomputed = summarize_w3_results(all_results, run_seeds=V2_RUN_SEEDS)

        for key, value in recomputed.items():
            self.assertEqual(committed[key], value)
        self.assertEqual(committed["record_count"], 15)
        self.assertEqual(committed["new_training_run_count"], 10)
        self.assertEqual(committed["aggregation_paired_delta"]["n_pairs"], 5)
        self.assertEqual(committed["dropout_paired_delta"]["n_pairs"], 5)
        self.assertFalse(committed["test_set_used"])

    def test_initial_backbones_and_capacity_lock_are_auditable(self) -> None:
        initialization_path = (
            self.artifact_root / "initial_backbones" / "initialization_manifest.json"
        )
        initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
        self.assertEqual(len(initialization["records"]), 5)
        for record in initialization["records"]:
            backbone_path = self.repository_root / record["a2_shared_backbone_file"]
            self.assertEqual(
                sha256_file(backbone_path),
                record["a2_shared_backbone_file_sha256"],
            )
            backbone = torch.load(backbone_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                state_dict_sha256(backbone),
                record["a2_shared_backbone_state_sha256"],
            )
            self.assertTrue(record["a3_initial_state_matches_a1"])
            self.assertEqual(
                record["a3_reference_a1_initial_state_sha256"],
                record["a3_initial_state_sha256"],
            )

        matrix_lock = json.loads(
            (self.artifact_root / "matrix_lock.json").read_text(encoding="utf-8")
        )
        control = matrix_lock["capacity_control"]
        self.assertTrue(control["target_satisfied"])
        self.assertFalse(control["strict_single_variable_ablation"])
        self.assertLessEqual(control["full_parameter_percent_difference"], 1.0)
        self.assertEqual(control["full_parameter_absolute_difference"], 28)

    def test_no_failed_run_record_exists(self) -> None:
        self.assertEqual(list(self.artifact_root.rglob("failure.json")), [])


if __name__ == "__main__":
    unittest.main()
