"""Integrity checks for the five committed A2-L validation runs."""

import json
import unittest
from pathlib import Path

import torch

from scripts.run_v2_lstm_ablation import evaluate_preregistered_hypothesis
from signal_modulation.data_integrity import sha256_file
from signal_modulation.v2_experiment import V2_RUN_SEEDS
from signal_modulation.v2_statistics import descriptive_summary
from signal_modulation.w3_ablation import create_w3_model, state_dict_sha256


class W5ArtifactTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    w5_root = repository_root / "experiments/v2/w5"
    expected_original_summary_sha256 = (
        "9beca76b8e5d453da0ee232b39114a641f7aa2ce2343c895d0f14636b4d5c61a"
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.original = json.loads(
            (cls.w5_root / "w5_summary.json").read_text(encoding="utf-8")
        )
        cls.corrected = json.loads(
            (cls.w5_root / "w5_summary_corrected.json").read_text(encoding="utf-8")
        )
        cls.errata = json.loads(
            (cls.w5_root / "SUMMARY_ERRATA.json").read_text(encoding="utf-8")
        )
        cls.initialization = json.loads(
            (
                cls.repository_root
                / "experiments/v2/w3/initial_backbones/initialization_manifest.json"
            ).read_text(encoding="utf-8")
        )
        cls.initialization_by_seed = {
            int(record["run_seed"]): record
            for record in cls.initialization["records"]
        }
        cls.results = {
            seed: json.loads(
                (
                    cls.w5_root
                    / f"A2-L/run_seed_{seed}/validation_result.json"
                ).read_text(encoding="utf-8")
            )
            for seed in V2_RUN_SEEDS
        }

    def test_every_seed_has_a_hashed_loadable_checkpoint_and_shared_start(self) -> None:
        self.assertEqual(set(self.results), set(V2_RUN_SEEDS))
        self.assertEqual(list(self.w5_root.rglob("failure.json")), [])
        for seed, result in self.results.items():
            self.assertFalse(result["test_set_used"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["configuration"], "A2-L")
            self.assertEqual(result["effective_model_config"]["dropout"], 0.3)
            self.assertEqual(result["model"]["trainable_parameters"], 223_932)

            checkpoint_path = self.repository_root / result["checkpoint_file"]
            self.assertEqual(sha256_file(checkpoint_path), result["checkpoint_sha256"])
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            initial_record = self.initialization_by_seed[seed]
            initial_path = self.repository_root / initial_record[
                "a2_shared_backbone_file"
            ]
            self.assertEqual(
                sha256_file(initial_path),
                initial_record["a2_shared_backbone_file_sha256"],
            )
            initial_state = torch.load(initial_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                state_dict_sha256(initial_state),
                initial_record["a2_shared_backbone_state_sha256"],
            )
            self.assertEqual(
                result["initialization"]["shared_backbone_state_sha256"],
                initial_record["a2_shared_backbone_state_sha256"],
            )
            model = create_w3_model(
                "A2-L",
                num_classes=11,
                initial_backbone=initial_state,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def test_corrected_summary_recomputes_raw_means_without_overwriting_original(self) -> None:
        self.assertEqual(
            sha256_file(self.w5_root / "w5_summary.json"),
            self.expected_original_summary_sha256,
        )
        accuracy = descriptive_summary(
            [self.results[seed]["validation"]["accuracy"] for seed in V2_RUN_SEEDS]
        )
        macro_f1 = descriptive_summary(
            [self.results[seed]["validation"]["macro_f1"] for seed in V2_RUN_SEEDS]
        )
        self.assertEqual(self.corrected["validation_accuracy"], accuracy)
        self.assertEqual(self.corrected["validation_macro_f1"], macro_f1)
        self.assertEqual(self.original["validation_macro_f1"], macro_f1)
        self.assertEqual(self.original["hypothesis_evaluation"]["verdict"], "partially_supported")
        self.assertEqual(
            self.corrected["hypothesis_evaluation"],
            evaluate_preregistered_hypothesis(
                self.corrected["snr_segment_paired_delta"]
            ),
        )
        self.assertEqual(self.corrected["hypothesis_evaluation"]["verdict"], "not_supported")
        self.assertEqual(self.errata["affected_sha256"], self.expected_original_summary_sha256)
        self.assertEqual(self.errata["effective_value"], "not_supported")

    def test_capacity_and_scope_match_preregistration(self) -> None:
        capacity = self.corrected["capacity_control"]
        self.assertEqual(capacity["a2l_trainable_parameters"], 223_932)
        self.assertEqual(capacity["a2t_trainable_parameters"], 224_587)
        self.assertLessEqual(capacity["absolute_percent_difference"], 1.0)
        self.assertFalse(capacity["strict_single_variable_ablation"])
        self.assertEqual(self.corrected["raw_run_count"], 5)
        self.assertEqual(self.corrected["run_seeds"], list(V2_RUN_SEEDS))
        self.assertFalse(self.corrected["test_set_used"])


if __name__ == "__main__":
    unittest.main()
