"""Repository-level checks for validation-only W-B analysis artifacts."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from signal_modulation.data_integrity import sha256_file
from signal_modulation.run_manifest import load_json_object
from signal_modulation.v2_experiment import V2_RUN_SEEDS


class AnalysisArtifactTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    analysis_root = repository_root / "experiments" / "v2" / "analysis"
    summary = load_json_object(analysis_root / "analysis_summary.json")

    def test_analysis_is_validation_only_and_covers_every_arm_seed(self) -> None:
        self.assertEqual(self.summary["scope"], "fixed_validation_only")
        self.assertFalse(self.summary["test_set_used"])
        self.assertEqual(self.summary["run_seeds"], list(V2_RUN_SEEDS))
        self.assertEqual(
            set(self.summary["arms"]),
            {"A0", "A1/A2-T", "A2-G", "A3"},
        )
        for arm in self.summary["arms"].values():
            self.assertEqual(arm["n_runs"], 5)
            self.assertEqual(
                {record["run_seed"] for record in arm["runs"]},
                set(V2_RUN_SEEDS),
            )

    def test_derived_records_match_committed_sources_and_expected_shapes(self) -> None:
        for arm_name, arm in self.summary["arms"].items():
            with self.subTest(arm=arm_name):
                self.assertEqual(len(arm["by_snr"]), 20)
                self.assertEqual(len(arm["class_snr_accuracy_mean"]), 11)
                self.assertTrue(
                    all(len(row) == 20 for row in arm["class_snr_accuracy_mean"])
                )
                self.assertEqual(
                    sum(sum(row) for row in arm["low_snr_confusion_matrix_sum"]),
                    49_500,
                )
                self.assertEqual(
                    sum(sum(row) for row in arm["high_snr_confusion_matrix_sum"]),
                    41_250,
                )
                for record in arm["runs"]:
                    source_path = self.repository_root / record["source_result_file"]
                    checkpoint_path = self.repository_root / record["checkpoint_file"]
                    self.assertEqual(
                        sha256_file(source_path), record["source_result_sha256"]
                    )
                    self.assertEqual(
                        sha256_file(checkpoint_path), record["checkpoint_sha256"]
                    )
                    source = load_json_object(source_path)
                    self.assertEqual(record["accuracy"], source["validation"]["accuracy"])
                    self.assertEqual(record["macro_f1"], source["validation"]["macro_f1"])
                    self.assertTrue(
                        all(
                            count == 150
                            for row in record["class_snr_sample_counts"]
                            for count in row
                        )
                    )

    def test_all_expected_svg_files_are_well_formed(self) -> None:
        for filename in (
            "accuracy_vs_snr.svg",
            "confusion_high_snr.svg",
            "confusion_low_snr.svg",
            "class_snr_accuracy.svg",
        ):
            with self.subTest(filename=filename):
                root = ET.parse(self.analysis_root / filename).getroot()
                self.assertTrue(root.tag.endswith("svg"))


if __name__ == "__main__":
    unittest.main()
