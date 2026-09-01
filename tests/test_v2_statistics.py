"""Tests for W2 descriptive and paired statistics."""

import math
import unittest

from signal_modulation.v2_statistics import (
    descriptive_summary,
    summarize_w2_results,
    summarize_w3_results,
)


def _record(
    configuration: str,
    run_seed: int,
    *,
    accuracy: float,
    macro_f1: float,
    best_epoch: int,
) -> dict:
    return {
        "status": "completed",
        "configuration": configuration,
        "run_seed": run_seed,
        "model": {
            "trainable_parameters": {
                "A0": 10,
                "A1": 20,
                "A2-T": 30,
                "A2-G": 31,
                "A3": 30,
            }[configuration]
        },
        "training": {"best_epoch": best_epoch},
        "validation": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "by_snr": [
                {
                    "snr": -2.0,
                    "accuracy": accuracy - 0.1,
                    "macro_f1": macro_f1 - 0.1,
                    "sample_count": 4,
                },
                {
                    "snr": 0.0,
                    "accuracy": accuracy + 0.1,
                    "macro_f1": macro_f1 + 0.1,
                    "sample_count": 4,
                },
            ],
        },
        "result_file": f"{configuration}/{run_seed}/result.json",
        "checkpoint_file": f"{configuration}/{run_seed}/best.pt",
    }


class V2StatisticsTests(unittest.TestCase):
    def test_descriptive_summary_uses_sample_standard_deviation(self) -> None:
        summary = descriptive_summary([1.0, 3.0])

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["sample_std"], math.sqrt(2.0))

    def test_w2_summary_reports_all_seeds_and_a1_minus_a0_deltas(self) -> None:
        records = [
            _record("A0", 1, accuracy=0.4, macro_f1=0.3, best_epoch=2),
            _record("A1", 1, accuracy=0.6, macro_f1=0.5, best_epoch=4),
            _record("A0", 2, accuracy=0.5, macro_f1=0.4, best_epoch=3),
            _record("A1", 2, accuracy=0.8, macro_f1=0.7, best_epoch=6),
        ]

        summary = summarize_w2_results(records, run_seeds=(1, 2))

        self.assertEqual(summary["raw_run_count"], 4)
        self.assertEqual(summary["models"]["A0"]["validation_accuracy"]["mean"], 0.45)
        self.assertEqual(summary["paired_delta"]["direction"], "A1_minus_A0")
        self.assertEqual(summary["paired_delta"]["n_pairs"], 2)
        self.assertAlmostEqual(
            summary["paired_delta"]["validation_macro_f1"]["mean"],
            0.25,
        )
        self.assertEqual(summary["paired_delta"]["best_epoch"]["mean"], 2.5)

    def test_missing_seed_pair_is_rejected(self) -> None:
        records = [
            _record("A0", 1, accuracy=0.4, macro_f1=0.3, best_epoch=2),
            _record("A1", 1, accuracy=0.6, macro_f1=0.5, best_epoch=4),
            _record("A0", 2, accuracy=0.5, macro_f1=0.4, best_epoch=3),
        ]

        with self.assertRaisesRegex(ValueError, "A1 does not contain"):
            summarize_w2_results(records, run_seeds=(1, 2))

    def test_failed_run_cannot_be_silently_summarized(self) -> None:
        failed = _record("A0", 1, accuracy=0.4, macro_f1=0.3, best_epoch=2)
        failed["status"] = "failed"

        with self.assertRaisesRegex(ValueError, "every formal run"):
            summarize_w2_results([failed], run_seeds=(1,))

    def test_w3_summary_keeps_two_independent_paired_directions(self) -> None:
        records = []
        for seed, offset in ((1, 0.0), (2, 0.1)):
            records.extend(
                [
                    _record(
                        "A2-T",
                        seed,
                        accuracy=0.70 + offset,
                        macro_f1=0.68 + offset,
                        best_epoch=8,
                    ),
                    _record(
                        "A2-G",
                        seed,
                        accuracy=0.65 + offset,
                        macro_f1=0.63 + offset,
                        best_epoch=9,
                    ),
                    _record(
                        "A3",
                        seed,
                        accuracy=0.60 + offset,
                        macro_f1=0.58 + offset,
                        best_epoch=10,
                    ),
                ]
            )

        summary = summarize_w3_results(records, run_seeds=(1, 2))

        self.assertEqual(summary["new_training_run_count"], 4)
        self.assertEqual(
            summary["aggregation_paired_delta"]["direction"],
            "A2-T_minus_A2-G",
        )
        self.assertAlmostEqual(
            summary["aggregation_paired_delta"]["validation_macro_f1"]["mean"],
            0.05,
        )
        self.assertEqual(
            summary["dropout_paired_delta"]["direction"],
            "A2-T_p0.3_minus_A3_p0.0",
        )
        self.assertAlmostEqual(
            summary["dropout_paired_delta"]["validation_macro_f1"]["mean"],
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
