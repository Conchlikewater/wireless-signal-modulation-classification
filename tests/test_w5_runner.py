"""Tests for the preregistered A2-L runner and decision rule."""

import unittest

from scripts.run_v2_lstm_ablation import (
    W5_CONFIGURATION,
    _per_snr_summary,
    evaluate_preregistered_hypothesis,
)


def _segments(high_mean: float, positive: int, low_mean: float) -> dict:
    return {
        "high_snr_ge_10": {
            "macro_f1": {"mean": high_mean},
            "positive_macro_f1_seed_count": positive,
        },
        "low_snr_le_-10": {
            "macro_f1": {"mean": low_mean},
            "positive_macro_f1_seed_count": 2,
        },
    }


class W5RunnerTests(unittest.TestCase):
    def test_runner_contains_only_the_preregistered_a2l_arm(self) -> None:
        self.assertEqual(W5_CONFIGURATION, "A2-L")

    def test_hypothesis_requires_both_comparators_to_pass(self) -> None:
        result = evaluate_preregistered_hypothesis(
            {
                "A2-T": _segments(0.012, 4, -0.004),
                "A2-G": _segments(0.015, 5, 0.002),
            }
        )

        self.assertEqual(result["verdict"], "supported")

    def test_small_or_inconsistent_high_snr_delta_is_not_supported(self) -> None:
        result = evaluate_preregistered_hypothesis(
            {
                "A2-T": _segments(0.004, 3, 0.0),
                "A2-G": _segments(-0.010, 1, 0.0),
            }
        )

        self.assertEqual(result["verdict"], "not_supported")

    def test_failed_low_snr_rule_is_not_mislabeled_as_partial_support(self) -> None:
        result = evaluate_preregistered_hypothesis(
            {
                "A2-T": _segments(0.031, 4, -0.006),
                "A2-G": _segments(0.028, 4, -0.015),
            }
        )

        self.assertEqual(result["verdict"], "not_supported")
        self.assertFalse(result["comparators"]["A2-G"]["low_rule_passed"])

    def test_per_snr_summary_uses_all_frozen_seeds(self) -> None:
        records = {
            seed: {
                "validation": {
                    "by_snr": [
                        {
                            "snr": -10.0,
                            "sample_count": 1_650,
                            "accuracy": index / 10.0,
                            "macro_f1": index / 20.0,
                        }
                    ]
                }
            }
            for index, seed in enumerate(
                (20260901, 20260902, 20260903, 20260904, 20260905),
                start=1,
            )
        }

        summary = _per_snr_summary(records)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["snr"], -10.0)
        self.assertAlmostEqual(summary[0]["accuracy"]["mean"], 0.3)
        self.assertEqual(summary[0]["accuracy"]["n"], 5)
        self.assertEqual(summary[0]["sample_count_per_seed"], 1_650)


if __name__ == "__main__":
    unittest.main()
