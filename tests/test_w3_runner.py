"""Tests for W3 matrix scope and W2 A2-T result reuse."""

import unittest
from pathlib import Path

from scripts.run_v2_ablation_experiments import (
    W3_MATRIX,
    load_reused_a2t_records,
)
from signal_modulation.v2_experiment import V2_RUN_SEEDS


class W3RunnerTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]

    def test_matrix_trains_only_a2g_and_a3(self) -> None:
        self.assertEqual(
            W3_MATRIX,
            (("A2-G", "GlobalPoolingTemporalCNN1D"), ("A3", "TemporalCNN1D")),
        )

    def test_a2t_reuses_all_w2_a1_results_without_retraining(self) -> None:
        records = load_reused_a2t_records(
            self.repository_root / "experiments" / "v2" / "w2"
        )

        self.assertEqual(len(records), 5)
        self.assertEqual({record["run_seed"] for record in records}, set(V2_RUN_SEEDS))
        self.assertTrue(all(record["configuration"] == "A2-T" for record in records))
        self.assertTrue(all(record["source_configuration"] == "A1" for record in records))
        self.assertTrue(all(record["reused_from_w2"] for record in records))


if __name__ == "__main__":
    unittest.main()
