"""Guard recorded V2 configuration semantics against the instantiated models."""

import unittest
from dataclasses import asdict
from pathlib import Path

from torch import nn

from scripts.run_v2_paired_experiments import create_w2_model
from signal_modulation.config_semantics import (
    effective_model_config,
    validate_shared_training_config,
)
from signal_modulation.model import TemporalCNN1D
from signal_modulation.run_manifest import load_json_object
from signal_modulation.v2_experiment import V2ExperimentConfig
from signal_modulation.w3_ablation import (
    create_w3_model,
    temporal_backbone_state,
)


class ConfigSemanticsTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    catalog = load_json_object(
        repository_root / "experiments" / "v2" / "run_manifests" / "catalog.json"
    )
    erratum = load_json_object(
        repository_root / "experiments" / "v2" / "w3" / "A3" / "ERRATA.json"
    )

    @staticmethod
    def _instantiate_model(configuration: str) -> nn.Module:
        if configuration in {"A0", "A1"}:
            return create_w2_model(configuration, num_classes=11)
        if configuration == "A2-G":
            reference = TemporalCNN1D(num_classes=11, dropout=0.3)
            return create_w3_model(
                configuration,
                num_classes=11,
                initial_backbone=temporal_backbone_state(reference),
            )
        return create_w3_model(configuration, num_classes=11)

    def test_recorded_values_match_effective_model_and_shared_training_config(self) -> None:
        for record in self.catalog["records"]:
            with self.subTest(run_id=record["run_id"]):
                manifest = load_json_object(
                    self.repository_root / record["manifest_file"]
                )
                historical_result = load_json_object(
                    self.repository_root
                    / manifest["historical_artifacts"]["source_result_file"]
                )
                recorded = manifest["training"]["config"]
                self.assertEqual(recorded, historical_result["config"])
                validate_shared_training_config(
                    recorded,
                    run_seed=record["run_seed"],
                )
                expected = asdict(V2ExperimentConfig(run_seed=record["run_seed"]))
                for field in (
                    "epochs",
                    "learning_rate",
                    "train_batch_size",
                    "evaluation_batch_size",
                ):
                    self.assertEqual(recorded[field], expected[field])

                effective = effective_model_config(
                    record["configuration"],
                    run_id=record["run_id"],
                    recorded_config=recorded,
                    erratum=(self.erratum if record["configuration"] == "A3" else None),
                )
                model = self._instantiate_model(record["configuration"])
                observed = [
                    module.p for module in model.modules() if isinstance(module, nn.Dropout)
                ]
                if effective["dropout"] is None:
                    self.assertEqual(observed, [])
                else:
                    self.assertEqual(observed, [effective["dropout"]])

    def test_erratum_covers_exactly_the_five_historical_a3_runs(self) -> None:
        catalog_a3_runs = {
            record["run_id"]
            for record in self.catalog["records"]
            if record["configuration"] == "A3"
        }

        self.assertEqual(set(self.erratum["affected_runs"]), catalog_a3_runs)
        self.assertEqual(self.erratum["recorded_value"], 0.3)
        self.assertEqual(self.erratum["effective_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
