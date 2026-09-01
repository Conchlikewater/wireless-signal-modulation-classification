"""Repository-level checks for the committed W4 run-manifest catalog."""

import json
import unittest
from pathlib import Path

from signal_modulation.data_integrity import sha256_file
from signal_modulation.run_manifest import (
    RUN_CATALOG_SCHEMA,
    build_replay_commands,
    load_json_object,
    verify_run_manifest,
)
from signal_modulation.v2_experiment import V2_RUN_SEEDS


class RunManifestArtifactTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_root = repository_root / "experiments" / "v2" / "run_manifests"
    catalog_path = manifest_root / "catalog.json"
    expected_catalog_sha256 = (
        "3cc19ec242101405e0a9304034d2370f00dcf5afb5619cd9d259a5eaaa2645dd"
    )

    def test_catalog_contains_every_frozen_configuration_seed_pair(self) -> None:
        self.assertEqual(sha256_file(self.catalog_path), self.expected_catalog_sha256)
        catalog = load_json_object(self.catalog_path)
        self.assertEqual(catalog["schema_version"], RUN_CATALOG_SCHEMA)
        self.assertEqual(catalog["run_count"], 20)
        self.assertEqual(catalog["scope"], "fixed_validation_only")
        self.assertFalse(catalog["test_set_used"])

        observed = {
            (record["configuration"], record["run_seed"])
            for record in catalog["records"]
        }
        expected = {
            (configuration, run_seed)
            for configuration in ("A0", "A1", "A2-G", "A3")
            for run_seed in V2_RUN_SEEDS
        }
        self.assertEqual(observed, expected)

    def test_all_manifest_hashes_and_repository_dependencies_verify(self) -> None:
        catalog = load_json_object(self.catalog_path)
        for record in catalog["records"]:
            manifest_path = self.repository_root / record["manifest_file"]
            self.assertEqual(sha256_file(manifest_path), record["manifest_sha256"])
            manifest = load_json_object(manifest_path)
            verification = verify_run_manifest(
                manifest,
                repository_root=self.repository_root,
            )
            self.assertEqual(verification["run_id"], record["run_id"])
            self.assertEqual(verification["data_status"], "not_checked")
            self.assertFalse(verification["test_set_used"])
            expected_file_count = 8 if manifest["configuration"] == "A2-G" else 7
            self.assertEqual(
                verification["verified_repository_file_count"],
                expected_file_count,
            )

    def test_any_catalog_record_can_reconstruct_an_explicit_replay_command(self) -> None:
        catalog = load_json_object(self.catalog_path)
        for record in catalog["records"]:
            manifest = load_json_object(self.repository_root / record["manifest_file"])
            commands = build_replay_commands(
                record["manifest_file"],
                data_file=manifest["dataset"]["default_path"],
                output_directory=manifest["replay"]["default_output_directory"],
            )
            self.assertNotIn("--execute", commands["dry_run"])
            self.assertEqual(commands["execute"][-1], "--execute")
            self.assertIn(str(record["run_seed"]), record["run_id"])


if __name__ == "__main__":
    unittest.main()
