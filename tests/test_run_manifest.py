"""Tests for W4 run-manifest safety and replay planning."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.create_v2_run_manifests import build_manifest
from signal_modulation.run_manifest import (
    build_replay_commands,
    validate_run_manifest,
    verify_run_manifest,
)


class RunManifestTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    result_path = (
        repository_root
        / "experiments"
        / "v2"
        / "w2"
        / "A0"
        / "run_seed_20260901"
        / "validation_result.json"
    )

    def _manifest(self) -> dict:
        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        return build_manifest(
            result,
            result_path=self.result_path,
            repository_root=self.repository_root,
            replay_code_commit="a" * 40,
        )

    def _committed_manifest(self, name: str) -> dict:
        path = self.repository_root / "experiments" / "v2" / "run_manifests" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def test_historical_result_builds_a_validation_only_manifest(self) -> None:
        manifest = self._manifest()
        validate_run_manifest(manifest)

        self.assertEqual(manifest["run_id"], "w2-a0-20260901")
        self.assertFalse(manifest["test_set_used"])
        self.assertTrue(manifest["replay"]["requires_explicit_execute"])
        self.assertEqual(manifest["training"]["config"]["run_seed"], 20260901)

    def test_changed_seed_or_test_scope_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["run_seed"] = 999
        with self.assertRaises(ValueError):
            validate_run_manifest(manifest)

        manifest = self._manifest()
        manifest["test_set_used"] = True
        with self.assertRaises(ValueError):
            validate_run_manifest(manifest)

    def test_repository_artifacts_verify_without_loading_signals(self) -> None:
        verification = verify_run_manifest(
            self._manifest(),
            repository_root=self.repository_root,
        )

        self.assertEqual(verification["data_status"], "not_checked")
        self.assertFalse(verification["ready_for_explicit_execute"])
        self.assertFalse(verification["test_set_used"])
        self.assertEqual(verification["verified_repository_file_count"], 7)

    def test_initialization_schema_is_configuration_specific(self) -> None:
        manifest = self._manifest()
        manifest["initialization"] = {"unexpected": True}
        with self.assertRaises(ValueError):
            validate_run_manifest(manifest)

        manifest = self._committed_manifest("w3-a2-g-20260901.json")
        del manifest["initialization"]["shared_backbone_state_sha256"]
        with self.assertRaises(ValueError):
            validate_run_manifest(manifest)

        manifest = self._committed_manifest("w3-a3-20260901.json")
        manifest["initialization"]["matches_a1_initial_state"] = False
        with self.assertRaises(ValueError):
            validate_run_manifest(manifest)

    def test_a2g_shared_initial_backbone_is_part_of_preflight(self) -> None:
        manifest = self._committed_manifest("w3-a2-g-20260901.json")
        verification = verify_run_manifest(
            manifest,
            repository_root=self.repository_root,
        )
        self.assertEqual(verification["verified_repository_file_count"], 8)

        manifest["initialization"]["shared_backbone_file_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_run_manifest(manifest, repository_root=self.repository_root)

    def test_a2l_manifest_requires_effective_config_and_trainable_backbone(self) -> None:
        result_path = (
            self.repository_root
            / "experiments/v2/w5/A2-L/run_seed_20260901/validation_result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = build_manifest(
            result,
            result_path=result_path,
            repository_root=self.repository_root,
            replay_code_commit="a" * 40,
        )

        validate_run_manifest(manifest)
        self.assertEqual(manifest["source_stage"], "W5")
        self.assertEqual(manifest["model_name"], "LSTMTemporalCNN1D")
        self.assertEqual(manifest["training"]["effective_model_config"]["dropout"], 0.3)

        manifest["training"]["effective_model_config"]["dropout"] = 0.0
        with self.assertRaises(ValueError):
            validate_run_manifest(manifest)

    def test_tampered_dependency_spec_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["provenance"]["dependency_specs"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_run_manifest(manifest, repository_root=self.repository_root)

    def test_replay_command_is_dry_run_unless_execute_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            commands = build_replay_commands(
                "experiments/v2/run_manifests/w2-a0-20260901.json",
                data_file="data/raw/RML2016.10a/RML2016.10a_dict.pkl",
                output_directory=Path(directory) / "replay",
            )

        self.assertNotIn("--execute", commands["dry_run"])
        self.assertEqual(commands["execute"][-1], "--execute")


if __name__ == "__main__":
    unittest.main()
