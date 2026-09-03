"""Regression checks for cross-platform release verification."""

import subprocess
import unittest
from pathlib import Path


class ReleasePortabilityTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    representative_json = "experiments/v2/w2/A0/run_seed_20260901/validation_result.json"
    lf_erratum = "experiments/v2/w3/A3/ERRATA.json"

    def test_json_artifacts_have_a_platform_independent_checkout_eol(self) -> None:
        result = subprocess.run(
            ["git", "check-attr", "text", "eol", "--", self.representative_json],
            cwd=self.repository_root,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn(f"{self.representative_json}: text: set", result.stdout)
        self.assertIn(f"{self.representative_json}: eol: crlf", result.stdout)

        erratum_result = subprocess.run(
            ["git", "check-attr", "text", "eol", "--", self.lf_erratum],
            cwd=self.repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(f"{self.lf_erratum}: text: set", erratum_result.stdout)
        self.assertIn(f"{self.lf_erratum}: eol: lf", erratum_result.stdout)

    def test_ci_fetches_the_commit_used_by_historical_integrity_checks(self) -> None:
        workflow = (
            self.repository_root / ".github" / "workflows" / "ci.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("uses: actions/checkout@v6", workflow)
        self.assertIn("fetch-depth: 0", workflow)


if __name__ == "__main__":
    unittest.main()
