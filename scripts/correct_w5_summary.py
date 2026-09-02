"""Preserve the original W5 summary and write its deterministic verdict correction."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_v2_lstm_ablation import evaluate_preregistered_hypothesis
from signal_modulation.data_integrity import sha256_file
from signal_modulation.experiment import write_json_atomic


EXPECTED_ORIGINAL_SHA256 = (
    "9beca76b8e5d453da0ee232b39114a641f7aa2ce2343c895d0f14636b4d5c61a"
)


def current_commit(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    w5_root = repository_root / "experiments/v2/w5"
    original_path = w5_root / "w5_summary.json"
    corrected_path = w5_root / "w5_summary_corrected.json"
    errata_path = w5_root / "SUMMARY_ERRATA.json"
    if corrected_path.exists() or errata_path.exists():
        raise FileExistsError("W5 correction artifacts already exist")
    original_sha256 = sha256_file(original_path)
    if original_sha256 != EXPECTED_ORIGINAL_SHA256:
        raise ValueError("original W5 summary changed before correction")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original["hypothesis_evaluation"]["verdict"] != "partially_supported":
        raise ValueError("original W5 summary no longer contains the recorded defect")

    corrected_evaluation = evaluate_preregistered_hypothesis(
        original["snr_segment_paired_delta"]
    )
    if corrected_evaluation["verdict"] != "not_supported":
        raise RuntimeError("corrected evaluator did not reproduce the preregistered rule")
    correction_commit = current_commit(repository_root)
    corrected_at = datetime.now(timezone.utc).isoformat()
    corrected = {
        **original,
        "schema_version": "wireless-v2-w5-summary-v2",
        "corrected_at_utc": corrected_at,
        "correction_git_commit": correction_commit,
        "supersedes_file": "experiments/v2/w5/w5_summary.json",
        "supersedes_sha256": original_sha256,
        "hypothesis_evaluation": corrected_evaluation,
    }
    write_json_atomic(corrected_path, corrected)
    write_json_atomic(
        errata_path,
        {
            "schema_version": "wireless-v2-w5-summary-errata-v1",
            "affected_file": "experiments/v2/w5/w5_summary.json",
            "affected_sha256": original_sha256,
            "field": "hypothesis_evaluation.verdict",
            "recorded_value": "partially_supported",
            "effective_value": "not_supported",
            "discovered_at_utc": corrected_at,
            "discovered_by": "Codex during W-C artifact audit",
            "evidence_file": "experiments/v2/w5/HYPOTHESIS.md",
            "correction_git_commit": correction_commit,
            "corrected_summary_file": (
                "experiments/v2/w5/w5_summary_corrected.json"
            ),
            "impact_assessment": (
                "Only the categorical hypothesis verdict was wrong. The five raw "
                "validation results, checkpoints, metrics, paired deltas, parameter "
                "counts, and training decisions are unchanged; no retraining is needed."
            ),
        },
    )
    print(f"original_summary_sha256={original_sha256}")
    print("corrected_verdict=not_supported")
    print("training_rerun=false")
    print("test_set_used=false")


if __name__ == "__main__":
    main()
