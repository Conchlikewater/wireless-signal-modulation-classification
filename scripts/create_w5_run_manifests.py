"""Append five immutable A2-L run manifests to the existing V2 catalog."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.create_v2_run_manifests import build_manifest
from signal_modulation.data_integrity import sha256_file
from signal_modulation.experiment import write_json_atomic
from signal_modulation.run_manifest import load_json_object
from signal_modulation.v2_experiment import V2_RUN_SEEDS


def clean_repository_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("commit W5 manifest support before catalog generation")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    replay_commit = clean_repository_commit()
    manifest_root = REPOSITORY_ROOT / "experiments/v2/run_manifests"
    catalog_path = manifest_root / "catalog.json"
    catalog = load_json_object(catalog_path)
    if catalog.get("run_count") != 20 or len(catalog.get("records", [])) != 20:
        raise ValueError("W5 append expects the immutable 20-run W2/W3 catalog")
    if any(record.get("configuration") == "A2-L" for record in catalog["records"]):
        raise ValueError("A2-L is already registered")

    new_records = []
    for seed in V2_RUN_SEEDS:
        result_path = (
            REPOSITORY_ROOT
            / f"experiments/v2/w5/A2-L/run_seed_{seed}/validation_result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = build_manifest(
            result,
            result_path=result_path,
            repository_root=REPOSITORY_ROOT,
            replay_code_commit=replay_commit,
        )
        manifest_path = manifest_root / f"{manifest['run_id']}.json"
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        write_json_atomic(manifest_path, manifest)
        new_records.append(
            {
                "run_id": manifest["run_id"],
                "configuration": manifest["configuration"],
                "run_seed": manifest["run_seed"],
                "manifest_file": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )

    summary_errata_path = REPOSITORY_ROOT / "experiments/v2/w5/SUMMARY_ERRATA.json"
    catalog["records"].extend(new_records)
    catalog["run_count"] = len(catalog["records"])
    catalog["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    catalog["replay_code_git_commit"] = replay_commit
    catalog.setdefault("errata", []).append(
        {
            "affected_runs": [record["run_id"] for record in new_records],
            "errata_file": summary_errata_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "errata_sha256": sha256_file(summary_errata_path),
            "field": "w5_summary.hypothesis_evaluation.verdict",
        }
    )
    write_json_atomic(catalog_path, catalog)
    print(f"catalog={catalog_path}")
    print(f"appended_run_count={len(new_records)}")
    print(f"catalog_run_count={catalog['run_count']}")
    print("test_set_used=false")


if __name__ == "__main__":
    main()
