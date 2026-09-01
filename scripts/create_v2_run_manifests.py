"""Create immutable replay manifests from the committed W2/W3 raw results."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from signal_modulation.data_integrity import sha256_file
from signal_modulation.experiment import prepare_new_run_directory, write_json_atomic
from signal_modulation.run_manifest import (
    REPLAYABLE_CONFIGURATIONS,
    RUN_CATALOG_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    validate_run_manifest,
)


DEFAULT_DATA_PATH = "data/raw/RML2016.10a/RML2016.10a_dict.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def clean_repository_commit(repository_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("commit W4 manifest implementation before generation")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _result_paths(repository_root: Path) -> list[Path]:
    patterns = (
        "experiments/v2/w2/A*/run_seed_*/validation_result.json",
        "experiments/v2/w3/A*/run_seed_*/validation_result.json",
    )
    paths = sorted(path for pattern in patterns for path in repository_root.glob(pattern))
    if len(paths) != 20:
        raise ValueError("W4 requires exactly 20 committed W2/W3 raw results")
    return paths


def build_manifest(
    result: dict[str, Any],
    *,
    result_path: Path,
    repository_root: Path,
    replay_code_commit: str,
) -> dict[str, Any]:
    configuration = result["configuration"]
    expected_stage, expected_model = REPLAYABLE_CONFIGURATIONS[configuration]
    if (
        result.get("stage") != expected_stage
        or result.get("model", {}).get("name") != expected_model
        or result.get("status") != "completed"
        or result.get("test_set_used") is not False
    ):
        raise ValueError(f"raw result is not eligible for a run manifest: {result_path}")
    run_seed = result["run_seed"]
    protocol_path = repository_root / "docs" / "v2_protocol.md"
    dependency_paths = [
        repository_root / "pyproject.toml",
        repository_root / "requirements-gpu.txt",
    ]
    checkpoint_path = repository_root / result["checkpoint_file"]
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": f"{expected_stage.lower()}-{configuration.lower()}-{run_seed}",
        "source_stage": expected_stage,
        "configuration": configuration,
        "model_name": expected_model,
        "run_seed": run_seed,
        "scope": "fixed_validation_only",
        "test_set_used": False,
        "dataset": {
            "name": result["dataset"]["name"],
            "default_path": DEFAULT_DATA_PATH,
            "sha256": result["dataset"]["pickle_sha256"],
            "total_samples": result["dataset"]["total_samples"],
        },
        "split": {
            "manifest_file": "manifests/v2/split_manifest.json",
            "manifest_sha256": result["protocol"]["split_manifest_sha256"],
            "split_seed": result["protocol"]["split_seed"],
            "train_samples": result["dataset"]["train_samples"],
            "validation_samples": result["dataset"]["validation_samples"],
            "sealed_test_samples_metadata_only": result["dataset"][
                "sealed_test_samples_metadata_only"
            ],
        },
        "training": {
            "config": result["config"],
            "optimizer": result["optimizer"],
            "checkpoint_selection": result["protocol"]["selection_criterion"],
        },
        "initialization": result.get("initialization"),
        "historical_environment": result["runtime"],
        "historical_artifacts": {
            "source_result_file": _relative(result_path, repository_root),
            "source_result_sha256": sha256_file(result_path),
            "checkpoint_file": result["checkpoint_file"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "implementation_git_commit": result["implementation_git_commit"],
            "best_epoch": result["training"]["best_epoch"],
        },
        "provenance": {
            "protocol_file": _relative(protocol_path, repository_root),
            "protocol_sha256": sha256_file(protocol_path),
            "replay_code_git_commit": replay_code_commit,
            "dependency_specs": [
                {
                    "file": _relative(path, repository_root),
                    "sha256": sha256_file(path),
                }
                for path in dependency_paths
            ],
            "dependency_limit": (
                "Top-level packages are pinned; historical_environment records the "
                "actual Python, NumPy, PyTorch, CUDA and cuDNN versions."
            ),
        },
        "replay": {
            "entrypoint": "scripts/replay_v2_run.py",
            "requires_explicit_execute": True,
            "default_output_directory": f"artifacts/replay/{expected_stage.lower()}-{configuration.lower()}-{run_seed}",
            "does_not_use_v1_test": True,
            "bitwise_reproduction_not_guaranteed": True,
        },
    }
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != result["checkpoint_sha256"]:
        raise ValueError(f"checkpoint is missing or changed: {checkpoint_path}")
    validate_run_manifest(manifest)
    return manifest


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    replay_code_commit = clean_repository_commit(repository_root)
    output = prepare_new_run_directory(args.output_directory)
    catalog_records = []
    for result_path in _result_paths(repository_root):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = build_manifest(
            result,
            result_path=result_path,
            repository_root=repository_root,
            replay_code_commit=replay_code_commit,
        )
        manifest_path = write_json_atomic(output / f"{manifest['run_id']}.json", manifest)
        catalog_records.append(
            {
                "run_id": manifest["run_id"],
                "configuration": manifest["configuration"],
                "run_seed": manifest["run_seed"],
                "manifest_file": _relative(manifest_path, repository_root),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    catalog_path = write_json_atomic(
        output / "catalog.json",
        {
            "schema_version": RUN_CATALOG_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "replay_code_git_commit": replay_code_commit,
            "run_count": len(catalog_records),
            "scope": "fixed_validation_only",
            "test_set_used": False,
            "records": catalog_records,
        },
    )
    print(f"catalog={catalog_path}")
    print(f"run_count={len(catalog_records)}")
    print("test_set_used=false")


if __name__ == "__main__":
    main()
