"""Prove that protected V2 artifacts still match the pre-erratum baseline bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BASELINE_COMMIT = "c05d7ef31229abd564e6feceed71dd09e6b6d2af"
EXPECTED_COUNTS = {
    "validation_result": 20,
    "checkpoint": 20,
    "initial_backbone": 5,
    "summary": 2,
    "run_manifest": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def _git_bytes(repository_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout


def _baseline_worktree_bytes(repository_root: Path, relative_path: str) -> bytes:
    """Read baseline bytes after applying the path's checkout/EOL filters."""

    return _git_bytes(
        repository_root,
        "cat-file",
        "--filters",
        f"--path={relative_path}",
        f"{BASELINE_COMMIT}:{relative_path}",
    )


def protected_paths(repository_root: Path) -> dict[str, list[str]]:
    output = _git_bytes(
        repository_root,
        "ls-tree",
        "-r",
        "--name-only",
        BASELINE_COMMIT,
        "--",
        "experiments/v2",
    ).decode("utf-8")
    paths = [line for line in output.splitlines() if line]
    groups = {
        "validation_result": [
            path for path in paths if path.endswith("/validation_result.json")
        ],
        "checkpoint": [path for path in paths if path.endswith("/best_checkpoint.pt")],
        "initial_backbone": [
            path
            for path in paths
            if path.startswith("experiments/v2/w3/initial_backbones/run_seed_")
            and path.endswith(".pt")
        ],
        "summary": [
            path
            for path in paths
            if path in {
                "experiments/v2/w2/w2_summary.json",
                "experiments/v2/w3/w3_summary.json",
            }
        ],
        "run_manifest": [
            path
            for path in paths
            if path.startswith("experiments/v2/run_manifests/")
            and path.endswith(".json")
            and not path.endswith("/catalog.json")
        ],
    }
    actual_counts = {name: len(items) for name, items in groups.items()}
    if actual_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"protected artifact inventory changed: {actual_counts} != {EXPECTED_COUNTS}"
        )
    return groups


def verify_protected_artifacts(repository_root: Path) -> dict[str, object]:
    repository_root = repository_root.resolve()
    groups = protected_paths(repository_root)
    records = []
    for group, paths in groups.items():
        for relative_path in paths:
            baseline_bytes = _baseline_worktree_bytes(repository_root, relative_path)
            current_path = repository_root / relative_path
            if not current_path.is_file():
                raise FileNotFoundError(current_path)
            current_bytes = current_path.read_bytes()
            baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
            current_sha256 = hashlib.sha256(current_bytes).hexdigest()
            records.append(
                {
                    "group": group,
                    "path": relative_path,
                    "sha256": current_sha256,
                    "matches_baseline": current_sha256 == baseline_sha256,
                }
            )

    mismatches = [record for record in records if not record["matches_baseline"]]
    if mismatches:
        raise RuntimeError(f"protected V2 artifacts changed: {mismatches}")
    return {
        "baseline_commit": BASELINE_COMMIT,
        "counts": {name: len(items) for name, items in groups.items()},
        "protected_file_count": len(records),
        "all_sha256_match": True,
        "records": records,
    }


def main() -> None:
    result = verify_protected_artifacts(parse_args().repository_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
