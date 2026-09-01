"""Verify and optionally relaunch exactly one V2 run from its manifest."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_v2_ablation_experiments import run_one_experiment as run_w3_experiment
from scripts.run_v2_paired_experiments import run_one_experiment as run_w2_experiment
from signal_modulation.data_integrity import sha256_file
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import RadioMLTorchDataset
from signal_modulation.run_manifest import (
    build_replay_commands,
    load_json_object,
    verify_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_file", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _current_commit(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_environment(manifest: dict) -> None:
    expected = manifest["historical_environment"]
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for key, value in actual.items():
        if value != expected.get(key):
            raise RuntimeError(
                f"replay environment mismatch for {key}: expected {expected.get(key)!r}, got {value!r}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("historical V2 run requires CUDA")


def execute_replay(
    manifest: dict,
    *,
    repository_root: Path,
    data_file: Path,
    output_directory: Path,
) -> dict:
    """Relaunch one registered run; never create or evaluate a test loader."""

    _verify_environment(manifest)
    if output_directory.exists():
        raise FileExistsError("replay output directory must not already exist")
    dataset_sha256 = sha256_file(data_file)
    grouped = load_restricted_radioml_pickle(data_file)
    dataset = RadioMLTorchDataset(grouped)
    common = {
        "configuration": manifest["configuration"],
        "run_seed": manifest["run_seed"],
        "dataset": dataset,
        "dataset_sha256": dataset_sha256,
        "split_manifest": repository_root / manifest["split"]["manifest_file"],
        "output_root": output_directory,
        "repository_root": repository_root,
        "implementation_commit": _current_commit(repository_root),
        "device": torch.device("cuda"),
    }
    if manifest["source_stage"] == "W2":
        return run_w2_experiment(**common)

    initialization = manifest.get("initialization") or {}
    if manifest["configuration"] == "A2-G":
        backbone_file = repository_root / initialization["shared_backbone_file"]
        initial_backbone = torch.load(
            backbone_file,
            map_location="cpu",
            weights_only=True,
        )
        initialization_record = {
            "a2_shared_backbone_state_sha256": initialization[
                "shared_backbone_state_sha256"
            ],
            "a2_shared_backbone_file": initialization["shared_backbone_file"],
            "a2_shared_backbone_file_sha256": initialization[
                "shared_backbone_file_sha256"
            ],
        }
    else:
        initial_backbone = {}
        initialization_record = {
            "a3_reference_a1_initial_state_sha256": initialization[
                "reference_a1_initial_state_sha256"
            ],
            "a3_initial_state_sha256": initialization["initial_state_sha256"],
        }
    return run_w3_experiment(
        **common,
        initial_backbone=initial_backbone,
        initialization_record=initialization_record,
    )


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    manifest = load_json_object(args.manifest_file)
    data_file = args.data_file or repository_root / manifest["dataset"]["default_path"]
    output = args.output_directory or repository_root / manifest["replay"][
        "default_output_directory"
    ]
    verification = verify_run_manifest(
        manifest,
        repository_root=repository_root,
        data_file=data_file,
    )
    commands = build_replay_commands(
        args.manifest_file,
        data_file=data_file,
        output_directory=output,
    )
    if not args.execute:
        print(
            json.dumps(
                {"verification": verification, "commands": commands},
                ensure_ascii=False,
                indent=2,
            )
        )
        print("dry_run_only=true")
        return
    result = execute_replay(
        manifest,
        repository_root=repository_root,
        data_file=Path(data_file),
        output_directory=Path(output),
    )
    print(f"completed_replay={manifest['run_id']}")
    print(f"result_file={result['result_file']}")
    print("test_set_used=false")


if __name__ == "__main__":
    main()
