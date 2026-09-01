"""Schema, integrity checks, and replay plans for frozen V2 runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from signal_modulation.data_integrity import sha256_file
from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_PROTOCOL_VERSION,
    V2_RUN_SEEDS,
    V2_SPLIT_SEED,
    V2_SPLIT_MANIFEST_SHA256,
)


RUN_MANIFEST_SCHEMA = "wireless-v2-run-manifest-v1"
RUN_CATALOG_SCHEMA = "wireless-v2-run-catalog-v1"
REPLAYABLE_CONFIGURATIONS = {
    "A0": ("W2", "SimpleCNN1D"),
    "A1": ("W2", "TemporalCNN1D"),
    "A2-G": ("W3", "GlobalPoolingTemporalCNN1D"),
    "A3": ("W3", "TemporalCNN1D"),
}


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _require_relative_file(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside the repository")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return value.lower()


def _require_git_commit(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{field} must be a full Git object ID")
    return value.lower()


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject a manifest that could change the frozen run identity or data boundary."""

    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported run manifest schema")
    configuration = manifest.get("configuration")
    if configuration not in REPLAYABLE_CONFIGURATIONS:
        raise ValueError("manifest configuration is not replayable")
    expected_stage, expected_model = REPLAYABLE_CONFIGURATIONS[configuration]
    if manifest.get("source_stage") != expected_stage:
        raise ValueError("configuration and source stage do not match")
    if manifest.get("model_name") != expected_model:
        raise ValueError("configuration and model do not match")
    run_seed = manifest.get("run_seed")
    if run_seed not in V2_RUN_SEEDS:
        raise ValueError("manifest run seed was not pre-registered")
    if manifest.get("run_id") != f"{expected_stage.lower()}-{configuration.lower()}-{run_seed}":
        raise ValueError("run_id does not match the frozen run identity")
    if manifest.get("scope") != "fixed_validation_only":
        raise ValueError("run manifest must remain validation-only")
    if manifest.get("test_set_used") is not False:
        raise ValueError("run manifest must not use the V1 test set")

    dataset = manifest.get("dataset")
    split = manifest.get("split")
    historical = manifest.get("historical_artifacts")
    replay = manifest.get("replay")
    if not all(isinstance(item, Mapping) for item in (dataset, split, historical, replay)):
        raise ValueError("manifest is missing a required object")
    if _require_sha256(dataset.get("sha256"), "dataset.sha256") != V2_DATASET_SHA256:
        raise ValueError("dataset identity does not match the frozen protocol")
    _require_relative_file(dataset.get("default_path"), "dataset.default_path")
    if (
        _require_sha256(split.get("manifest_sha256"), "split.manifest_sha256")
        != V2_SPLIT_MANIFEST_SHA256
    ):
        raise ValueError("split manifest identity does not match the frozen protocol")
    _require_relative_file(split.get("manifest_file"), "split.manifest_file")
    if split.get("split_seed") != 20260812:
        raise ValueError("split seed does not match the frozen protocol")

    for path_field, hash_field in (
        ("source_result_file", "source_result_sha256"),
        ("checkpoint_file", "checkpoint_sha256"),
    ):
        _require_relative_file(historical.get(path_field), f"historical_artifacts.{path_field}")
        _require_sha256(historical.get(hash_field), f"historical_artifacts.{hash_field}")
    _require_git_commit(
        historical.get("implementation_git_commit"),
        "historical_artifacts.implementation_git_commit",
    )
    if replay.get("entrypoint") != "scripts/replay_v2_run.py":
        raise ValueError("manifest replay entrypoint is not frozen")
    if replay.get("requires_explicit_execute") is not True:
        raise ValueError("manifest must require an explicit execute flag")

    training = manifest.get("training")
    if not isinstance(training, Mapping) or not isinstance(training.get("config"), Mapping):
        raise ValueError("manifest training config is missing")
    if training["config"].get("run_seed") != run_seed:
        raise ValueError("training config run seed does not match manifest")
    if training["config"].get("split_seed") != split.get("split_seed"):
        raise ValueError("training config split seed does not match manifest")

    initialization = manifest.get("initialization")
    if configuration in {"A0", "A1"}:
        if initialization is not None:
            raise ValueError(f"{configuration} must not declare W3 initialization")
    elif configuration == "A2-G":
        if not isinstance(initialization, Mapping):
            raise ValueError("A2-G shared-backbone initialization is missing")
        _require_relative_file(
            initialization.get("shared_backbone_file"),
            "initialization.shared_backbone_file",
        )
        _require_sha256(
            initialization.get("shared_backbone_file_sha256"),
            "initialization.shared_backbone_file_sha256",
        )
        _require_sha256(
            initialization.get("shared_backbone_state_sha256"),
            "initialization.shared_backbone_state_sha256",
        )
    elif configuration == "A3":
        if not isinstance(initialization, Mapping):
            raise ValueError("A3 initialization audit is missing")
        initial_state_sha256 = _require_sha256(
            initialization.get("initial_state_sha256"),
            "initialization.initial_state_sha256",
        )
        reference_sha256 = _require_sha256(
            initialization.get("reference_a1_initial_state_sha256"),
            "initialization.reference_a1_initial_state_sha256",
        )
        if initialization.get("matches_a1_initial_state") is not True:
            raise ValueError("A3 initialization must match its paired A1 run")
        if initial_state_sha256 != reference_sha256:
            raise ValueError("A3 and paired A1 initialization hashes do not match")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("manifest provenance is missing")
    _require_sha256(provenance.get("protocol_sha256"), "provenance.protocol_sha256")
    _require_relative_file(provenance.get("protocol_file"), "provenance.protocol_file")
    _require_git_commit(
        provenance.get("replay_code_git_commit"),
        "provenance.replay_code_git_commit",
    )
    dependencies = provenance.get("dependency_specs")
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("manifest dependency specification hashes are missing")
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, Mapping):
            raise ValueError("dependency specification must be an object")
        _require_relative_file(dependency.get("file"), f"dependency_specs[{index}].file")
        _require_sha256(dependency.get("sha256"), f"dependency_specs[{index}].sha256")


def _repository_file(repository_root: Path, relative_path: str) -> Path:
    safe_relative = _require_relative_file(relative_path, "repository file")
    candidate = (repository_root / safe_relative).resolve()
    try:
        candidate.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError("repository file escapes repository root") from error
    return candidate


def verify_run_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: str | Path,
    data_file: str | Path | None = None,
) -> dict[str, Any]:
    """Verify repository artifacts without reading signals or running a model."""

    validate_run_manifest(manifest)
    root = Path(repository_root).resolve()
    historical = manifest["historical_artifacts"]
    split = manifest["split"]
    provenance = manifest["provenance"]
    verified_files: list[str] = []

    checks = [
        (historical["source_result_file"], historical["source_result_sha256"]),
        (historical["checkpoint_file"], historical["checkpoint_sha256"]),
        (split["manifest_file"], split["manifest_sha256"]),
        (provenance["protocol_file"], provenance["protocol_sha256"]),
        *[(item["file"], item["sha256"]) for item in provenance["dependency_specs"]],
    ]
    for relative_path, expected_hash in checks:
        path = _repository_file(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_hash:
            raise ValueError(f"file hash does not match manifest: {relative_path}")
        verified_files.append(relative_path)

    split_manifest = load_json_object(root / split["manifest_file"])
    if split_manifest.get("protocol_version") != V2_PROTOCOL_VERSION:
        raise ValueError("split manifest protocol version does not match V2")
    split_definition = split_manifest.get("split")
    split_indices = split_manifest.get("indices")
    if not isinstance(split_definition, Mapping) or not isinstance(
        split_indices, Mapping
    ):
        raise ValueError("split manifest is missing split or index metadata")
    if split_definition.get("split_seed") != V2_SPLIT_SEED:
        raise ValueError("split manifest seed does not match V2")
    index_filename = split_indices.get("filename")
    if index_filename != "split_indices.npz":
        raise ValueError("split index archive filename is not frozen")
    index_sha256 = _require_sha256(
        split_indices.get("file_sha256"),
        "split_manifest.indices.file_sha256",
    )
    split_directory = Path(split["manifest_file"]).parent
    index_relative_path = (split_directory / index_filename).as_posix()
    index_path = _repository_file(root, index_relative_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if sha256_file(index_path) != index_sha256:
        raise ValueError("split index archive hash does not match split manifest")
    verified_files.append(index_relative_path)

    source_result = load_json_object(root / historical["source_result_file"])
    if (
        source_result.get("status") != "completed"
        or source_result.get("configuration") != manifest["configuration"]
        or source_result.get("run_seed") != manifest["run_seed"]
        or source_result.get("test_set_used") is not False
        or source_result.get("checkpoint_sha256") != historical["checkpoint_sha256"]
        or source_result.get("implementation_git_commit")
        != historical["implementation_git_commit"]
        or source_result.get("initialization") != manifest.get("initialization")
    ):
        raise ValueError("historical result does not match run manifest")

    initialization = manifest.get("initialization")
    if manifest["configuration"] == "A2-G":
        backbone_relative_path = initialization["shared_backbone_file"]
        backbone_path = _repository_file(root, backbone_relative_path)
        if not backbone_path.is_file():
            raise FileNotFoundError(backbone_path)
        if sha256_file(backbone_path) != initialization[
            "shared_backbone_file_sha256"
        ]:
            raise ValueError("shared initial backbone hash does not match manifest")
        verified_files.append(backbone_relative_path)

    data_status = "not_checked"
    if data_file is not None:
        path = Path(data_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != manifest["dataset"]["sha256"]:
            raise ValueError("data file does not match run manifest")
        data_status = "sha256_verified_without_loading_signals"

    return {
        "run_id": manifest["run_id"],
        "configuration": manifest["configuration"],
        "run_seed": manifest["run_seed"],
        "verified_repository_file_count": len(verified_files),
        "data_status": data_status,
        "test_set_used": False,
        "ready_for_explicit_execute": data_status.startswith("sha256_verified"),
    }


def build_replay_commands(
    manifest_file: str | Path,
    *,
    data_file: str | Path,
    output_directory: str | Path,
) -> dict[str, list[str]]:
    """Build non-executing and explicit-execution commands for one manifest."""

    base = [
        ".\\.venv\\Scripts\\python.exe",
        "scripts\\replay_v2_run.py",
        str(manifest_file),
        "--data-file",
        str(data_file),
        "--output-directory",
        str(output_directory),
    ]
    return {"dry_run": base, "execute": [*base, "--execute"]}
