"""Run W3 A2-G/A3 and summarize against the reused W2 A2-T baseline."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from signal_modulation.data_integrity import sha256_file
from signal_modulation.evaluation import evaluate_classifier
from signal_modulation.experiment import prepare_new_run_directory, write_json_atomic
from signal_modulation.model import (
    GlobalPoolingTemporalCNN1D,
    TemporalCNN1D,
    count_trainable_parameters,
)
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import RadioMLTorchDataset
from signal_modulation.training import EpochRecord, fit
from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_PROTOCOL_VERSION,
    V2_RUN_SEEDS,
    V2_SPLIT_MANIFEST_SHA256,
    V2ExperimentConfig,
    configure_v2_run_randomness,
    create_v2_development_loaders,
)
from signal_modulation.v2_statistics import summarize_w3_results
from signal_modulation.w3_ablation import (
    W3_LATENCY_REFERENCE_SEED,
    W3_NEW_CONFIGURATIONS,
    architecture_profile,
    benchmark_inference_latency,
    create_w3_model,
    reconstruct_a2t_initial_backbone,
    state_dict_sha256,
    verify_dropout_initialization,
)


W2_SUMMARY_SHA256 = "7c5f2513ae78903abad0781801e4f5577ec009a7203cd7a56ac5503bbaa1c00f"
W3_MATRIX = (("A2-G", "GlobalPoolingTemporalCNN1D"), ("A3", "TemporalCNN1D"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("w2_artifact_directory", type=Path)
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
        raise RuntimeError("commit W3 implementation before running formal experiments")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_relative(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _print_epoch(configuration: str, run_seed: int, record: EpochRecord) -> None:
    print(
        f"configuration={configuration} run_seed={run_seed} epoch={record.epoch} "
        f"validation_loss={record.validation.loss:.6f}",
        flush=True,
    )


def load_reused_a2t_records(w2_artifact_directory: Path) -> list[dict[str, Any]]:
    """Map the five committed W2 A1 runs to the identical A2-T baseline."""

    summary_path = w2_artifact_directory / "w2_summary.json"
    if sha256_file(summary_path) != W2_SUMMARY_SHA256:
        raise ValueError("W2 summary does not match the frozen A2-T reuse source")
    records = []
    for run_seed in V2_RUN_SEEDS:
        result_path = (
            w2_artifact_directory
            / "A1"
            / f"run_seed_{run_seed}"
            / "validation_result.json"
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("configuration") != "A1"
            or payload.get("run_seed") != run_seed
            or payload.get("status") != "completed"
            or payload.get("test_set_used") is not False
        ):
            raise ValueError("W2 A1 result is not eligible for A2-T reuse")
        reused = dict(payload)
        reused["configuration"] = "A2-T"
        reused["source_configuration"] = "A1"
        reused["reused_from_w2"] = True
        records.append(reused)
    return records


def create_initialization_artifacts(
    output_root: Path,
    *,
    repository_root: Path,
    num_classes: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]], Path]:
    """Save deterministic A2-T initial backbones and audit A3 initialization."""

    directory = prepare_new_run_directory(output_root / "initial_backbones")
    states: dict[int, dict[str, torch.Tensor]] = {}
    records: dict[int, dict[str, Any]] = {}
    for run_seed in V2_RUN_SEEDS:
        backbone, state_hash = reconstruct_a2t_initial_backbone(
            run_seed,
            num_classes=num_classes,
        )
        path = directory / f"run_seed_{run_seed}.pt"
        torch.save(backbone, path)
        baseline_hash, a3_hash = verify_dropout_initialization(
            run_seed,
            num_classes=num_classes,
        )
        states[run_seed] = backbone
        records[run_seed] = {
            "run_seed": run_seed,
            "a2_shared_backbone_state_sha256": state_hash,
            "a2_shared_backbone_file": _repository_relative(path, repository_root),
            "a2_shared_backbone_file_sha256": sha256_file(path),
            "a3_reference_a1_initial_state_sha256": baseline_hash,
            "a3_initial_state_sha256": a3_hash,
            "a3_initial_state_matches_a1": baseline_hash == a3_hash,
        }
    manifest_path = write_json_atomic(
        directory / "initialization_manifest.json",
        {
            "schema_version": "wireless-v2-w3-initialization-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "records": [records[seed] for seed in V2_RUN_SEEDS],
            "a2t_source_note": (
                "A2-T initial backbones are deterministically reconstructed from the "
                "same run seed and unchanged TemporalCNN initialization used in W2."
            ),
        },
    )
    return states, records, manifest_path


def run_one_experiment(
    *,
    configuration: str,
    run_seed: int,
    dataset: RadioMLTorchDataset,
    dataset_sha256: str,
    split_manifest: Path,
    output_root: Path,
    repository_root: Path,
    implementation_commit: str,
    device: torch.device,
    initial_backbone: dict[str, torch.Tensor],
    initialization_record: dict[str, Any],
) -> dict[str, Any]:
    run_directory = prepare_new_run_directory(
        output_root / configuration / f"run_seed_{run_seed}"
    )
    checkpoint_path = run_directory / "best_checkpoint.pt"
    result_path = run_directory / "validation_result.json"
    failure_path = run_directory / "failure.json"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    try:
        config = V2ExperimentConfig(run_seed=run_seed)
        configure_v2_run_randomness(config)
        torch.cuda.reset_peak_memory_stats()
        loaders = create_v2_development_loaders(
            dataset,
            split_manifest,
            config,
            dataset_sha256=dataset_sha256,
            pin_memory=True,
        )
        model = create_w3_model(
            configuration,
            num_classes=len(dataset.modulations),
            initial_backbone=(initial_backbone if configuration == "A2-G" else None),
        )
        if configuration == "A2-G":
            loaded_hash = state_dict_sha256(model.features.state_dict())
            if loaded_hash != initialization_record["a2_shared_backbone_state_sha256"]:
                raise RuntimeError("A2-G did not load the frozen shared backbone")
            initialization = {
                "shared_backbone_state_sha256": loaded_hash,
                "shared_backbone_file": initialization_record[
                    "a2_shared_backbone_file"
                ],
                "shared_backbone_file_sha256": initialization_record[
                    "a2_shared_backbone_file_sha256"
                ],
            }
        else:
            initial_hash = state_dict_sha256(model.state_dict())
            if initial_hash != initialization_record["a3_initial_state_sha256"]:
                raise RuntimeError("A3 initial state does not match its audit record")
            initialization = {
                "initial_state_sha256": initial_hash,
                "reference_a1_initial_state_sha256": initialization_record[
                    "a3_reference_a1_initial_state_sha256"
                ],
                "matches_a1_initial_state": True,
            }

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        training_result = fit(
            model,
            loaders.train,
            loaders.validation,
            optimizer,
            device=device,
            epochs=config.epochs,
            patience=config.patience,
            min_delta=config.min_delta,
            checkpoint_path=checkpoint_path,
            on_epoch=lambda record: _print_epoch(configuration, run_seed, record),
        )
        validation = evaluate_classifier(
            model,
            loaders.validation,
            device=device,
            num_classes=len(dataset.modulations),
        )
        if validation.sample_count != 33_000:
            raise RuntimeError("W3 validation did not consume exactly 33,000 samples")

        payload: dict[str, Any] = {
            "schema_version": "wireless-v2-w3-run-v1",
            "stage": "W3",
            "status": "completed",
            "configuration": configuration,
            "run_seed": run_seed,
            "scope": "fixed_validation_only",
            "test_set_used": False,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_git_commit": implementation_commit,
            "protocol": {
                "version": V2_PROTOCOL_VERSION,
                "split_manifest_sha256": loaders.manifest_sha256,
                "split_seed": loaders.split_seed,
                "run_seed": loaders.run_seed,
                "selection_criterion": "lowest_validation_loss",
                "test_index_count_metadata_only": loaders.test_index_count,
                "test_index_sha256_metadata_only": loaders.test_index_sha256,
            },
            "config": asdict(config),
            "controlled_comparison": (
                "shared_backbone_near_parameter_budget_aggregation"
                if configuration == "A2-G"
                else "dropout_p0.3_vs_p0.0"
            ),
            "initialization": initialization,
            "optimizer": {
                "name": "Adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "scheduler": None,
            },
            "dataset": {
                "name": "RadioML 2016.10A",
                "pickle_sha256": dataset_sha256,
                "total_samples": len(dataset),
                "train_samples": len(loaders.train.dataset),
                "validation_samples": len(loaders.validation.dataset),
                "sealed_test_samples_metadata_only": loaders.test_index_count,
                "modulations": list(dataset.modulations),
                "class_to_index": dataset.class_to_index,
            },
            "model": {
                "name": type(model).__name__,
                "trainable_parameters": count_trainable_parameters(model),
                "architecture_profile": architecture_profile(configuration),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(0),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "training": {
                "history": [
                    {
                        "epoch": record.epoch,
                        "train": asdict(record.train),
                        "validation": asdict(record.validation),
                    }
                    for record in training_result.history
                ],
                "best_epoch": training_result.best_epoch,
                "best_validation_loss": training_result.best_validation_loss,
                "stopped_early": training_result.stopped_early,
            },
            "validation": {
                "loss": validation.loss,
                "sample_count": validation.sample_count,
                "accuracy": validation.classification.accuracy,
                "macro_precision": validation.classification.macro_precision,
                "macro_recall": validation.classification.macro_recall,
                "macro_f1": validation.classification.macro_f1,
                "per_class_precision": validation.classification.per_class_precision,
                "per_class_recall": validation.classification.per_class_recall,
                "per_class_f1": validation.classification.per_class_f1,
                "confusion_matrix": validation.classification.confusion_matrix,
                "by_snr": [asdict(item) for item in validation.by_snr],
            },
            "checkpoint_file": _repository_relative(checkpoint_path, repository_root),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "result_file": _repository_relative(result_path, repository_root),
        }
        write_json_atomic(result_path, payload)
        return payload
    except Exception as error:
        write_json_atomic(
            failure_path,
            {
                "schema_version": "wireless-v2-w3-failure-v1",
                "stage": "W3",
                "status": "failed",
                "configuration": configuration,
                "run_seed": run_seed,
                "started_at_utc": started_at,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "implementation_git_commit": implementation_commit,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "retry_permitted_only_with_same_configuration_and_seed": True,
            },
        )
        raise


def _load_model_checkpoint(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def benchmark_frozen_architectures(
    *,
    repository_root: Path,
    w2_artifact_directory: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark a preselected seed checkpoint for each W3 configuration."""

    seed = W3_LATENCY_REFERENCE_SEED
    checkpoint_paths = {
        "A2-T": w2_artifact_directory
        / "A1"
        / f"run_seed_{seed}"
        / "best_checkpoint.pt",
        "A2-G": output_root
        / "A2-G"
        / f"run_seed_{seed}"
        / "best_checkpoint.pt",
        "A3": output_root / "A3" / f"run_seed_{seed}" / "best_checkpoint.pt",
    }
    models = {
        "A2-T": TemporalCNN1D(num_classes=11, dropout=0.3),
        "A2-G": GlobalPoolingTemporalCNN1D(num_classes=11),
        "A3": TemporalCNN1D(num_classes=11, dropout=0.0),
    }
    return {
        "reference_run_seed": seed,
        "selection_rule": "lowest pre-registered run seed; not selected by score",
        "results": {
            configuration: {
                "checkpoint_file": _repository_relative(
                    checkpoint_paths[configuration], repository_root
                ),
                **benchmark_inference_latency(
                    _load_model_checkpoint(model, checkpoint_paths[configuration]),
                    device=device,
                ),
            }
            for configuration, model in models.items()
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal W3 matrix requires the verified CUDA GPU environment")
    repository_root = Path(__file__).resolve().parents[1]
    implementation_commit = clean_repository_commit(repository_root)
    dataset_sha256 = sha256_file(args.pickle_file)
    if dataset_sha256 != V2_DATASET_SHA256:
        raise ValueError("dataset SHA-256 does not match the frozen V2 protocol")
    if sha256_file(args.split_manifest) != V2_SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA-256 does not match the frozen V2 protocol")
    reused_records = load_reused_a2t_records(args.w2_artifact_directory)

    output_root = prepare_new_run_directory(args.output_directory)
    initial_states, initialization_records, initialization_manifest = (
        create_initialization_artifacts(
            output_root,
            repository_root=repository_root,
            num_classes=11,
        )
    )
    profiles = {
        configuration: architecture_profile(configuration)
        for configuration in ("A2-T", "A2-G", "A3")
    }
    a2t_parameters = profiles["A2-T"]["trainable_parameters"]
    a2g_parameters = profiles["A2-G"]["trainable_parameters"]
    capacity_control = {
        "full_parameter_absolute_difference": abs(a2t_parameters - a2g_parameters),
        "full_parameter_percent_difference": (
            abs(a2t_parameters - a2g_parameters) / a2t_parameters * 100.0
        ),
        "target_at_most_percent": 1.0,
        "hard_limit_percent": 2.0,
        "target_satisfied": abs(a2t_parameters - a2g_parameters)
        / a2t_parameters
        * 100.0
        <= 1.0,
        "safe_description": (
            "shared-backbone comparison of temporal-position aggregation and global "
            "pooling under a near-equal parameter budget"
        ),
        "strict_single_variable_ablation": False,
    }
    matrix_lock_path = write_json_atomic(
        output_root / "matrix_lock.json",
        {
            "schema_version": "wireless-v2-w3-matrix-lock-v1",
            "stage": "W3",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_git_commit": implementation_commit,
            "protocol_version": V2_PROTOCOL_VERSION,
            "dataset_sha256": dataset_sha256,
            "split_manifest_sha256": V2_SPLIT_MANIFEST_SHA256,
            "w2_summary_sha256": W2_SUMMARY_SHA256,
            "run_seeds": list(V2_RUN_SEEDS),
            "reused_configuration": "A2-T from W2 A1; no retraining",
            "new_configurations": list(W3_NEW_CONFIGURATIONS),
            "planned_new_training_run_count": len(V2_RUN_SEEDS) * 2,
            "profiles": profiles,
            "capacity_control": capacity_control,
            "scope": "fixed_validation_only",
            "test_set_used": False,
        },
    )

    grouped_signals = load_restricted_radioml_pickle(args.pickle_file)
    dataset = RadioMLTorchDataset(grouped_signals)
    device = torch.device("cuda")
    new_records: list[dict[str, Any]] = []
    for run_seed in V2_RUN_SEEDS:
        for configuration, _model_name in W3_MATRIX:
            print(
                f"starting configuration={configuration} run_seed={run_seed}",
                flush=True,
            )
            record = run_one_experiment(
                configuration=configuration,
                run_seed=run_seed,
                dataset=dataset,
                dataset_sha256=dataset_sha256,
                split_manifest=args.split_manifest,
                output_root=output_root,
                repository_root=repository_root,
                implementation_commit=implementation_commit,
                device=device,
                initial_backbone=initial_states[run_seed],
                initialization_record=initialization_records[run_seed],
            )
            new_records.append(record)
            print(
                f"completed configuration={configuration} run_seed={run_seed} "
                f"elapsed_seconds={record['runtime']['elapsed_seconds']:.2f}",
                flush=True,
            )

    summary = summarize_w3_results(
        reused_records + new_records,
        run_seeds=V2_RUN_SEEDS,
    )
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_git_commit": implementation_commit,
            "protocol_version": V2_PROTOCOL_VERSION,
            "dataset_sha256": dataset_sha256,
            "split_manifest_sha256": V2_SPLIT_MANIFEST_SHA256,
            "w2_summary_sha256": W2_SUMMARY_SHA256,
            "matrix_lock_file": _repository_relative(
                matrix_lock_path, repository_root
            ),
            "initialization_manifest_file": _repository_relative(
                initialization_manifest, repository_root
            ),
            "architecture_profiles": profiles,
            "aggregation_capacity_control": capacity_control,
            "latency_benchmark": benchmark_frozen_architectures(
                repository_root=repository_root,
                w2_artifact_directory=args.w2_artifact_directory,
                output_root=output_root,
                device=device,
            ),
        }
    )
    summary_path = write_json_atomic(output_root / "w3_summary.json", summary)
    print(f"summary={summary_path}", flush=True)
    print("test_set_used=false", flush=True)


if __name__ == "__main__":
    main()
