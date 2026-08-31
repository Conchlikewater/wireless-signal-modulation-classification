"""Run the pre-registered W2 A0/A1 paired experiment matrix on validation only."""

from __future__ import annotations

import argparse
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
    SimpleCNN1D,
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
from signal_modulation.v2_statistics import summarize_w2_results


W2_MATRIX = (("A0", "SimpleCNN1D"), ("A1", "TemporalCNN1D"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def create_w2_model(configuration: str, *, num_classes: int) -> torch.nn.Module:
    """Construct only the two architectures pre-registered for W2."""

    if configuration == "A0":
        return SimpleCNN1D(num_classes=num_classes)
    if configuration == "A1":
        return TemporalCNN1D(num_classes=num_classes)
    raise ValueError(f"unsupported W2 configuration: {configuration}")


def clean_repository_commit(repository_root: Path) -> str:
    """Resolve the implementation commit only when no source change is pending."""

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("commit W2 implementation before running formal experiments")
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
) -> dict[str, Any]:
    """Train and evaluate one registered configuration/seed pair on validation."""

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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loaders = create_v2_development_loaders(
            dataset,
            split_manifest,
            config,
            dataset_sha256=dataset_sha256,
            pin_memory=device.type == "cuda",
        )
        model = create_w2_model(
            configuration,
            num_classes=len(dataset.modulations),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        result = fit(
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
            raise RuntimeError("W2 validation did not consume exactly 33,000 samples")

        elapsed_seconds = time.perf_counter() - started
        payload: dict[str, Any] = {
            "schema_version": "wireless-v2-w2-run-v1",
            "stage": "W2",
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
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(0),
                "elapsed_seconds": elapsed_seconds,
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
                    for record in result.history
                ],
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "stopped_early": result.stopped_early,
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
            "checkpoint_file": _repository_relative(
                checkpoint_path, repository_root
            ),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "result_file": _repository_relative(result_path, repository_root),
        }
        write_json_atomic(result_path, payload)
        return payload
    except Exception as error:
        write_json_atomic(
            failure_path,
            {
                "schema_version": "wireless-v2-w2-failure-v1",
                "stage": "W2",
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal W2 matrix requires the verified CUDA GPU environment")
    repository_root = Path(__file__).resolve().parents[1]
    implementation_commit = clean_repository_commit(repository_root)
    dataset_sha256 = sha256_file(args.pickle_file)
    if dataset_sha256 != V2_DATASET_SHA256:
        raise ValueError("dataset SHA-256 does not match the frozen V2 protocol")
    if sha256_file(args.split_manifest) != V2_SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA-256 does not match the frozen V2 protocol")

    output_root = prepare_new_run_directory(args.output_directory)
    matrix_lock_path = output_root / "matrix_lock.json"
    write_json_atomic(
        matrix_lock_path,
        {
            "schema_version": "wireless-v2-w2-matrix-lock-v1",
            "stage": "W2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_git_commit": implementation_commit,
            "protocol_version": V2_PROTOCOL_VERSION,
            "dataset_sha256": dataset_sha256,
            "split_manifest_sha256": V2_SPLIT_MANIFEST_SHA256,
            "run_seeds": list(V2_RUN_SEEDS),
            "configurations": [name for name, _model in W2_MATRIX],
            "planned_run_count": len(V2_RUN_SEEDS) * len(W2_MATRIX),
            "scope": "fixed_validation_only",
            "test_set_used": False,
        },
    )

    grouped_signals = load_restricted_radioml_pickle(args.pickle_file)
    dataset = RadioMLTorchDataset(grouped_signals)
    device = torch.device("cuda")
    records: list[dict[str, Any]] = []
    for run_seed in V2_RUN_SEEDS:
        for configuration, _model_name in W2_MATRIX:
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
            )
            records.append(record)
            print(
                f"completed configuration={configuration} run_seed={run_seed} "
                f"elapsed_seconds={record['runtime']['elapsed_seconds']:.2f}",
                flush=True,
            )

    summary = summarize_w2_results(records, run_seeds=V2_RUN_SEEDS)
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_git_commit": implementation_commit,
            "protocol_version": V2_PROTOCOL_VERSION,
            "dataset_sha256": dataset_sha256,
            "split_manifest_sha256": V2_SPLIT_MANIFEST_SHA256,
            "matrix_lock_file": _repository_relative(
                matrix_lock_path, repository_root
            ),
        }
    )
    summary_path = write_json_atomic(output_root / "w2_summary.json", summary)
    print(f"summary={summary_path}", flush=True)
    print("test_set_used=false", flush=True)


if __name__ == "__main__":
    main()
