"""Run the preregistered five-seed A2-L validation experiment."""

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
from typing import Any, Mapping

import numpy as np
import torch

from signal_modulation.config_semantics import effective_model_config
from signal_modulation.data_integrity import sha256_file
from signal_modulation.evaluation import (
    calculate_class_snr_accuracy,
    calculate_classification_metrics,
    calculate_snr_metrics,
    calculate_snr_segment_metrics,
    collect_classifier_predictions,
)
from signal_modulation.experiment import prepare_new_run_directory, write_json_atomic
from signal_modulation.model import count_trainable_parameters
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import RadioMLTorchDataset
from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_PROTOCOL_VERSION,
    V2_RUN_SEEDS,
    V2_SPLIT_MANIFEST_SHA256,
    V2ExperimentConfig,
    configure_v2_run_randomness,
    create_v2_development_loaders,
)
from signal_modulation.v2_statistics import descriptive_summary
from signal_modulation.w3_ablation import (
    architecture_profile,
    benchmark_inference_latency,
    create_w3_model,
    state_dict_sha256,
)


W5_CONFIGURATION = "A2-L"
SNR_SEGMENTS = {"low_snr_le_-10": (None, -10.0), "high_snr_ge_10": (10.0, None)}
HYPOTHESIS_RELATIVE_PATH = Path("experiments/v2/w5/HYPOTHESIS.md")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle-file",
        type=Path,
        default=root / "data/raw/RML2016.10a/RML2016.10a_dict.pkl",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=root / "manifests/v2/split_manifest.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=root / "experiments/v2/w5",
    )
    return parser.parse_args()


def clean_repository_commit(repository_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("formal A2-L runs require a clean Git working tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _load_initialization_records(repository_root: Path) -> dict[int, dict[str, Any]]:
    path = repository_root / "experiments/v2/w3/initial_backbones/initialization_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {int(record["run_seed"]): record for record in payload["records"]}
    if set(records) != set(V2_RUN_SEEDS):
        raise ValueError("shared initialization manifest does not cover frozen seeds")
    return records


def _print_epoch(configuration: str, run_seed: int, record: Any) -> None:
    print(
        f"configuration={configuration} run_seed={run_seed} epoch={record.epoch} "
        f"validation_loss={record.validation.loss:.6f}",
        flush=True,
    )


def run_one_experiment(
    *,
    run_seed: int,
    dataset: RadioMLTorchDataset,
    dataset_sha256: str,
    split_manifest: Path,
    output_root: Path,
    repository_root: Path,
    implementation_commit: str,
    device: torch.device,
    initialization_record: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = f"w5-a2-l-{run_seed}"
    run_directory = prepare_new_run_directory(
        output_root / W5_CONFIGURATION / f"run_seed_{run_seed}"
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
        backbone_path = repository_root / initialization_record["a2_shared_backbone_file"]
        if sha256_file(backbone_path) != initialization_record[
            "a2_shared_backbone_file_sha256"
        ]:
            raise ValueError("shared backbone file hash mismatch")
        initial_backbone = torch.load(
            backbone_path,
            map_location="cpu",
            weights_only=True,
        )
        if state_dict_sha256(initial_backbone) != initialization_record[
            "a2_shared_backbone_state_sha256"
        ]:
            raise ValueError("shared backbone state hash mismatch")
        model = create_w3_model(
            W5_CONFIGURATION,
            num_classes=len(dataset.modulations),
            initial_backbone=initial_backbone,
        )
        if state_dict_sha256(model.features.state_dict()) != initialization_record[
            "a2_shared_backbone_state_sha256"
        ]:
            raise RuntimeError("A2-L did not load the paired initial backbone")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        from signal_modulation.training import fit

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
            on_epoch=lambda record: _print_epoch(W5_CONFIGURATION, run_seed, record),
        )
        collected = collect_classifier_predictions(
            model,
            loaders.validation,
            device=device,
            num_classes=len(dataset.modulations),
        )
        if collected.sample_count != 33_000:
            raise RuntimeError("A2-L validation did not consume exactly 33,000 samples")
        classification = calculate_classification_metrics(
            collected.labels,
            collected.predictions,
            num_classes=len(dataset.modulations),
        )
        by_snr = calculate_snr_metrics(
            collected.labels,
            collected.predictions,
            collected.snrs,
            num_classes=len(dataset.modulations),
        )
        class_snr = calculate_class_snr_accuracy(
            collected.labels,
            collected.predictions,
            collected.snrs,
            num_classes=len(dataset.modulations),
        )
        segments = calculate_snr_segment_metrics(
            collected.labels,
            collected.predictions,
            collected.snrs,
            num_classes=len(dataset.modulations),
            segments=SNR_SEGMENTS,
        )
        recorded_config = asdict(config)
        payload: dict[str, Any] = {
            "schema_version": "wireless-v2-w5-run-v1",
            "stage": "W5",
            "status": "completed",
            "run_id": run_id,
            "configuration": W5_CONFIGURATION,
            "run_seed": run_seed,
            "scope": "fixed_validation_only",
            "test_set_used": False,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_git_commit": implementation_commit,
            "protocol": {
                "version": V2_PROTOCOL_VERSION,
                "hypothesis_file": HYPOTHESIS_RELATIVE_PATH.as_posix(),
                "hypothesis_sha256": sha256_file(repository_root / HYPOTHESIS_RELATIVE_PATH),
                "split_manifest_sha256": loaders.manifest_sha256,
                "split_seed": loaders.split_seed,
                "run_seed": loaders.run_seed,
                "selection_criterion": "lowest_validation_loss",
                "test_index_count_metadata_only": loaders.test_index_count,
                "test_index_sha256_metadata_only": loaders.test_index_sha256,
            },
            "config": recorded_config,
            "effective_model_config": {
                **effective_model_config(
                    W5_CONFIGURATION,
                    run_id=run_id,
                    recorded_config=recorded_config,
                ),
                "lstm_hidden_size": 127,
                "lstm_steps": 32,
                "lstm_layers": 1,
                "bidirectional": False,
            },
            "controlled_comparison": (
                "shared_initial_backbone_near_parameter_budget_lstm_aggregation"
            ),
            "initialization": {
                "shared_backbone_state_sha256": initialization_record[
                    "a2_shared_backbone_state_sha256"
                ],
                "shared_backbone_file": initialization_record[
                    "a2_shared_backbone_file"
                ],
                "shared_backbone_file_sha256": initialization_record[
                    "a2_shared_backbone_file_sha256"
                ],
                "backbone_trainable": True,
            },
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
                "architecture_profile": architecture_profile(W5_CONFIGURATION),
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
                "loss": collected.loss,
                "sample_count": collected.sample_count,
                "accuracy": classification.accuracy,
                "macro_precision": classification.macro_precision,
                "macro_recall": classification.macro_recall,
                "macro_f1": classification.macro_f1,
                "per_class_precision": classification.per_class_precision,
                "per_class_recall": classification.per_class_recall,
                "per_class_f1": classification.per_class_f1,
                "confusion_matrix": classification.confusion_matrix,
                "by_snr": [asdict(item) for item in by_snr],
                "class_snr_accuracy": [list(row) for row in class_snr.accuracy],
                "class_snr_sample_counts": [list(row) for row in class_snr.sample_counts],
                "snr_segments": {
                    name: {
                        "accuracy": metrics.accuracy,
                        "macro_f1": metrics.macro_f1,
                        "sample_count": sum(sum(row) for row in metrics.confusion_matrix),
                        "confusion_matrix": metrics.confusion_matrix,
                    }
                    for name, metrics in segments.items()
                },
            },
            "checkpoint_file": _relative(checkpoint_path, repository_root),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "result_file": _relative(result_path, repository_root),
        }
        write_json_atomic(result_path, payload)
        return payload
    except Exception as error:
        write_json_atomic(
            failure_path,
            {
                "schema_version": "wireless-v2-w5-failure-v1",
                "stage": "W5",
                "status": "failed",
                "configuration": W5_CONFIGURATION,
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


def _comparator_records(repository_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    directories = {
        "A2-T": repository_root / "experiments/v2/w2/A1",
        "A2-G": repository_root / "experiments/v2/w3/A2-G",
    }
    return {
        name: {
            seed: json.loads(
                (directory / f"run_seed_{seed}/validation_result.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in V2_RUN_SEEDS
        }
        for name, directory in directories.items()
    }


def _paired_summary(
    a2l: Mapping[int, Mapping[str, Any]],
    comparator: Mapping[int, Mapping[str, Any]],
    *,
    comparator_name: str,
) -> dict[str, Any]:
    per_seed = []
    for seed in V2_RUN_SEEDS:
        positive = a2l[seed]
        negative = comparator[seed]
        per_seed.append(
            {
                "run_seed": seed,
                "validation_accuracy_delta": positive["validation"]["accuracy"]
                - negative["validation"]["accuracy"],
                "validation_macro_f1_delta": positive["validation"]["macro_f1"]
                - negative["validation"]["macro_f1"],
            }
        )
    return {
        "direction": f"A2-L_minus_{comparator_name}",
        "n_pairs": len(per_seed),
        "per_seed": per_seed,
        "validation_accuracy": descriptive_summary(
            [item["validation_accuracy_delta"] for item in per_seed]
        ),
        "validation_macro_f1": descriptive_summary(
            [item["validation_macro_f1_delta"] for item in per_seed]
        ),
    }


def _segment_paired_summary(
    a2l: Mapping[int, Mapping[str, Any]],
    analysis: Mapping[str, Any],
    *,
    comparator_name: str,
) -> dict[str, Any]:
    comparator_runs = {
        int(record["run_seed"]): record
        for record in analysis["arms"][comparator_name]["runs"]
    }
    result: dict[str, Any] = {}
    for segment in SNR_SEGMENTS:
        per_seed = []
        for seed in V2_RUN_SEEDS:
            positive = a2l[seed]["validation"]["snr_segments"][segment]
            negative = comparator_runs[seed]["segments"][segment]
            per_seed.append(
                {
                    "run_seed": seed,
                    "accuracy_delta": positive["accuracy"] - negative["accuracy"],
                    "macro_f1_delta": positive["macro_f1"] - negative["macro_f1"],
                }
            )
        result[segment] = {
            "n_pairs": len(per_seed),
            "per_seed": per_seed,
            "accuracy": descriptive_summary([item["accuracy_delta"] for item in per_seed]),
            "macro_f1": descriptive_summary([item["macro_f1_delta"] for item in per_seed]),
            "positive_macro_f1_seed_count": sum(
                item["macro_f1_delta"] > 0 for item in per_seed
            ),
        }
    return result


def _per_snr_summary(
    records: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize deterministic SNR bins across the five frozen run seeds."""

    by_seed = {
        seed: {float(item["snr"]): item for item in record["validation"]["by_snr"]}
        for seed, record in records.items()
    }
    snrs = sorted(by_seed[V2_RUN_SEEDS[0]])
    if any(sorted(values) != snrs for values in by_seed.values()):
        raise ValueError("A2-L runs do not contain the same SNR bins")
    return [
        {
            "snr": snr,
            "accuracy": descriptive_summary(
                [by_seed[seed][snr]["accuracy"] for seed in V2_RUN_SEEDS]
            ),
            "macro_f1": descriptive_summary(
                [by_seed[seed][snr]["macro_f1"] for seed in V2_RUN_SEEDS]
            ),
            "sample_count_per_seed": by_seed[V2_RUN_SEEDS[0]][snr]["sample_count"],
        }
        for snr in snrs
    ]


def evaluate_preregistered_hypothesis(
    paired_segments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    high_passes = []
    low_passes = []
    details = {}
    for comparator, segments in paired_segments.items():
        high = segments["high_snr_ge_10"]
        low = segments["low_snr_le_-10"]
        high_mean_pp = high["macro_f1"]["mean"] * 100.0
        low_mean_pp = low["macro_f1"]["mean"] * 100.0
        high_pass = high_mean_pp >= 1.0 and high["positive_macro_f1_seed_count"] >= 4
        low_pass = -1.0 <= low_mean_pp <= 0.5
        high_passes.append(high_pass)
        low_passes.append(low_pass)
        details[comparator] = {
            "high_macro_f1_mean_delta_pp": high_mean_pp,
            "high_positive_seed_count": high["positive_macro_f1_seed_count"],
            "low_macro_f1_mean_delta_pp": low_mean_pp,
            "high_rule_passed": high_pass,
            "low_rule_passed": low_pass,
        }
    if all(high_passes) and all(low_passes):
        verdict = "supported"
    elif any(
        detail["high_macro_f1_mean_delta_pp"] >= 0.5
        and detail["high_positive_seed_count"] >= 4
        for detail in details.values()
    ):
        verdict = "partially_supported"
    else:
        verdict = "not_supported"
    return {"verdict": verdict, "comparators": details}


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if not torch.cuda.is_available():
        raise RuntimeError("formal A2-L runs require CUDA")
    implementation_commit = clean_repository_commit(repository_root)
    if sha256_file(args.pickle_file) != V2_DATASET_SHA256:
        raise ValueError("dataset SHA-256 does not match V2")
    if sha256_file(args.split_manifest) != V2_SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA-256 does not match V2")
    output_root = args.output_directory.resolve()
    if not output_root.is_dir() or not (output_root / "HYPOTHESIS.md").is_file():
        raise ValueError("W5 output root must contain the committed HYPOTHESIS.md")
    for reserved in (output_root / W5_CONFIGURATION, output_root / "matrix_lock.json", output_root / "w5_summary.json"):
        if reserved.exists():
            raise FileExistsError(f"formal W5 output already exists: {reserved}")
    hypothesis_sha256 = sha256_file(repository_root / HYPOTHESIS_RELATIVE_PATH)
    profile = architecture_profile(W5_CONFIGURATION)
    a1_parameters = architecture_profile("A2-T")["trainable_parameters"]
    parameter_difference = profile["trainable_parameters"] - a1_parameters
    capacity = {
        "a2t_trainable_parameters": a1_parameters,
        "a2l_trainable_parameters": profile["trainable_parameters"],
        "signed_difference": parameter_difference,
        "absolute_percent_difference": abs(parameter_difference) / a1_parameters * 100.0,
        "target_at_most_percent": 1.0,
        "target_satisfied": abs(parameter_difference) / a1_parameters * 100.0 <= 1.0,
        "strict_single_variable_ablation": False,
    }
    matrix_lock_path = write_json_atomic(
        output_root / "matrix_lock.json",
        {
            "schema_version": "wireless-v2-w5-matrix-lock-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_git_commit": implementation_commit,
            "protocol_version": V2_PROTOCOL_VERSION,
            "hypothesis_file": HYPOTHESIS_RELATIVE_PATH.as_posix(),
            "hypothesis_sha256": hypothesis_sha256,
            "dataset_sha256": V2_DATASET_SHA256,
            "split_manifest_sha256": V2_SPLIT_MANIFEST_SHA256,
            "configuration": W5_CONFIGURATION,
            "run_seeds": list(V2_RUN_SEEDS),
            "planned_training_run_count": len(V2_RUN_SEEDS),
            "architecture_profile": profile,
            "capacity_control": capacity,
            "scope": "fixed_validation_only",
            "test_set_used": False,
        },
    )
    dataset = RadioMLTorchDataset(load_restricted_radioml_pickle(args.pickle_file))
    initialization_records = _load_initialization_records(repository_root)
    records = []
    device = torch.device("cuda")
    for run_seed in V2_RUN_SEEDS:
        print(f"starting configuration={W5_CONFIGURATION} run_seed={run_seed}", flush=True)
        records.append(
            run_one_experiment(
                run_seed=run_seed,
                dataset=dataset,
                dataset_sha256=V2_DATASET_SHA256,
                split_manifest=args.split_manifest,
                output_root=output_root,
                repository_root=repository_root,
                implementation_commit=implementation_commit,
                device=device,
                initialization_record=initialization_records[run_seed],
            )
        )
    by_seed = {int(record["run_seed"]): record for record in records}
    comparators = _comparator_records(repository_root)
    analysis = json.loads(
        (repository_root / "experiments/v2/analysis/analysis_summary.json").read_text(
            encoding="utf-8"
        )
    )
    overall_paired = {
        name: _paired_summary(by_seed, values, comparator_name=name)
        for name, values in comparators.items()
    }
    segment_paired = {
        "A2-T": _segment_paired_summary(
            by_seed,
            analysis,
            comparator_name="A1/A2-T",
        ),
        "A2-G": _segment_paired_summary(
            by_seed,
            analysis,
            comparator_name="A2-G",
        ),
    }
    reference_model = create_w3_model(
        W5_CONFIGURATION,
        num_classes=11,
        initial_backbone=torch.load(
            repository_root
            / initialization_records[V2_RUN_SEEDS[0]]["a2_shared_backbone_file"],
            map_location="cpu",
            weights_only=True,
        ),
    )
    first_checkpoint = torch.load(
        output_root
        / W5_CONFIGURATION
        / f"run_seed_{V2_RUN_SEEDS[0]}"
        / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    reference_model.load_state_dict(first_checkpoint["model_state_dict"], strict=True)
    summary = {
        "schema_version": "wireless-v2-w5-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_git_commit": implementation_commit,
        "protocol_version": V2_PROTOCOL_VERSION,
        "hypothesis_file": HYPOTHESIS_RELATIVE_PATH.as_posix(),
        "hypothesis_sha256": hypothesis_sha256,
        "matrix_lock_file": _relative(matrix_lock_path, repository_root),
        "dataset_sha256": V2_DATASET_SHA256,
        "split_manifest_sha256": V2_SPLIT_MANIFEST_SHA256,
        "scope": "fixed_validation_only",
        "test_set_used": False,
        "run_seeds": list(V2_RUN_SEEDS),
        "raw_run_count": len(records),
        "architecture_profile": profile,
        "capacity_control": capacity,
        "validation_accuracy": descriptive_summary(
            [record["validation"]["accuracy"] for record in records]
        ),
        "validation_macro_f1": descriptive_summary(
            [record["validation"]["macro_f1"] for record in records]
        ),
        "per_snr": _per_snr_summary(by_seed),
        "best_epoch": descriptive_summary(
            [record["training"]["best_epoch"] for record in records]
        ),
        "per_seed": [
            {
                "run_seed": record["run_seed"],
                "validation_accuracy": record["validation"]["accuracy"],
                "validation_macro_f1": record["validation"]["macro_f1"],
                "best_epoch": record["training"]["best_epoch"],
                "result_file": record["result_file"],
                "checkpoint_file": record["checkpoint_file"],
            }
            for record in records
        ],
        "overall_paired_delta": overall_paired,
        "snr_segment_paired_delta": segment_paired,
        "hypothesis_evaluation": evaluate_preregistered_hypothesis(segment_paired),
        "latency_reference_seed": V2_RUN_SEEDS[0],
        "latency_benchmark": benchmark_inference_latency(
            reference_model,
            device=device,
        ),
        "interpretation_limit": (
            "Five seeds are descriptive only. A2-L is a shared-initial-backbone, "
            "near-parameter-budget comparison rather than a strict single-variable ablation."
        ),
    }
    summary_path = write_json_atomic(output_root / "w5_summary.json", summary)
    print(f"w5_summary={summary_path}")
    print("test_set_used=false")


if __name__ == "__main__":
    main()
