"""Generate four-arm error analysis from committed checkpoints on validation only."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_v2_paired_experiments import create_w2_model
from signal_modulation.data_integrity import sha256_file
from signal_modulation.evaluation import (
    calculate_class_snr_accuracy,
    calculate_classification_metrics,
    calculate_snr_metrics,
    calculate_snr_segment_metrics,
)
from signal_modulation.experiment import prepare_new_run_directory, write_json_atomic
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import RadioMLTorchDataset
from signal_modulation.reporting import (
    render_class_snr_grid_svg,
    render_confusion_grid_svg,
    render_multi_snr_accuracy_svg,
)
from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_RUN_SEEDS,
    V2ExperimentConfig,
    configure_v2_run_randomness,
    create_v2_development_loaders,
)
from signal_modulation.w3_ablation import create_w3_model


ARM_DIRECTORIES = {
    "A0": Path("experiments/v2/w2/A0"),
    "A1/A2-T": Path("experiments/v2/w2/A1"),
    "A2-G": Path("experiments/v2/w3/A2-G"),
    "A3": Path("experiments/v2/w3/A3"),
}
SNR_SEGMENTS = {"low_snr_le_-10": (None, -10.0), "high_snr_ge_10": (10.0, None)}


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=repository_root / "data/raw/RML2016.10a/RML2016.10a_dict.pkl",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository_root / "experiments/v2/analysis",
    )
    return parser.parse_args()


def _git_commit(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _collect_predictions(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    labels: list[Tensor] = []
    predictions: list[Tensor] = []
    snrs: list[Tensor] = []
    with torch.inference_mode():
        for batch in data_loader:
            logits = model(batch["signal"].to(device))
            labels.append(batch["label"].detach().cpu())
            predictions.append(logits.argmax(dim=1).detach().cpu())
            snrs.append(batch["snr"].detach().cpu())
    return torch.cat(labels), torch.cat(predictions), torch.cat(snrs)


def _load_model(
    arm: str,
    *,
    run_seed: int,
    repository_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    if arm == "A0":
        model = create_w2_model("A0", num_classes=11)
    elif arm == "A1/A2-T":
        model = create_w2_model("A1", num_classes=11)
    elif arm == "A2-G":
        initial_backbone = torch.load(
            repository_root
            / "experiments/v2/w3/initial_backbones"
            / f"run_seed_{run_seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = create_w3_model(
            "A2-G",
            num_classes=11,
            initial_backbone=initial_backbone,
        )
    else:
        model = create_w3_model("A3", num_classes=11)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def _matrix_sum(matrices: list[list[list[int]]]) -> list[list[int]]:
    return [
        [sum(matrix[row][column] for matrix in matrices) for column in range(len(matrices[0]))]
        for row in range(len(matrices[0]))
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def _aggregate_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    snrs = [row["snr"] for row in records[0]["by_snr"]]
    by_snr = []
    for column, snr in enumerate(snrs):
        accuracies = [record["by_snr"][column]["accuracy"] for record in records]
        macro_f1s = [record["by_snr"][column]["macro_f1"] for record in records]
        accuracy_mean, accuracy_std = _mean_std(accuracies)
        f1_mean, f1_std = _mean_std(macro_f1s)
        by_snr.append(
            {
                "snr": snr,
                "accuracy_mean": accuracy_mean,
                "accuracy_sample_std": accuracy_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_sample_std": f1_std,
                "n": len(records),
            }
        )

    class_count = len(records[0]["class_snr_accuracy"])
    snr_count = len(snrs)
    class_snr_mean: list[list[float]] = []
    class_snr_std: list[list[float]] = []
    for class_index in range(class_count):
        mean_row = []
        std_row = []
        for snr_index in range(snr_count):
            values = [
                record["class_snr_accuracy"][class_index][snr_index]
                for record in records
            ]
            mean, std = _mean_std(values)
            mean_row.append(mean)
            std_row.append(std)
        class_snr_mean.append(mean_row)
        class_snr_std.append(std_row)

    return {
        "n_runs": len(records),
        "by_snr": by_snr,
        "class_snr_accuracy_mean": class_snr_mean,
        "class_snr_accuracy_sample_std": class_snr_std,
        "low_snr_confusion_matrix_sum": _matrix_sum(
            [record["segments"]["low_snr_le_-10"]["confusion_matrix"] for record in records]
        ),
        "high_snr_confusion_matrix_sum": _matrix_sum(
            [record["segments"]["high_snr_ge_10"]["confusion_matrix"] for record in records]
        ),
        "runs": records,
    }


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if not torch.cuda.is_available():
        raise RuntimeError("validation analysis requires the verified CUDA environment")
    if sha256_file(args.data_file) != V2_DATASET_SHA256:
        raise ValueError("dataset SHA-256 does not match the frozen V2 identity")
    output = prepare_new_run_directory(args.output_directory)
    grouped = load_restricted_radioml_pickle(args.data_file)
    dataset = RadioMLTorchDataset(grouped)
    config = V2ExperimentConfig(run_seed=V2_RUN_SEEDS[0])
    configure_v2_run_randomness(config)
    loaders = create_v2_development_loaders(
        dataset,
        repository_root / "manifests/v2/split_manifest.json",
        config,
        dataset_sha256=V2_DATASET_SHA256,
        pin_memory=True,
    )
    device = torch.device("cuda")
    arms: dict[str, Any] = {}
    for arm, relative_directory in ARM_DIRECTORIES.items():
        records = []
        for run_seed in V2_RUN_SEEDS:
            run_directory = repository_root / relative_directory / f"run_seed_{run_seed}"
            result_path = run_directory / "validation_result.json"
            checkpoint_path = run_directory / "best_checkpoint.pt"
            historical = json.loads(result_path.read_text(encoding="utf-8"))
            if sha256_file(checkpoint_path) != historical["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
            model = _load_model(
                arm,
                run_seed=run_seed,
                repository_root=repository_root,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            labels, predictions, snrs = _collect_predictions(
                model,
                loaders.validation,
                device=device,
            )
            overall = calculate_classification_metrics(labels, predictions, num_classes=11)
            by_snr = calculate_snr_metrics(labels, predictions, snrs, num_classes=11)
            if (
                abs(overall.accuracy - historical["validation"]["accuracy"]) > 1e-12
                or abs(overall.macro_f1 - historical["validation"]["macro_f1"]) > 1e-12
            ):
                raise RuntimeError(f"validation inference does not match history: {arm}/{run_seed}")
            class_snr = calculate_class_snr_accuracy(
                labels,
                predictions,
                snrs,
                num_classes=11,
            )
            segments = calculate_snr_segment_metrics(
                labels,
                predictions,
                snrs,
                num_classes=11,
                segments=SNR_SEGMENTS,
            )
            records.append(
                {
                    "run_seed": run_seed,
                    "source_result_file": relative_directory.as_posix()
                    + f"/run_seed_{run_seed}/validation_result.json",
                    "source_result_sha256": sha256_file(result_path),
                    "checkpoint_file": relative_directory.as_posix()
                    + f"/run_seed_{run_seed}/best_checkpoint.pt",
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "accuracy": overall.accuracy,
                    "macro_f1": overall.macro_f1,
                    "by_snr": [asdict(item) for item in by_snr],
                    "class_snr_accuracy": [list(row) for row in class_snr.accuracy],
                    "class_snr_sample_counts": [list(row) for row in class_snr.sample_counts],
                    "segments": {
                        name: {
                            "accuracy": metrics.accuracy,
                            "macro_f1": metrics.macro_f1,
                            "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
                        }
                        for name, metrics in segments.items()
                    },
                }
            )
            print(f"validated arm={arm} run_seed={run_seed}", flush=True)
        arms[arm] = _aggregate_arm(records)

    payload = {
        "schema_version": "wireless-v2-validation-error-analysis-v1",
        "scope": "fixed_validation_only",
        "test_set_used": False,
        "implementation_git_commit": _git_commit(repository_root),
        "dataset_sha256": V2_DATASET_SHA256,
        "split_manifest": "manifests/v2/split_manifest.json",
        "run_seeds": list(V2_RUN_SEEDS),
        "snrs": list(arms["A0"]["runs"][0]["by_snr"][index]["snr"] for index in range(20)),
        "modulations": list(dataset.modulations),
        "aggregation": "five pre-registered seeds; confusion counts summed and accuracy cells summarized with mean/sample std",
        "arms": arms,
    }
    summary_path = write_json_atomic(output / "analysis_summary.json", payload)
    render_multi_snr_accuracy_svg(
        {name: arm["by_snr"] for name, arm in arms.items()},
        output / "accuracy_vs_snr.svg",
    )
    render_confusion_grid_svg(
        {name: arm["high_snr_confusion_matrix_sum"] for name, arm in arms.items()},
        dataset.modulations,
        output / "confusion_high_snr.svg",
        title="Validation Confusion Matrices — High SNR (≥ 10 dB), 5-seed aggregate",
    )
    render_confusion_grid_svg(
        {name: arm["low_snr_confusion_matrix_sum"] for name, arm in arms.items()},
        dataset.modulations,
        output / "confusion_low_snr.svg",
        title="Validation Confusion Matrices — Low SNR (≤ −10 dB), 5-seed aggregate",
    )
    render_class_snr_grid_svg(
        {name: arm["class_snr_accuracy_mean"] for name, arm in arms.items()},
        dataset.modulations,
        payload["snrs"],
        output / "class_snr_accuracy.svg",
    )
    print(f"analysis_summary={summary_path}")
    print("test_set_used=false")


if __name__ == "__main__":
    main()
