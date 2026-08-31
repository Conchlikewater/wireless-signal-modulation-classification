"""Descriptive summaries for the pre-registered Wireless V2 W2 matrix."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


W2_CONFIGURATIONS = ("A0", "A1")


def descriptive_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    """Return n, arithmetic mean, and sample standard deviation (n-1)."""

    numeric = [float(value) for value in values]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise ValueError("descriptive values must be non-empty and finite")
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "sample_std": statistics.stdev(numeric) if len(numeric) >= 2 else None,
    }


def _completed_records_by_configuration(
    records: Sequence[Mapping[str, Any]],
    run_seeds: Sequence[int],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    expected_seeds = tuple(run_seeds)
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("run_seeds must be a non-empty unique sequence")
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {
        name: {} for name in W2_CONFIGURATIONS
    }
    for record in records:
        if record.get("status") != "completed":
            raise ValueError("W2 summary requires every formal run to be completed")
        configuration = record.get("configuration")
        run_seed = record.get("run_seed")
        if configuration not in grouped or run_seed not in expected_seeds:
            raise ValueError("result contains an unregistered configuration or run seed")
        if run_seed in grouped[configuration]:
            raise ValueError("result contains a duplicate configuration/run_seed pair")
        grouped[configuration][run_seed] = record

    for configuration, by_seed in grouped.items():
        if set(by_seed) != set(expected_seeds):
            raise ValueError(f"{configuration} does not contain every pre-registered seed")
    return grouped


def _validation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = record.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("result validation must be an object")
    return validation


def _training(record: Mapping[str, Any]) -> Mapping[str, Any]:
    training = record.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("result training must be an object")
    return training


def _finite_metric(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"metric {name} must be finite")
    return float(value)


def _snr_map(record: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    by_snr = _validation(record).get("by_snr")
    if not isinstance(by_snr, list) or not by_snr:
        raise ValueError("validation.by_snr must be a non-empty list")
    mapped: dict[float, Mapping[str, Any]] = {}
    for item in by_snr:
        if not isinstance(item, Mapping):
            raise ValueError("each validation.by_snr item must be an object")
        snr = _finite_metric(item, "snr")
        if snr in mapped:
            raise ValueError("validation.by_snr contains a duplicate SNR")
        mapped[snr] = item
    return mapped


def _model_summary(
    by_seed: Mapping[int, Mapping[str, Any]],
    run_seeds: Sequence[int],
) -> dict[str, Any]:
    ordered = [by_seed[seed] for seed in run_seeds]
    parameters = {
        int(record["model"]["trainable_parameters"]) for record in ordered
    }
    if len(parameters) != 1:
        raise ValueError("trainable parameter count changed across run seeds")

    per_seed = []
    for run_seed, record in zip(run_seeds, ordered, strict=True):
        validation = _validation(record)
        training = _training(record)
        per_seed.append(
            {
                "run_seed": run_seed,
                "validation_accuracy": _finite_metric(validation, "accuracy"),
                "validation_macro_f1": _finite_metric(validation, "macro_f1"),
                "best_epoch": int(training["best_epoch"]),
                "result_file": record.get("result_file"),
                "checkpoint_file": record.get("checkpoint_file"),
            }
        )

    snr_maps = [_snr_map(record) for record in ordered]
    snr_values = tuple(sorted(snr_maps[0]))
    if any(tuple(sorted(values)) != snr_values for values in snr_maps[1:]):
        raise ValueError("SNR levels differ across formal runs")
    per_snr = []
    for snr in snr_values:
        sample_counts = {int(values[snr]["sample_count"]) for values in snr_maps}
        if len(sample_counts) != 1:
            raise ValueError("per-SNR sample count changed across run seeds")
        per_snr.append(
            {
                "snr": snr,
                "sample_count": sample_counts.pop(),
                "accuracy": descriptive_summary(
                    [_finite_metric(values[snr], "accuracy") for values in snr_maps]
                ),
                "macro_f1": descriptive_summary(
                    [_finite_metric(values[snr], "macro_f1") for values in snr_maps]
                ),
            }
        )

    return {
        "n_runs": len(ordered),
        "trainable_parameters": parameters.pop(),
        "per_seed": per_seed,
        "validation_accuracy": descriptive_summary(
            [item["validation_accuracy"] for item in per_seed]
        ),
        "validation_macro_f1": descriptive_summary(
            [item["validation_macro_f1"] for item in per_seed]
        ),
        "best_epoch": descriptive_summary([item["best_epoch"] for item in per_seed]),
        "per_snr": per_snr,
    }


def _paired_summary(
    grouped: Mapping[str, Mapping[int, Mapping[str, Any]]],
    run_seeds: Sequence[int],
) -> dict[str, Any]:
    per_seed = []
    per_snr = []
    for run_seed in run_seeds:
        a0_validation = _validation(grouped["A0"][run_seed])
        a1_validation = _validation(grouped["A1"][run_seed])
        per_seed.append(
            {
                "run_seed": run_seed,
                "validation_accuracy": _finite_metric(
                    a1_validation, "accuracy"
                )
                - _finite_metric(a0_validation, "accuracy"),
                "validation_macro_f1": _finite_metric(a1_validation, "macro_f1")
                - _finite_metric(a0_validation, "macro_f1"),
                "best_epoch": int(_training(grouped["A1"][run_seed])["best_epoch"])
                - int(_training(grouped["A0"][run_seed])["best_epoch"]),
            }
        )

    a0_snr_maps = [_snr_map(grouped["A0"][seed]) for seed in run_seeds]
    a1_snr_maps = [_snr_map(grouped["A1"][seed]) for seed in run_seeds]
    snr_values = tuple(sorted(a0_snr_maps[0]))
    if any(tuple(sorted(values)) != snr_values for values in a0_snr_maps + a1_snr_maps):
        raise ValueError("paired runs do not share identical SNR levels")
    for snr in snr_values:
        accuracy_deltas = [
            _finite_metric(a1_values[snr], "accuracy")
            - _finite_metric(a0_values[snr], "accuracy")
            for a0_values, a1_values in zip(a0_snr_maps, a1_snr_maps, strict=True)
        ]
        macro_f1_deltas = [
            _finite_metric(a1_values[snr], "macro_f1")
            - _finite_metric(a0_values[snr], "macro_f1")
            for a0_values, a1_values in zip(a0_snr_maps, a1_snr_maps, strict=True)
        ]
        per_snr.append(
            {
                "snr": snr,
                "accuracy": descriptive_summary(accuracy_deltas),
                "macro_f1": descriptive_summary(macro_f1_deltas),
            }
        )

    return {
        "direction": "A1_minus_A0",
        "n_pairs": len(per_seed),
        "per_seed": per_seed,
        "validation_accuracy": descriptive_summary(
            [item["validation_accuracy"] for item in per_seed]
        ),
        "validation_macro_f1": descriptive_summary(
            [item["validation_macro_f1"] for item in per_seed]
        ),
        "best_epoch": descriptive_summary([item["best_epoch"] for item in per_seed]),
        "per_snr": per_snr,
    }


def summarize_w2_results(
    records: Sequence[Mapping[str, Any]],
    *,
    run_seeds: Sequence[int],
) -> dict[str, Any]:
    """Summarize all A0/A1 runs without selecting a best seed."""

    grouped = _completed_records_by_configuration(records, run_seeds)
    return {
        "schema_version": "wireless-v2-w2-summary-v1",
        "stage": "W2",
        "scope": "fixed_validation_only",
        "test_set_used": False,
        "run_seeds": list(run_seeds),
        "raw_run_count": len(records),
        "models": {
            configuration: _model_summary(grouped[configuration], run_seeds)
            for configuration in W2_CONFIGURATIONS
        },
        "paired_delta": _paired_summary(grouped, run_seeds),
        "interpretation_limit": (
            "Five seeds support descriptive mean/sample-std and paired deltas only; "
            "they do not establish statistical significance."
        ),
    }
