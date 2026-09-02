"""Resolve recorded training fields into the effective model configuration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from signal_modulation.v2_experiment import V2ExperimentConfig


SHARED_TRAINING_FIELDS = (
    "epochs",
    "evaluation_batch_size",
    "learning_rate",
    "min_delta",
    "num_workers",
    "patience",
    "run_seed",
    "split_seed",
    "train_batch_size",
    "use_amp",
    "use_scheduler",
)


def validate_shared_training_config(
    recorded_config: Mapping[str, Any],
    *,
    run_seed: int,
) -> None:
    """Reject a historical record that differs from the executed shared config."""

    expected = asdict(V2ExperimentConfig(run_seed=run_seed))
    for field in SHARED_TRAINING_FIELDS:
        if recorded_config.get(field) != expected[field]:
            raise ValueError(
                f"recorded {field} does not match the frozen execution config"
            )


def resolve_effective_dropout(
    configuration: str,
    *,
    run_id: str,
    recorded_config: Mapping[str, Any],
    erratum: Mapping[str, Any] | None = None,
) -> float | None:
    """Return the Dropout probability that the model factory actually used.

    ``None`` means that the arm has no ``nn.Dropout`` module. A3 is the only
    historical arm whose generic recorded config needs an immutable erratum.
    """

    recorded_value = recorded_config.get("dropout")
    if configuration == "A0":
        return None
    if configuration in {"A1", "A2-G", "A2-L"}:
        return float(recorded_value)
    if configuration != "A3":
        raise ValueError(f"unsupported configuration semantics: {configuration}")
    if erratum is None:
        raise ValueError("A3 effective Dropout requires its registered erratum")
    if erratum.get("field") != "config.dropout":
        raise ValueError("A3 erratum targets the wrong field")
    if run_id not in erratum.get("affected_runs", []):
        raise ValueError("A3 run is not covered by the registered erratum")
    if erratum.get("recorded_value") != recorded_value:
        raise ValueError("A3 erratum recorded value does not match the manifest")
    effective_value = erratum.get("effective_value")
    if not isinstance(effective_value, (int, float)):
        raise ValueError("A3 erratum effective value must be numeric")
    return float(effective_value)


def effective_model_config(
    configuration: str,
    *,
    run_id: str,
    recorded_config: Mapping[str, Any],
    erratum: Mapping[str, Any] | None = None,
) -> dict[str, float | None]:
    """Build explicit model semantics for new manifests and audit reports."""

    return {
        "dropout": resolve_effective_dropout(
            configuration,
            run_id=run_id,
            recorded_config=recorded_config,
            erratum=erratum,
        )
    }
