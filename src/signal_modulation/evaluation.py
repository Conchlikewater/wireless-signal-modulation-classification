"""Dependency-light classification metrics and model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Overall and per-class metrics derived from a confusion matrix."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class SNRMetrics:
    """Accuracy, Macro F1, and sample count for one SNR level."""

    snr: float
    accuracy: float
    macro_f1: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Loss and classification metrics over a complete data-loader pass."""

    loss: float
    sample_count: int
    classification: ClassificationMetrics
    by_snr: tuple[SNRMetrics, ...]


@dataclass(frozen=True, slots=True)
class ClassSNRAccuracy:
    """Per-class accuracy rows over ordered SNR columns."""

    snrs: tuple[float, ...]
    accuracy: tuple[tuple[float, ...], ...]
    sample_counts: tuple[tuple[int, ...], ...]


def _as_cpu_long(values: Tensor, name: str) -> Tensor:
    tensor = torch.as_tensor(values).detach().cpu()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return tensor.to(dtype=torch.int64)


def calculate_classification_metrics(
    labels: Tensor,
    predictions: Tensor,
    *,
    num_classes: int,
) -> ClassificationMetrics:
    """Calculate accuracy, macro metrics, and a row=true/column=predicted matrix."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    label_values = _as_cpu_long(labels, "labels")
    prediction_values = _as_cpu_long(predictions, "predictions")
    if label_values.numel() == 0 or label_values.shape != prediction_values.shape:
        raise ValueError("labels and predictions must have the same non-zero shape")
    if (
        torch.any(label_values < 0)
        or torch.any(label_values >= num_classes)
        or torch.any(prediction_values < 0)
        or torch.any(prediction_values >= num_classes)
    ):
        raise ValueError("labels and predictions must be valid class indices")

    flat_indices = label_values * num_classes + prediction_values
    matrix = torch.bincount(
        flat_indices, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    true_positive = matrix.diag().to(dtype=torch.float64)
    predicted_count = matrix.sum(dim=0).to(dtype=torch.float64)
    actual_count = matrix.sum(dim=1).to(dtype=torch.float64)

    precision = torch.where(
        predicted_count > 0,
        true_positive / predicted_count,
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        actual_count > 0,
        true_positive / actual_count,
        torch.zeros_like(true_positive),
    )
    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2.0 * precision * recall / f1_denominator,
        torch.zeros_like(precision),
    )

    return ClassificationMetrics(
        accuracy=float(true_positive.sum() / label_values.numel()),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        per_class_precision=tuple(float(value) for value in precision),
        per_class_recall=tuple(float(value) for value in recall),
        per_class_f1=tuple(float(value) for value in f1),
        confusion_matrix=tuple(
            tuple(int(value) for value in row) for row in matrix.tolist()
        ),
    )


def calculate_snr_metrics(
    labels: Tensor,
    predictions: Tensor,
    snrs: Tensor,
    *,
    num_classes: int | None = None,
) -> tuple[SNRMetrics, ...]:
    """Group prediction Accuracy and Macro F1 by each observed SNR value."""

    label_values = _as_cpu_long(labels, "labels")
    prediction_values = _as_cpu_long(predictions, "predictions")
    snr_values = torch.as_tensor(snrs).detach().cpu().to(dtype=torch.float32)
    if snr_values.ndim != 1:
        raise ValueError("snrs must be one-dimensional")
    if (
        label_values.numel() == 0
        or label_values.shape != prediction_values.shape
        or label_values.shape != snr_values.shape
    ):
        raise ValueError("labels, predictions, and snrs must have the same non-zero shape")
    if not torch.isfinite(snr_values).all():
        raise ValueError("snrs must contain only finite values")
    if num_classes is None:
        num_classes = int(torch.maximum(label_values.max(), prediction_values.max())) + 1

    results: list[SNRMetrics] = []
    for snr in torch.unique(snr_values, sorted=True):
        mask = snr_values == snr
        sample_count = int(mask.sum())
        classification = calculate_classification_metrics(
            label_values[mask],
            prediction_values[mask],
            num_classes=num_classes,
        )
        results.append(
            SNRMetrics(
                snr=float(snr),
                accuracy=classification.accuracy,
                macro_f1=classification.macro_f1,
                sample_count=sample_count,
            )
        )
    return tuple(results)


def calculate_class_snr_accuracy(
    labels: Tensor,
    predictions: Tensor,
    snrs: Tensor,
    *,
    num_classes: int,
    snr_values: Sequence[float] | None = None,
) -> ClassSNRAccuracy:
    """Return a ``(num_classes, num_snrs)`` conditional-accuracy matrix."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    label_values = _as_cpu_long(labels, "labels")
    prediction_values = _as_cpu_long(predictions, "predictions")
    observed_snrs = torch.as_tensor(snrs).detach().cpu().to(dtype=torch.float32)
    if observed_snrs.ndim != 1:
        raise ValueError("snrs must be one-dimensional")
    if (
        label_values.numel() == 0
        or label_values.shape != prediction_values.shape
        or label_values.shape != observed_snrs.shape
    ):
        raise ValueError("labels, predictions, and snrs must have the same non-zero shape")
    if not torch.isfinite(observed_snrs).all():
        raise ValueError("snrs must contain only finite values")
    if (
        torch.any(label_values < 0)
        or torch.any(label_values >= num_classes)
        or torch.any(prediction_values < 0)
        or torch.any(prediction_values >= num_classes)
    ):
        raise ValueError("labels and predictions must be valid class indices")

    if snr_values is None:
        ordered_snrs = tuple(float(value) for value in torch.unique(observed_snrs, sorted=True))
    else:
        ordered_snrs = tuple(float(value) for value in snr_values)
        if (
            not ordered_snrs
            or len(set(ordered_snrs)) != len(ordered_snrs)
            or any(not torch.isfinite(torch.tensor(value)) for value in ordered_snrs)
        ):
            raise ValueError("snr_values must contain unique finite values")

    accuracy_rows: list[tuple[float, ...]] = []
    count_rows: list[tuple[int, ...]] = []
    for class_index in range(num_classes):
        class_accuracies: list[float] = []
        class_counts: list[int] = []
        for snr in ordered_snrs:
            mask = (label_values == class_index) & (observed_snrs == snr)
            sample_count = int(mask.sum())
            class_counts.append(sample_count)
            class_accuracies.append(
                float((prediction_values[mask] == class_index).to(torch.float64).mean())
                if sample_count
                else float("nan")
            )
        accuracy_rows.append(tuple(class_accuracies))
        count_rows.append(tuple(class_counts))
    return ClassSNRAccuracy(
        snrs=ordered_snrs,
        accuracy=tuple(accuracy_rows),
        sample_counts=tuple(count_rows),
    )


def calculate_snr_segment_metrics(
    labels: Tensor,
    predictions: Tensor,
    snrs: Tensor,
    *,
    num_classes: int,
    segments: Mapping[str, tuple[float | None, float | None]],
) -> dict[str, ClassificationMetrics]:
    """Calculate confusion-derived metrics for inclusive SNR ranges."""

    label_values = _as_cpu_long(labels, "labels")
    prediction_values = _as_cpu_long(predictions, "predictions")
    snr_values = torch.as_tensor(snrs).detach().cpu().to(dtype=torch.float32)
    if snr_values.ndim != 1:
        raise ValueError("snrs must be one-dimensional")
    if (
        label_values.numel() == 0
        or label_values.shape != prediction_values.shape
        or label_values.shape != snr_values.shape
    ):
        raise ValueError("labels, predictions, and snrs must have the same non-zero shape")
    if not torch.isfinite(snr_values).all():
        raise ValueError("snrs must contain only finite values")
    if not segments:
        raise ValueError("segments must not be empty")

    results: dict[str, ClassificationMetrics] = {}
    for name, (minimum, maximum) in segments.items():
        if not name:
            raise ValueError("segment names must be non-empty")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("segment minimum must not exceed maximum")
        mask = torch.ones_like(snr_values, dtype=torch.bool)
        if minimum is not None:
            mask &= snr_values >= minimum
        if maximum is not None:
            mask &= snr_values <= maximum
        if not bool(mask.any()):
            raise ValueError(f"SNR segment contains no samples: {name}")
        results[name] = calculate_classification_metrics(
            label_values[mask],
            prediction_values[mask],
            num_classes=num_classes,
        )
    return results


def evaluate_classifier(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
) -> ModelEvaluation:
    """Run one no-gradient pass and retain labels, predictions, and SNR metadata."""

    loss_function = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    sample_count = 0
    labels: list[Tensor] = []
    predictions: list[Tensor] = []
    snrs: list[Tensor] = []

    with torch.inference_mode():
        for batch in data_loader:
            try:
                signals = batch["signal"].to(device)
                batch_labels = batch["label"].to(device)
                batch_snrs = batch["snr"]
            except KeyError as error:
                raise ValueError("each evaluation batch requires signal, label, and snr") from error
            logits = model(signals)
            if logits.ndim != 2 or logits.shape != (batch_labels.shape[0], num_classes):
                raise ValueError("model logits must have shape (batch, num_classes)")
            loss = loss_function(logits, batch_labels)
            batch_size = int(batch_labels.shape[0])
            total_loss += float(loss) * batch_size
            sample_count += batch_size
            labels.append(batch_labels.cpu())
            predictions.append(logits.argmax(dim=1).cpu())
            snrs.append(batch_snrs.detach().cpu())

    if sample_count == 0:
        raise ValueError("data loader produced no evaluation samples")
    all_labels = torch.cat(labels)
    all_predictions = torch.cat(predictions)
    all_snrs = torch.cat(snrs)
    return ModelEvaluation(
        loss=total_loss / sample_count,
        sample_count=sample_count,
        classification=calculate_classification_metrics(
            all_labels, all_predictions, num_classes=num_classes
        ),
        by_snr=calculate_snr_metrics(
            all_labels,
            all_predictions,
            all_snrs,
            num_classes=num_classes,
        ),
    )
