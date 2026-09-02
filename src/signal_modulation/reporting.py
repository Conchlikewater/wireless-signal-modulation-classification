"""Dependency-free SVG reporting for saved modulation-classification results."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_result(path: str | Path) -> Mapping[str, Any]:
    """Read one UTF-8 JSON experiment result."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment result must be a JSON object")
    return payload


def _write_svg(path: str | Path, lines: Sequence[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _snr_series(
    result: Mapping[str, Any], *, section: str = "validation"
) -> list[tuple[float, float]]:
    try:
        rows = result[section]["by_snr"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"result is missing {section}.by_snr") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{section}.by_snr must be a non-empty list")

    series: list[tuple[float, float]] = []
    for row in rows:
        try:
            snr = float(row["snr"])
            accuracy = float(row["accuracy"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each SNR row requires numeric snr and accuracy") from error
        if not math.isfinite(snr) or not math.isfinite(accuracy):
            raise ValueError("SNR values and accuracies must be finite")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("SNR accuracy must be between zero and one")
        series.append((snr, accuracy))
    series.sort(key=lambda item: item[0])
    if len({snr for snr, _ in series}) != len(series):
        raise ValueError("SNR values must be unique")
    return series


def render_snr_comparison_svg(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Render comparable validation-accuracy curves against SNR."""

    baseline_series = _snr_series(baseline)
    candidate_series = _snr_series(candidate)
    if [item[0] for item in baseline_series] != [item[0] for item in candidate_series]:
        raise ValueError("baseline and candidate must contain the same SNR values")

    width, height = 960, 560
    left, top, right, bottom = 90, 55, 40, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    minimum_snr = baseline_series[0][0]
    maximum_snr = baseline_series[-1][0]

    def point(snr: float, accuracy: float) -> tuple[float, float]:
        x = left + (snr - minimum_snr) / (maximum_snr - minimum_snr) * plot_width
        y = top + (1.0 - accuracy) * plot_height
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#e5e7eb;stroke-width:1}.label{font-size:13px}.title{font-size:22px;font-weight:700}.legend{font-size:14px;font-weight:600}</style>',
        '<text x="480" y="30" text-anchor="middle" class="title">Validation Accuracy by SNR</text>',
    ]
    for index in range(6):
        accuracy = index / 5
        y = top + (1.0 - accuracy) * plot_height
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" class="label">{accuracy:.1f}</text>')
    for index, (snr, _) in enumerate(baseline_series):
        x, _ = point(snr, 0.0)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" class="grid"/>')
        if index % 2 == 0 or index == len(baseline_series) - 1:
            lines.append(f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" class="label">{snr:g}</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
            f'<text x="{left+plot_width/2:.2f}" y="{height-24}" text-anchor="middle" class="label">SNR (dB)</text>',
            f'<text x="24" y="{top+plot_height/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 24 {top+plot_height/2:.2f})">Accuracy</text>',
        ]
    )

    for name, series, color in (
        ("SimpleCNN", baseline_series, "#6b7280"),
        ("TemporalCNN", candidate_series, "#047857"),
    ):
        points = " ".join(f"{point(snr, accuracy)[0]:.2f},{point(snr, accuracy)[1]:.2f}" for snr, accuracy in series)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for snr, accuracy in series:
            x, y = point(snr, accuracy)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
        legend_x = 675 if name == "SimpleCNN" else 800
        lines.append(f'<line x1="{legend_x}" y1="45" x2="{legend_x+28}" y2="45" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+34}" y="50" class="legend">{name}</text>')
    lines.append("</svg>")
    return _write_svg(path, lines)


def render_snr_accuracy_svg(
    result: Mapping[str, Any],
    path: str | Path,
    *,
    section: str,
    title: str,
    series_label: str,
) -> Path:
    """Render one split's accuracy curve against SNR."""

    series = _snr_series(result, section=section)
    width, height = 960, 560
    left, top, right, bottom = 90, 55, 40, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    minimum_snr = series[0][0]
    maximum_snr = series[-1][0]

    def point(snr: float, accuracy: float) -> tuple[float, float]:
        x = left + (snr - minimum_snr) / (maximum_snr - minimum_snr) * plot_width
        y = top + (1.0 - accuracy) * plot_height
        return x, y

    escaped_title = html.escape(title)
    escaped_label = html.escape(series_label)
    color = "#047857"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#e5e7eb;stroke-width:1}.label{font-size:13px}.title{font-size:22px;font-weight:700}.legend{font-size:14px;font-weight:600}</style>',
        f'<text x="480" y="30" text-anchor="middle" class="title">{escaped_title}</text>',
    ]
    for index in range(6):
        accuracy = index / 5
        y = top + (1.0 - accuracy) * plot_height
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" class="label">{accuracy:.1f}</text>')
    for index, (snr, _) in enumerate(series):
        x, _ = point(snr, 0.0)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" class="grid"/>')
        if index % 2 == 0 or index == len(series) - 1:
            lines.append(f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" class="label">{snr:g}</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
            f'<text x="{left+plot_width/2:.2f}" y="{height-24}" text-anchor="middle" class="label">SNR (dB)</text>',
            f'<text x="24" y="{top+plot_height/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 24 {top+plot_height/2:.2f})">Accuracy</text>',
        ]
    )
    points = " ".join(
        f"{point(snr, accuracy)[0]:.2f},{point(snr, accuracy)[1]:.2f}"
        for snr, accuracy in series
    )
    lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    for snr, accuracy in series:
        x, y = point(snr, accuracy)
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
    lines.append(f'<line x1="730" y1="45" x2="758" y2="45" stroke="{color}" stroke-width="3"/>')
    lines.append(f'<text x="764" y="50" class="legend">{escaped_label}</text>')
    lines.append("</svg>")
    return _write_svg(path, lines)


def _confusion_data(
    result: Mapping[str, Any],
    *,
    section: str = "validation",
) -> tuple[list[str], list[list[int]]]:
    try:
        labels = list(result["dataset"]["modulations"])
        matrix = [list(row) for row in result[section]["confusion_matrix"]]
    except (KeyError, TypeError) as error:
        raise ValueError("result is missing modulation labels or confusion matrix") from error
    if not labels or len(matrix) != len(labels):
        raise ValueError("confusion matrix must match the modulation labels")
    converted: list[list[int]] = []
    for row in matrix:
        if len(row) != len(labels):
            raise ValueError("confusion matrix must be square")
        converted_row = [int(value) for value in row]
        if any(value < 0 for value in converted_row) or sum(converted_row) == 0:
            raise ValueError("confusion rows require non-negative counts and samples")
        converted.append(converted_row)
    return [str(label) for label in labels], converted


def render_confusion_matrix_svg(
    result: Mapping[str, Any],
    path: str | Path,
    *,
    section: str = "validation",
    title: str = "TemporalCNN Validation Confusion Matrix",
) -> Path:
    """Render a row-normalized confusion matrix for one candidate model."""

    labels, matrix = _confusion_data(result, section=section)
    cell = 46
    left, top = 190, 110
    width = left + cell * len(labels) + 70
    height = top + cell * len(labels) + 90
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:22px;font-weight:700}.label{font-size:12px}.value{font-size:10px;font-weight:600}</style>',
        f'<text x="{width/2:.2f}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<text x="{left+cell*len(labels)/2:.2f}" y="{height-20}" text-anchor="middle" class="label">Predicted modulation</text>',
        f'<text x="24" y="{top+cell*len(labels)/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 24 {top+cell*len(labels)/2:.2f})">True modulation</text>',
    ]
    for index, label in enumerate(labels):
        escaped = html.escape(label)
        center_x = left + index * cell + cell / 2
        center_y = top + index * cell + cell / 2
        lines.append(f'<text x="{left-12}" y="{center_y+4:.2f}" text-anchor="end" class="label">{escaped}</text>')
        label_y = top - 10
        lines.append(f'<text x="{center_x:.2f}" y="{label_y}" text-anchor="start" class="label" transform="rotate(-45 {center_x:.2f} {label_y})">{escaped}</text>')
    for row_index, row in enumerate(matrix):
        row_total = sum(row)
        for column_index, count in enumerate(row):
            ratio = count / row_total
            red = round(247 - 239 * ratio)
            green = round(251 - 182 * ratio)
            blue = round(255 - 107 * ratio)
            fill = f"rgb({red},{green},{blue})"
            text_fill = "#ffffff" if ratio >= 0.5 else "#111827"
            x = left + column_index * cell
            y = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            lines.append(f'<text x="{x+cell/2:.2f}" y="{y+cell/2+4:.2f}" text-anchor="middle" class="value" style="fill:{text_fill}">{ratio*100:.1f}</text>')
    lines.append("</svg>")
    return _write_svg(path, lines)


def render_multi_snr_accuracy_svg(
    series_by_name: Mapping[str, Sequence[Mapping[str, float]]],
    path: str | Path,
    *,
    title: str = "Five-seed Validation Accuracy by SNR",
) -> Path:
    """Render multiple mean SNR curves with one-sigma dashed envelopes."""

    if not series_by_name:
        raise ValueError("at least one SNR series is required")
    converted: dict[str, list[tuple[float, float, float]]] = {}
    expected_snrs: list[float] | None = None
    for name, rows in series_by_name.items():
        values = sorted(
            (
                float(row["snr"]),
                float(row["accuracy_mean"]),
                float(row["accuracy_sample_std"]),
            )
            for row in rows
        )
        if not values or any(
            not all(math.isfinite(value) for value in row) for row in values
        ):
            raise ValueError("SNR summary rows must contain finite values")
        if any(not 0.0 <= mean <= 1.0 or std < 0.0 for _, mean, std in values):
            raise ValueError("SNR accuracy mean/std values are invalid")
        snrs = [row[0] for row in values]
        if expected_snrs is None:
            expected_snrs = snrs
        elif snrs != expected_snrs:
            raise ValueError("all arms must use the same SNR values")
        converted[str(name)] = values

    assert expected_snrs is not None
    width, height = 1080, 620
    left, top, right, bottom = 90, 70, 45, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    minimum_snr, maximum_snr = expected_snrs[0], expected_snrs[-1]

    def point(snr: float, accuracy: float) -> tuple[float, float]:
        x = left + (snr - minimum_snr) / (maximum_snr - minimum_snr) * plot_width
        y = top + (1.0 - min(1.0, max(0.0, accuracy))) * plot_height
        return x, y

    colors = ("#6b7280", "#047857", "#2563eb", "#c2410c", "#7c3aed")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#e5e7eb;stroke-width:1}.label{font-size:13px}.title{font-size:22px;font-weight:700}.legend{font-size:13px;font-weight:600}</style>',
        f'<text x="{width/2:.2f}" y="32" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]
    for index in range(6):
        accuracy = index / 5
        y = top + (1.0 - accuracy) * plot_height
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" class="label">{accuracy:.1f}</text>')
    for index, snr in enumerate(expected_snrs):
        x, _ = point(snr, 0.0)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" class="grid"/>')
        if index % 2 == 0 or index == len(expected_snrs) - 1:
            lines.append(f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" class="label">{snr:g}</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
            f'<text x="{left+plot_width/2:.2f}" y="{height-24}" text-anchor="middle" class="label">SNR (dB)</text>',
            f'<text x="24" y="{top+plot_height/2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 24 {top+plot_height/2:.2f})">Validation Accuracy</text>',
        ]
    )
    for arm_index, (name, values) in enumerate(converted.items()):
        color = colors[arm_index % len(colors)]
        for offset in (-1.0, 1.0):
            envelope = " ".join(
                f"{point(snr, mean + offset * std)[0]:.2f},{point(snr, mean + offset * std)[1]:.2f}"
                for snr, mean, std in values
            )
            lines.append(f'<polyline points="{envelope}" fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="4 4" opacity="0.55"/>')
        points = " ".join(
            f"{point(snr, mean)[0]:.2f},{point(snr, mean)[1]:.2f}"
            for snr, mean, _ in values
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_x = 610 + (arm_index % 2) * 210
        legend_y = 48 + (arm_index // 2) * 18
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+34}" y="{legend_y+4}" class="legend">{html.escape(name)}</text>')
    lines.append("</svg>")
    return _write_svg(path, lines)


def render_confusion_grid_svg(
    matrices: Mapping[str, Sequence[Sequence[int]]],
    labels: Sequence[str],
    path: str | Path,
    *,
    title: str,
) -> Path:
    """Render row-normalized confusion matrices for several arms in one SVG."""

    if not matrices or not labels:
        raise ValueError("confusion grid requires matrices and labels")
    class_count = len(labels)
    columns = 2
    rows = math.ceil(len(matrices) / columns)
    cell = 31
    panel_width = 150 + class_count * cell + 35
    panel_height = 105 + class_count * cell + 35
    width = columns * panel_width
    height = 60 + rows * panel_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:22px;font-weight:700}.panel{font-size:17px;font-weight:700}.label{font-size:9px}.value{font-size:7px;font-weight:600}</style>',
        f'<text x="{width/2:.2f}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]
    for panel_index, (name, matrix) in enumerate(matrices.items()):
        converted = [[int(value) for value in row] for row in matrix]
        if len(converted) != class_count or any(len(row) != class_count for row in converted):
            raise ValueError("each confusion matrix must match labels")
        origin_x = (panel_index % columns) * panel_width + 135
        origin_y = 60 + (panel_index // columns) * panel_height + 80
        lines.append(f'<text x="{origin_x+class_count*cell/2:.2f}" y="{origin_y-54}" text-anchor="middle" class="panel">{html.escape(name)}</text>')
        for index, label in enumerate(labels):
            center_x = origin_x + index * cell + cell / 2
            center_y = origin_y + index * cell + cell / 2
            escaped = html.escape(str(label))
            lines.append(f'<text x="{origin_x-8}" y="{center_y+3:.2f}" text-anchor="end" class="label">{escaped}</text>')
            lines.append(f'<text x="{center_x:.2f}" y="{origin_y-7}" text-anchor="start" class="label" transform="rotate(-45 {center_x:.2f} {origin_y-7})">{escaped}</text>')
        for row_index, row in enumerate(converted):
            row_total = sum(row)
            if row_total <= 0:
                raise ValueError("confusion rows require samples")
            for column_index, count in enumerate(row):
                ratio = count / row_total
                shade = round(247 - 210 * ratio)
                fill = f"rgb({shade},{min(251, shade+12)},{255})"
                text_fill = "#ffffff" if ratio >= 0.55 else "#111827"
                x = origin_x + column_index * cell
                y = origin_y + row_index * cell
                lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
                lines.append(f'<text x="{x+cell/2:.2f}" y="{y+cell/2+2.5:.2f}" text-anchor="middle" class="value" style="fill:{text_fill}">{ratio*100:.0f}</text>')
    lines.append("</svg>")
    return _write_svg(path, lines)


def render_class_snr_grid_svg(
    matrices: Mapping[str, Sequence[Sequence[float]]],
    labels: Sequence[str],
    snrs: Sequence[float],
    path: str | Path,
    *,
    title: str = "Class × SNR Validation Accuracy",
) -> Path:
    """Render per-class/SNR accuracy heatmaps for several arms."""

    if not matrices or not labels or not snrs:
        raise ValueError("class-SNR grid requires matrices, labels, and SNRs")
    columns = 2
    rows = math.ceil(len(matrices) / columns)
    cell_x, cell_y = 25, 25
    panel_width = 145 + len(snrs) * cell_x + 35
    panel_height = 80 + len(labels) * cell_y + 45
    width = columns * panel_width
    height = 60 + rows * panel_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:22px;font-weight:700}.panel{font-size:17px;font-weight:700}.label{font-size:9px}</style>',
        f'<text x="{width/2:.2f}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]
    for panel_index, (name, matrix) in enumerate(matrices.items()):
        converted = [[float(value) for value in row] for row in matrix]
        if len(converted) != len(labels) or any(len(row) != len(snrs) for row in converted):
            raise ValueError("each class-SNR matrix must match labels and SNRs")
        if any(not 0.0 <= value <= 1.0 for row in converted for value in row):
            raise ValueError("class-SNR accuracy must be in [0, 1]")
        origin_x = (panel_index % columns) * panel_width + 125
        origin_y = 60 + (panel_index // columns) * panel_height + 55
        lines.append(f'<text x="{origin_x+len(snrs)*cell_x/2:.2f}" y="{origin_y-29}" text-anchor="middle" class="panel">{html.escape(name)}</text>')
        for index, label in enumerate(labels):
            center_y = origin_y + index * cell_y + cell_y / 2
            lines.append(f'<text x="{origin_x-8}" y="{center_y+3:.2f}" text-anchor="end" class="label">{html.escape(str(label))}</text>')
        for index, snr in enumerate(snrs):
            if index % 2 == 0 or index == len(snrs) - 1:
                center_x = origin_x + index * cell_x + cell_x / 2
                lines.append(f'<text x="{center_x:.2f}" y="{origin_y+len(labels)*cell_y+17}" text-anchor="middle" class="label">{float(snr):g}</text>')
        for row_index, row in enumerate(converted):
            for column_index, value in enumerate(row):
                red = round(247 - 220 * value)
                green = round(251 - 115 * value)
                blue = round(255 - 45 * value)
                x = origin_x + column_index * cell_x
                y = origin_y + row_index * cell_y
                lines.append(f'<rect x="{x}" y="{y}" width="{cell_x}" height="{cell_y}" fill="rgb({red},{green},{blue})" stroke="#ffffff"/>')
    lines.append("</svg>")
    return _write_svg(path, lines)
