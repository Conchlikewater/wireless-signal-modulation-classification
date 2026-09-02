"""Tests for dependency-free experiment visualizations."""

import json
import tempfile
import unittest
from pathlib import Path

from signal_modulation.reporting import (
    load_result,
    render_class_snr_grid_svg,
    render_confusion_grid_svg,
    render_confusion_matrix_svg,
    render_multi_snr_accuracy_svg,
    render_snr_accuracy_svg,
    render_snr_comparison_svg,
)


def _result(accuracies: tuple[float, float]) -> dict[str, object]:
    return {
        "dataset": {"modulations": ["BPSK", "QPSK"]},
        "validation": {
            "by_snr": [
                {"snr": -2, "accuracy": accuracies[0], "sample_count": 10},
                {"snr": 0, "accuracy": accuracies[1], "sample_count": 10},
            ],
            "confusion_matrix": [[8, 2], [1, 9]],
        },
    }


class ReportingTests(unittest.TestCase):
    def test_load_result_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_result(path)

    def test_snr_comparison_writes_svg_with_both_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snr.svg"

            render_snr_comparison_svg(_result((0.2, 0.4)), _result((0.3, 0.6)), path)

            svg = path.read_text(encoding="utf-8")
            self.assertIn("SimpleCNN", svg)
            self.assertIn("TemporalCNN", svg)
            self.assertIn("SNR (dB)", svg)

    def test_snr_comparison_requires_matching_snr_values(self) -> None:
        candidate = _result((0.3, 0.6))
        candidate["validation"]["by_snr"][1]["snr"] = 2  # type: ignore[index]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "same SNR"):
                render_snr_comparison_svg(
                    _result((0.2, 0.4)),
                    candidate,
                    Path(directory) / "snr.svg",
                )

    def test_single_snr_curve_supports_test_section(self) -> None:
        result = _result((0.3, 0.6))
        result["test"] = result.pop("validation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-snr.svg"

            render_snr_accuracy_svg(
                result,
                path,
                section="test",
                title="Final test",
                series_label="Candidate",
            )

            svg = path.read_text(encoding="utf-8")
            self.assertIn("Final test", svg)
            self.assertIn("Candidate", svg)

    def test_confusion_matrix_is_row_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confusion.svg"

            render_confusion_matrix_svg(_result((0.2, 0.4)), path)

            svg = path.read_text(encoding="utf-8")
            self.assertIn("80.0", svg)
            self.assertIn("90.0", svg)
            self.assertIn("True modulation", svg)

    def test_confusion_matrix_requires_square_rows(self) -> None:
        invalid = _result((0.2, 0.4))
        invalid["validation"]["confusion_matrix"] = [[1, 2], [3]]  # type: ignore[index]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "square"):
                render_confusion_matrix_svg(
                    invalid,
                    Path(directory) / "confusion.svg",
                )

    def test_confusion_matrix_supports_test_section_and_title(self) -> None:
        result = _result((0.3, 0.6))
        result["test"] = result.pop("validation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-confusion.svg"

            render_confusion_matrix_svg(
                result,
                path,
                section="test",
                title="Final Test Matrix",
            )

            self.assertIn("Final Test Matrix", path.read_text(encoding="utf-8"))

    def test_multi_arm_analysis_figures_render_without_optional_dependencies(self) -> None:
        snr_rows = [
            {"snr": -10.0, "accuracy_mean": 0.25, "accuracy_sample_std": 0.02},
            {"snr": 10.0, "accuracy_mean": 0.80, "accuracy_sample_std": 0.01},
        ]
        confusion = {"A0": [[8, 2], [3, 7]], "A1": [[9, 1], [1, 9]]}
        class_snr = {"A0": [[0.5, 0.8], [0.4, 0.7]], "A1": [[0.6, 0.9], [0.5, 0.8]]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snr_path = render_multi_snr_accuracy_svg(
                {"A0": snr_rows, "A1": snr_rows}, root / "snr.svg"
            )
            confusion_path = render_confusion_grid_svg(
                confusion,
                ["BPSK", "QPSK"],
                root / "confusion.svg",
                title="High SNR",
            )
            heatmap_path = render_class_snr_grid_svg(
                class_snr,
                ["BPSK", "QPSK"],
                [-10, 10],
                root / "heatmap.svg",
            )

            self.assertIn("A0", snr_path.read_text(encoding="utf-8"))
            self.assertIn("High SNR", confusion_path.read_text(encoding="utf-8"))
            self.assertIn("Class × SNR", heatmap_path.read_text(encoding="utf-8"))
