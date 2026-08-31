"""Tests for frozen split manifests and development-only data access."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from signal_modulation.split_manifest import (
    SPLIT_INDICES_FILENAME,
    create_development_loaders,
    create_development_partitions,
    create_split_manifest_bundle,
    load_frozen_development_split,
)


def _balanced_metadata() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    labels: list[int] = []
    snrs: list[int] = []
    for label in range(2):
        for snr in (-2, 0):
            labels.extend([label] * 10)
            snrs.extend([snr] * 10)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(snrs, dtype=np.int16),
        ("A", "B"),
    )


class _GuardedIndexDataset(Dataset):
    def __init__(self, sample_count: int, forbidden_indices: set[int]) -> None:
        self.sample_count = sample_count
        self.forbidden_indices = forbidden_indices

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> int:
        if index in self.forbidden_indices:
            raise AssertionError("sealed test sample was accessed")
        return index


class SplitManifestTests(unittest.TestCase):
    dataset_sha256 = "a" * 64
    protocol_version = "test-protocol-v1"
    split_seed = 123

    def _create_bundle(self, directory: Path) -> Path:
        labels, snrs, modulations = _balanced_metadata()
        return create_split_manifest_bundle(
            labels,
            snrs,
            modulations,
            directory,
            dataset_sha256=self.dataset_sha256,
            split_seed=self.split_seed,
            protocol_version=self.protocol_version,
            implementation_git_commit="deadbeef",
            train_fraction=0.60,
            validation_fraction=0.20,
        )

    def _load_bundle(self, manifest_path: Path):
        return load_frozen_development_split(
            manifest_path,
            expected_dataset_sha256=self.dataset_sha256,
            expected_split_seed=self.split_seed,
            expected_protocol_version=self.protocol_version,
            expected_sample_count=40,
        )

    def test_manifest_records_counts_hashes_and_strata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._create_bundle(Path(temporary_directory) / "split")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["indices"]["train"]["count"], 24)
            self.assertEqual(payload["indices"]["validation"]["count"], 8)
            self.assertEqual(payload["indices"]["test"]["count"], 8)
            self.assertEqual(len(payload["strata"]), 4)
            self.assertTrue(
                all(record["test_count"] == 2 for record in payload["strata"])
            )
            self.assertTrue((manifest_path.parent / SPLIT_INDICES_FILENAME).is_file())

    def test_loading_returns_no_test_indices_or_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._create_bundle(Path(temporary_directory) / "split")
            frozen_split = self._load_bundle(manifest_path)
            with np.load(
                manifest_path.parent / SPLIT_INDICES_FILENAME,
                allow_pickle=False,
            ) as archive:
                forbidden_indices = set(int(value) for value in archive["test"])
            dataset = _GuardedIndexDataset(40, forbidden_indices)
            partitions = create_development_partitions(dataset, frozen_split)

            self.assertFalse(hasattr(frozen_split, "test_indices"))
            self.assertFalse(hasattr(partitions, "test"))
            self.assertEqual(partitions.test_index_count, 8)
            list(partitions.train)
            list(partitions.validation)

    def test_run_seed_changes_train_order_not_the_frozen_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._create_bundle(Path(temporary_directory) / "split")
            frozen_split = self._load_bundle(manifest_path)
            dataset = _GuardedIndexDataset(40, forbidden_indices=set())

            first = create_development_loaders(
                dataset,
                frozen_split,
                run_seed=1,
                train_batch_size=5,
                evaluation_batch_size=4,
            )
            repeated = create_development_loaders(
                dataset,
                frozen_split,
                run_seed=1,
                train_batch_size=5,
                evaluation_batch_size=4,
            )
            changed = create_development_loaders(
                dataset,
                frozen_split,
                run_seed=2,
                train_batch_size=5,
                evaluation_batch_size=4,
            )

            first_order = np.concatenate([batch.numpy() for batch in first.train])
            repeated_order = np.concatenate(
                [batch.numpy() for batch in repeated.train]
            )
            changed_order = np.concatenate([batch.numpy() for batch in changed.train])
            validation_order = np.concatenate(
                [batch.numpy() for batch in first.validation]
            )

            np.testing.assert_array_equal(first_order, repeated_order)
            self.assertFalse(np.array_equal(first_order, changed_order))
            np.testing.assert_array_equal(
                validation_order,
                frozen_split.validation_indices,
            )
            self.assertEqual(first.manifest_sha256, changed.manifest_sha256)
            np.testing.assert_array_equal(
                frozen_split.train_indices,
                self._load_bundle(manifest_path).train_indices,
            )

    def test_loaded_indices_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            frozen_split = self._load_bundle(
                self._create_bundle(Path(temporary_directory) / "split")
            )

            with self.assertRaises(ValueError):
                frozen_split.train_indices[0] = 99

    def test_tampered_index_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._create_bundle(Path(temporary_directory) / "split")
            indices_path = manifest_path.parent / SPLIT_INDICES_FILENAME
            with indices_path.open("ab") as file_handle:
                file_handle.write(b"tampered")

            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                self._load_bundle(manifest_path)

    def test_existing_bundle_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "split"
            self._create_bundle(output_directory)

            with self.assertRaises(FileExistsError):
                self._create_bundle(output_directory)


if __name__ == "__main__":
    unittest.main()
