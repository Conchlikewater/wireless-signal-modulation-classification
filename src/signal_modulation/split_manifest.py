"""Frozen split manifests and development-only data access for Wireless V2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from signal_modulation.data_integrity import sha256_file
from signal_modulation.dataset import create_data_loader
from signal_modulation.experiment import write_json_atomic
from signal_modulation.splitting import (
    DatasetSplit,
    assert_split_integrity,
    stratified_split_indices,
)


SPLIT_MANIFEST_SCHEMA_VERSION = "wireless-split-manifest-v1"
SPLIT_INDICES_FILENAME = "split_indices.npz"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentSplit:
    """Train/validation indices plus non-readable metadata for the sealed test split."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    sample_count: int
    split_seed: int
    test_index_count: int
    test_index_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DevelopmentPartitions:
    """V2 development views; deliberately has no test Dataset or test indices."""

    train: Subset
    validation: Subset
    test_index_count: int
    test_index_sha256: str


@dataclass(frozen=True, slots=True)
class DevelopmentLoaders:
    """Train/validation loaders tied to one frozen split and one run seed."""

    train: DataLoader
    validation: DataLoader
    split_seed: int
    run_seed: int
    manifest_sha256: str
    test_index_count: int
    test_index_sha256: str


def sha256_indices(indices: np.ndarray) -> str:
    """Hash one index vector in a platform-independent little-endian int64 form."""

    values = np.asarray(indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    canonical = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_modulations(
    labels: np.ndarray,
    modulations: Sequence[str],
) -> tuple[str, ...]:
    names = tuple(modulations)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("modulations must contain non-empty class names")
    label_values = np.asarray(labels)
    if label_values.ndim != 1 or not np.issubdtype(label_values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if np.any(label_values < 0) or np.any(label_values >= len(names)):
        raise ValueError("labels contain a class index outside modulations")
    return names


def _stratum_records(
    labels: np.ndarray,
    snrs: np.ndarray,
    modulations: tuple[str, ...],
    split: DatasetSplit,
) -> list[dict[str, int | str]]:
    """Count test membership by subtraction without indexing test labels."""

    label_values = np.asarray(labels, dtype=np.int64)
    snr_values = np.asarray(snrs)
    records: list[dict[str, int | str]] = []
    keys = sorted(
        {
            (int(label), int(snr))
            for label, snr in zip(label_values, snr_values, strict=True)
        }
    )
    for label, snr in keys:
        all_mask = (label_values == label) & (snr_values == snr)
        train_count = int(np.count_nonzero(all_mask[split.train]))
        validation_count = int(np.count_nonzero(all_mask[split.validation]))
        total_count = int(np.count_nonzero(all_mask))
        records.append(
            {
                "label_index": label,
                "modulation": modulations[label],
                "snr": snr,
                "total_count": total_count,
                "train_count": train_count,
                "validation_count": validation_count,
                "test_count": total_count - train_count - validation_count,
            }
        )
    return records


def create_split_manifest_bundle(
    labels: np.ndarray,
    snrs: np.ndarray,
    modulations: Sequence[str],
    output_directory: str | Path,
    *,
    dataset_sha256: str,
    split_seed: int,
    protocol_version: str,
    implementation_git_commit: str,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Path:
    """Create one immutable index bundle without reading any signal sample."""

    label_values = np.asarray(labels)
    snr_values = np.asarray(snrs)
    names = _validate_modulations(label_values, modulations)
    if snr_values.ndim != 1 or snr_values.size != label_values.size:
        raise ValueError("snrs must be one-dimensional and match labels")
    if len(dataset_sha256) != 64:
        raise ValueError("dataset_sha256 must be a 64-character digest")
    if not protocol_version:
        raise ValueError("protocol_version cannot be empty")
    if not implementation_git_commit:
        raise ValueError("implementation_git_commit cannot be empty")

    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"split manifest directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    split = stratified_split_indices(
        label_values,
        snr_values,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    indices_path = destination / SPLIT_INDICES_FILENAME
    temporary_indices_path = indices_path.with_suffix(indices_path.suffix + ".tmp")
    with temporary_indices_path.open("wb") as file_handle:
        np.savez_compressed(
            file_handle,
            train=np.asarray(split.train, dtype=np.int64),
            validation=np.asarray(split.validation, dtype=np.int64),
            test=np.asarray(split.test, dtype=np.int64),
        )
    temporary_indices_path.replace(indices_path)

    payload = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "protocol_version": protocol_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "sha256": dataset_sha256.lower(),
            "sample_count": int(label_values.size),
            "modulations": list(names),
            "snrs": [int(value) for value in sorted(np.unique(snr_values))],
        },
        "split": {
            "split_seed": split_seed,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
            "algorithm": (
                "signal_modulation.splitting.stratified_split_indices-v1"
            ),
            "numpy_version": np.__version__,
            "implementation_git_commit": implementation_git_commit,
        },
        "indices": {
            "filename": SPLIT_INDICES_FILENAME,
            "file_sha256": sha256_file(indices_path),
            "train": {
                "count": int(split.train.size),
                "sha256": sha256_indices(split.train),
            },
            "validation": {
                "count": int(split.validation.size),
                "sha256": sha256_indices(split.validation),
            },
            "test": {
                "count": int(split.test.size),
                "sha256": sha256_indices(split.test),
            },
        },
        "strata": _stratum_records(
            label_values,
            snr_values,
            names,
            split,
        ),
    }
    return write_json_atomic(destination / SPLIT_MANIFEST_FILENAME, payload)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _index_record(
    indices_payload: dict[str, object],
    name: str,
) -> tuple[int, str]:
    record = _mapping(indices_payload.get(name), f"indices.{name}")
    count = record.get("count")
    digest = record.get("sha256")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"indices.{name}.count must be a positive integer")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"indices.{name}.sha256 must be a 64-character digest")
    return count, digest.lower()


def _immutable_indices(values: np.ndarray) -> np.ndarray:
    indices = np.array(values, dtype=np.int64, copy=True)
    indices.setflags(write=False)
    return indices


def load_frozen_development_split(
    manifest_path: str | Path,
    *,
    expected_dataset_sha256: str,
    expected_split_seed: int,
    expected_protocol_version: str,
    expected_sample_count: int | None = None,
) -> FrozenDevelopmentSplit:
    """Verify all indices, then return only train/validation and sealed-test metadata."""

    source = Path(manifest_path)
    payload = _mapping(
        json.loads(source.read_text(encoding="utf-8")),
        "manifest",
    )
    if payload.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("split manifest schema version does not match")
    if payload.get("protocol_version") != expected_protocol_version:
        raise ValueError("split manifest protocol version does not match")

    dataset_payload = _mapping(payload.get("dataset"), "dataset")
    dataset_digest = dataset_payload.get("sha256")
    sample_count = dataset_payload.get("sample_count")
    if dataset_digest != expected_dataset_sha256.lower():
        raise ValueError("split manifest dataset SHA-256 does not match")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("dataset.sample_count must be a positive integer")
    if expected_sample_count is not None and sample_count != expected_sample_count:
        raise ValueError("split manifest sample count does not match")

    split_payload = _mapping(payload.get("split"), "split")
    split_seed = split_payload.get("split_seed")
    if split_seed != expected_split_seed:
        raise ValueError("split manifest seed does not match")

    indices_payload = _mapping(payload.get("indices"), "indices")
    filename = indices_payload.get("filename")
    file_digest = indices_payload.get("file_sha256")
    if filename != SPLIT_INDICES_FILENAME:
        raise ValueError("split manifest index filename is not allowed")
    if not isinstance(file_digest, str) or len(file_digest) != 64:
        raise ValueError("indices.file_sha256 must be a 64-character digest")
    indices_path = source.parent / SPLIT_INDICES_FILENAME
    if sha256_file(indices_path) != file_digest.lower():
        raise ValueError("split index archive SHA-256 does not match")

    with np.load(indices_path, allow_pickle=False) as archive:
        if set(archive.files) != {"train", "validation", "test"}:
            raise ValueError("split index archive must contain train/validation/test")
        split = DatasetSplit(
            train=np.asarray(archive["train"]),
            validation=np.asarray(archive["validation"]),
            test=np.asarray(archive["test"]),
        )
    assert_split_integrity(split, sample_count=sample_count)

    expected_records = {
        name: _index_record(indices_payload, name)
        for name in ("train", "validation", "test")
    }
    for name in ("train", "validation", "test"):
        values = getattr(split, name)
        expected_count, expected_digest = expected_records[name]
        if values.size != expected_count or sha256_indices(values) != expected_digest:
            raise ValueError(f"{name} split indices do not match the manifest")

    return FrozenDevelopmentSplit(
        train_indices=_immutable_indices(split.train),
        validation_indices=_immutable_indices(split.validation),
        sample_count=sample_count,
        split_seed=expected_split_seed,
        test_index_count=expected_records["test"][0],
        test_index_sha256=expected_records["test"][1],
        manifest_sha256=sha256_file(source),
    )


def create_development_partitions(
    dataset: Dataset,
    split: FrozenDevelopmentSplit,
) -> DevelopmentPartitions:
    """Create train/validation views without constructing any test Dataset."""

    if len(dataset) != split.sample_count:
        raise ValueError("dataset length does not match the frozen split manifest")
    if np.intersect1d(split.train_indices, split.validation_indices).size:
        raise ValueError("train and validation indices overlap")
    return DevelopmentPartitions(
        train=Subset(dataset, split.train_indices),
        validation=Subset(dataset, split.validation_indices),
        test_index_count=split.test_index_count,
        test_index_sha256=split.test_index_sha256,
    )


def create_development_loaders(
    dataset: Dataset,
    split: FrozenDevelopmentSplit,
    *,
    run_seed: int,
    train_batch_size: int,
    evaluation_batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DevelopmentLoaders:
    """Use run_seed for loader order while keeping the loaded split unchanged."""

    partitions = create_development_partitions(dataset, split)
    return DevelopmentLoaders(
        train=create_data_loader(
            partitions.train,
            batch_size=train_batch_size,
            shuffle=True,
            seed=run_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        validation=create_data_loader(
            partitions.validation,
            batch_size=evaluation_batch_size,
            shuffle=False,
            seed=run_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        split_seed=split.split_seed,
        run_seed=run_seed,
        manifest_sha256=split.manifest_sha256,
        test_index_count=split.test_index_count,
        test_index_sha256=split.test_index_sha256,
    )
