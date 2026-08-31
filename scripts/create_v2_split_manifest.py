"""Create the frozen Wireless V2 split manifest without training or evaluation."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from signal_modulation.data_integrity import sha256_file
from signal_modulation.radioml import RadioMLDataset, load_restricted_radioml_pickle
from signal_modulation.split_manifest import create_split_manifest_bundle
from signal_modulation.v2_experiment import (
    V2_DATASET_SHA256,
    V2_PROTOCOL_VERSION,
    V2_SAMPLE_COUNT,
    V2_SPLIT_SEED,
)


@dataclass(frozen=True, slots=True)
class RadioMLIndexMetadata:
    """Labels and SNRs reconstructed from group keys and array sizes only."""

    labels: np.ndarray
    snrs: np.ndarray
    modulations: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def build_index_metadata(grouped_signals: RadioMLDataset) -> RadioMLIndexMetadata:
    """Reconstruct global index metadata without reading an I/Q sample value."""

    if not grouped_signals:
        raise ValueError("RadioML dataset cannot be empty")
    for key, signals in grouped_signals.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not isinstance(key[1], (int, np.integer))
        ):
            raise ValueError("every RadioML key must be a (modulation, snr) tuple")
        if not isinstance(signals, np.ndarray) or signals.shape != (1000, 2, 128):
            raise ValueError("every RadioML group must have shape (1000, 2, 128)")
        if signals.dtype != np.float32:
            raise ValueError("RadioML I/Q arrays must use float32 values")

    modulations = tuple(sorted({key[0] for key in grouped_signals}))
    snr_levels = tuple(sorted({int(key[1]) for key in grouped_signals}))
    if len(modulations) != 11 or snr_levels != tuple(range(-20, 20, 2)):
        raise ValueError("RadioML groups do not match the 2016.10A class/SNR profile")
    expected_keys = {
        (modulation, snr) for modulation in modulations for snr in snr_levels
    }
    if set(grouped_signals) != expected_keys:
        raise ValueError("RadioML 2016.10A must contain every modulation/SNR group")

    class_to_index = {
        modulation: index for index, modulation in enumerate(modulations)
    }
    label_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    for modulation, snr in sorted(grouped_signals, key=lambda key: (key[0], int(key[1]))):
        group_size = int(grouped_signals[(modulation, snr)].shape[0])
        label_parts.append(
            np.full(group_size, class_to_index[modulation], dtype=np.int64)
        )
        snr_parts.append(np.full(group_size, int(snr), dtype=np.int16))

    labels = np.concatenate(label_parts)
    snrs = np.concatenate(snr_parts)
    if labels.size != V2_SAMPLE_COUNT:
        raise ValueError(f"RadioML dataset must contain {V2_SAMPLE_COUNT} samples")
    return RadioMLIndexMetadata(labels=labels, snrs=snrs, modulations=modulations)


def clean_repository_commit(repository_root: Path) -> str:
    """Return HEAD only when the implementation being recorded is committed."""

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("commit W1 implementation before freezing the split manifest")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit:
        raise RuntimeError("cannot resolve the repository HEAD commit")
    return commit


def main() -> None:
    args = parse_args()
    actual_dataset_sha256 = sha256_file(args.pickle_file)
    if actual_dataset_sha256 != V2_DATASET_SHA256:
        raise ValueError("dataset SHA-256 does not match the frozen V2 protocol")

    repository_root = Path(__file__).resolve().parents[1]
    implementation_commit = clean_repository_commit(repository_root)
    grouped_signals = load_restricted_radioml_pickle(args.pickle_file)
    metadata = build_index_metadata(grouped_signals)
    manifest_path = create_split_manifest_bundle(
        metadata.labels,
        metadata.snrs,
        metadata.modulations,
        args.output_directory,
        dataset_sha256=actual_dataset_sha256,
        split_seed=V2_SPLIT_SEED,
        protocol_version=V2_PROTOCOL_VERSION,
        implementation_git_commit=implementation_commit,
    )
    print(f"split_manifest={manifest_path}")
    print(f"split_seed={V2_SPLIT_SEED}")
    print(f"implementation_git_commit={implementation_commit}")


if __name__ == "__main__":
    main()
