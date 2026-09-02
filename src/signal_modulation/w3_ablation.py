"""Frozen W3 model construction, initialization audits, and complexity metadata."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from signal_modulation.complexity import estimate_conv_linear_macs, estimate_lstm_macs
from signal_modulation.model import (
    GlobalPoolingTemporalCNN1D,
    LSTMTemporalCNN1D,
    TemporalCNN1D,
    count_trainable_parameters,
)
from signal_modulation.reproducibility import configure_reproducibility


W3_NEW_CONFIGURATIONS = ("A2-G", "A3")
W3_REUSED_CONFIGURATION = "A2-T"
W3_LATENCY_REFERENCE_SEED = 20260901


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and values in a canonical order."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def temporal_backbone_state(model: TemporalCNN1D) -> dict[str, Tensor]:
    """Copy the shared TemporalCNN layers before its final aggregation pool."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.features[:-1].state_dict().items()
    }


def reconstruct_a2t_initial_backbone(
    run_seed: int,
    *,
    num_classes: int,
) -> tuple[dict[str, Tensor], str]:
    """Reconstruct the deterministic A2-T initial backbone used by W2 A1."""

    configure_reproducibility(run_seed)
    reference = TemporalCNN1D(num_classes=num_classes, dropout=0.3)
    state = temporal_backbone_state(reference)
    return state, state_dict_sha256(state)


def create_w3_model(
    configuration: str,
    *,
    num_classes: int,
    initial_backbone: Mapping[str, Tensor] | None = None,
) -> nn.Module:
    """Construct registered W3 arms and the preregistered W5 A2-L arm."""

    if configuration == "A2-G":
        if initial_backbone is None:
            raise ValueError("A2-G requires the frozen initial backbone state")
        model = GlobalPoolingTemporalCNN1D(num_classes=num_classes)
        model.features.load_state_dict(initial_backbone, strict=True)
        return model
    if configuration == "A3":
        if initial_backbone is not None:
            raise ValueError("A3 does not accept an A2 backbone override")
        return TemporalCNN1D(num_classes=num_classes, dropout=0.0)
    if configuration == "A2-L":
        if initial_backbone is None:
            raise ValueError("A2-L requires the frozen initial backbone state")
        model = LSTMTemporalCNN1D(num_classes=num_classes, dropout=0.3)
        model.features.load_state_dict(initial_backbone, strict=True)
        return model
    raise ValueError(f"unsupported W3 configuration: {configuration}")


def verify_dropout_initialization(
    run_seed: int,
    *,
    num_classes: int,
) -> tuple[str, str]:
    """Prove p=0.3 and p=0 start from identical trainable/buffer state."""

    configure_reproducibility(run_seed)
    baseline = TemporalCNN1D(num_classes=num_classes, dropout=0.3)
    baseline_hash = state_dict_sha256(baseline.state_dict())
    configure_reproducibility(run_seed)
    no_dropout = TemporalCNN1D(num_classes=num_classes, dropout=0.0)
    no_dropout_hash = state_dict_sha256(no_dropout.state_dict())
    if baseline_hash != no_dropout_hash:
        raise RuntimeError("Dropout ablation changed initial model parameters")
    return baseline_hash, no_dropout_hash


def architecture_profile(configuration: str, *, num_classes: int = 11) -> dict[str, Any]:
    """Report deterministic parameter and Conv/Linear operation counts."""

    if configuration in {"A1", "A2-T"}:
        model: nn.Module = TemporalCNN1D(num_classes=num_classes, dropout=0.3)
    elif configuration == "A2-G":
        model = GlobalPoolingTemporalCNN1D(num_classes=num_classes)
    elif configuration == "A3":
        model = TemporalCNN1D(num_classes=num_classes, dropout=0.0)
    elif configuration == "A2-L":
        model = LSTMTemporalCNN1D(num_classes=num_classes, dropout=0.3)
    else:
        raise ValueError(f"unsupported architecture profile: {configuration}")
    conv_linear_macs = estimate_conv_linear_macs(model)
    lstm_macs = estimate_lstm_macs(model)
    total_macs = conv_linear_macs + lstm_macs
    profile = {
        "configuration": configuration,
        "trainable_parameters": count_trainable_parameters(model),
        "head_trainable_parameters": (
            count_trainable_parameters(model.aggregation)
            + count_trainable_parameters(model.classifier)
            if configuration == "A2-L"
            else count_trainable_parameters(model.classifier)
        ),
        "conv_linear_macs_per_sample": conv_linear_macs,
    }
    if configuration == "A2-L":
        profile.update(
            {
                "lstm_matrix_macs_per_sample": lstm_macs,
                "total_estimated_macs_per_sample": total_macs,
                "approx_total_flops_per_sample": 2 * total_macs,
                "mac_convention": (
                    "Conv1d/Linear and LSTM input/recurrent matrix multiply-"
                    "accumulates; excludes BatchNorm, activation, pooling, "
                    "Dropout, bias additions, and LSTM elementwise gate/cell updates"
                ),
            }
        )
    else:
        profile.update(
            {
                "approx_conv_linear_flops_per_sample": 2 * conv_linear_macs,
                "mac_convention": (
                    "Conv1d and Linear multiply-accumulates only; excludes "
                    "BatchNorm, activation, pooling, Dropout, and bias additions"
                ),
            }
        )
    return profile


def benchmark_inference_latency(
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int = 512,
    warmup_batches: int = 20,
    measured_batches: int = 100,
) -> dict[str, float | int | str]:
    """Benchmark one fixed zero-valued batch without using evaluation data."""

    if batch_size <= 0 or warmup_batches < 0 or measured_batches <= 0:
        raise ValueError("latency benchmark counts are invalid")
    model = model.to(device).eval()
    inputs = torch.zeros(batch_size, 2, 128, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup_batches):
            model(inputs)
        synchronize()
        durations_ms = []
        for _ in range(measured_batches):
            started = time.perf_counter()
            model(inputs)
            synchronize()
            durations_ms.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(durations_ms)
    mean_ms = statistics.fmean(durations_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "device": str(device),
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "mean_batch_ms": mean_ms,
        "median_batch_ms": statistics.median(durations_ms),
        "p95_batch_ms": ordered[p95_index],
        "mean_samples_per_second": batch_size / (mean_ms / 1000.0),
        "input": "zeros(batch, 2, 128)",
    }
