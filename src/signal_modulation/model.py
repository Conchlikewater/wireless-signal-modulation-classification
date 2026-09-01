"""Small 1D CNN models for I/Q modulation classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SimpleCNN1D(nn.Module):
    """Convert a batch of two-channel I/Q sequences into class logits."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")

        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return unnormalized class scores with shape (batch, num_classes)."""

        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape (batch, 2, length)")
        if inputs.shape[0] == 0 or inputs.shape[2] < 2:
            raise ValueError("inputs require a non-empty batch and length of at least two")

        features = self.features(inputs)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened)


class TemporalCNN1D(nn.Module):
    """A stronger fixed-window CNN that preserves coarse temporal position."""

    def __init__(
        self,
        num_classes: int,
        *,
        sequence_length: int = 128,
        temporal_bins: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if temporal_bins <= 0:
            raise ValueError("temporal_bins must be greater than zero")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        downsampled_length = sequence_length // 4
        if sequence_length < 4 or downsampled_length % temporal_bins != 0:
            raise ValueError(
                "sequence_length after two pooling stages must divide into temporal_bins"
            )

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.temporal_bins = temporal_bins
        self.dropout = dropout
        final_pool_size = downsampled_length // temporal_bins
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=final_pool_size, stride=final_pool_size),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * temporal_bins, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits while retaining coarse ordering in the I/Q sequence."""

        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape (batch, 2, length)")
        if inputs.shape[0] == 0:
            raise ValueError("inputs require a non-empty batch")
        if inputs.shape[2] != self.sequence_length:
            raise ValueError(
                f"inputs must use the configured sequence length {self.sequence_length}"
            )

        features = self.features(inputs)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened)


class GlobalPoolingTemporalCNN1D(nn.Module):
    """TemporalCNN backbone with global pooling and a capacity-controlled head."""

    def __init__(
        self,
        num_classes: int,
        *,
        sequence_length: int = 128,
        hidden_features: int = 947,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if sequence_length != 128:
            raise ValueError("the frozen A2-G structure requires sequence_length=128")
        if hidden_features <= 0:
            raise ValueError("hidden_features must be greater than zero")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_features = hidden_features
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.aggregation = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return logits after removing temporal position through global pooling."""

        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape (batch, 2, length)")
        if inputs.shape[0] == 0:
            raise ValueError("inputs require a non-empty batch")
        if inputs.shape[2] != self.sequence_length:
            raise ValueError(
                f"inputs must use the configured sequence length {self.sequence_length}"
            )

        features = self.features(inputs)
        pooled = self.aggregation(features)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.classifier(flattened)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that will be updated by an optimizer."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
