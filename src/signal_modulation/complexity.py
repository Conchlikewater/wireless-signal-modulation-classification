"""Dependency-free parameter and Conv/Linear MAC estimates for fixed inputs."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def estimate_conv_linear_macs(
    model: nn.Module,
    *,
    channels: int = 2,
    sequence_length: int = 128,
) -> int:
    """Count multiply-accumulates for Conv1d and Linear at batch size one.

    BatchNorm, activation, pooling, Dropout, and bias additions are excluded. The
    accompanying FLOP convention is two FLOPs per MAC.
    """

    if channels <= 0 or sequence_length <= 0:
        raise ValueError("channels and sequence_length must be greater than zero")
    macs = 0
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(
        module: nn.Conv1d,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal macs
        output_elements = output.shape[1] * output.shape[2]
        kernel_macs = (module.in_channels // module.groups) * module.kernel_size[0]
        macs += int(output_elements * kernel_macs)

    def linear_hook(
        module: nn.Linear,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal macs
        macs += int(output.numel() * module.in_features)

    hook_factories: tuple[tuple[type[nn.Module], Callable], ...] = (
        (nn.Conv1d, conv_hook),
        (nn.Linear, linear_hook),
    )
    for module in model.modules():
        for module_type, hook in hook_factories:
            if isinstance(module, module_type):
                hooks.append(module.register_forward_hook(hook))
                break

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, channels, sequence_length))
    finally:
        for handle in hooks:
            handle.remove()
        model.train(was_training)
    return macs
