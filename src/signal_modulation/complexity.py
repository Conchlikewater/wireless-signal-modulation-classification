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


def estimate_lstm_macs(
    model: nn.Module,
    *,
    channels: int = 2,
    sequence_length: int = 128,
) -> int:
    """Estimate matrix MACs inside standard LSTM gates at batch size one.

    The estimate includes input/recurrent matrix products for all four gates and
    excludes bias additions, sigmoid/tanh, and elementwise cell-state updates.
    """

    if channels <= 0 or sequence_length <= 0:
        raise ValueError("channels and sequence_length must be greater than zero")
    macs = 0
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def lstm_hook(
        module: nn.LSTM,
        inputs: tuple[Tensor, ...],
        _output: tuple[Tensor, tuple[Tensor, Tensor]],
    ) -> None:
        nonlocal macs
        if module.proj_size:
            raise ValueError("projected LSTMs are outside the estimator convention")
        sequence = inputs[0]
        if module.batch_first:
            batch_size, steps = int(sequence.shape[0]), int(sequence.shape[1])
        else:
            steps, batch_size = int(sequence.shape[0]), int(sequence.shape[1])
        directions = 2 if module.bidirectional else 1
        for layer in range(module.num_layers):
            layer_input = module.input_size if layer == 0 else module.hidden_size * directions
            macs += (
                batch_size
                * steps
                * directions
                * 4
                * module.hidden_size
                * (layer_input + module.hidden_size)
            )

    for module in model.modules():
        if isinstance(module, nn.LSTM):
            hooks.append(module.register_forward_hook(lstm_hook))
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
