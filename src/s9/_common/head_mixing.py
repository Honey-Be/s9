"""Shared utilities for multi-head channel normalisation and pointwise post-processing."""

from collections.abc import Sequence

import torch
import torch.nn as nn


def normalize_head_channels(d_model: int, n_heads: int, head_channels: Sequence[int] | int) -> list[int]:
    """Resolve per-head channel counts from user-supplied shorthand.

    Rules:
        * ``head_channels`` is an ``int`` → ``[head_channels]`` (single-value shorthand).
        * ``head_channels == []`` → ``d_model // n_heads`` for every head.
        * ``head_channels == [ch]`` → *ch* for every head.
        * ``len(head_channels) == n_heads`` → use as-is.
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")

    # Accept bare int as single-element shorthand (used by BiaffineRS9Layer
    # which passes latent_channels: int as channels to the parent).
    if isinstance(head_channels, int):
        head_channels = [head_channels]

    if len(head_channels) == 0:
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads} "
                "when head_channels is omitted"
            )
        return [d_model // n_heads] * n_heads

    if len(head_channels) == 1:
        ch = int(head_channels[0])
        if ch <= 0:
            raise ValueError(f"head_channels must be positive, got {head_channels[0]}")
        return [ch] * n_heads

    if len(head_channels) != n_heads:
        raise ValueError(f"Expected {n_heads} head channel sizes, got {len(head_channels)}")

    channels_list = [int(ch) for ch in head_channels]
    if any(ch <= 0 for ch in channels_list):
        raise ValueError("All head channel sizes must be positive")
    return channels_list


def apply_channel_last_pointwise(
    y: torch.Tensor,
    spatial_dims: int,
    activation: nn.Module,
    output_linear: nn.Linear,
    dropout: nn.Module,
) -> torch.Tensor:
    """Permute to channel-last, apply activation → linear → dropout, permute back.

    Works for both complex and real tensors; the caller is responsible for
    supplying matching *output_linear* and *dropout* dtypes.
    """
    permute_order = [0] + list(range(2, 2 + spatial_dims)) + [1]
    inv_permute_order = [0, spatial_dims + 1] + list(range(1, 1 + spatial_dims))

    y = y.permute(*permute_order)
    y = activation(y)
    y = output_linear(y)
    y = dropout(y)
    return y.permute(*inv_permute_order)
