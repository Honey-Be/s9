"""Outer-product construction of N-D global kernels from 1D per-axis kernels.

This is dtype-agnostic: works for both complex and real kernels.
"""

from collections.abc import Sequence

import torch


def build_outer_product_global_kernel(
    kernels_1d: Sequence[torch.Tensor],
    spatial_shapes: Sequence[int],
    channels: int,
) -> torch.Tensor:
    """Build an N-D global kernel via outer product of 1D kernels.

    Args:
        kernels_1d: Per-axis 1D kernels, each of shape ``(channels, L_i)``.
        spatial_shapes: Length of each spatial axis ``(L_0, L_1, ...)``.
        channels: Number of channels (first dim of each kernel).

    Returns:
        ``(channels, L_0, L_1, ...)`` global kernel tensor.
    """
    if len(kernels_1d) != len(spatial_shapes):
        raise ValueError(
            f"kernels_1d and spatial_shapes must have the same length, "
            f"got {len(kernels_1d)} and {len(spatial_shapes)}"
        )

    spatial_dims = len(spatial_shapes)
    k_global = kernels_1d[0]
    view_shape = [channels] + [1] * spatial_dims
    view_shape[1] = spatial_shapes[0]
    k_global = k_global.view(*view_shape)

    for dim_idx in range(1, spatial_dims):
        next_view_shape = [channels] + [1] * spatial_dims
        next_view_shape[dim_idx + 1] = spatial_shapes[dim_idx]
        k_global = k_global * kernels_1d[dim_idx].view(*next_view_shape)

    return k_global
