from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from collections.abc import Sequence

from s9.base import FPDTypeIdx, get_float_dtype
from s9.multihead_rs9_modules import (
    MultiheadRS9Head,
    _normalize_head_channels,
)
from s9.biaffine_rs9_modules import BiaffineRS9Head, BiaffineRS9Layer


__all__ = [
    "GatedDeltaRS9Layer",
    "BiaffineGatedDeltaRS9Layer",
]


class GatedDeltaRS9Layer(nn.Module):
    """Gated Delta RS9 Layer: RS9-based redesign of Gated DeltaNet (real-valued, multi-head).

    Replaces Gated DeltaNet's token-level rank-2 recurrence with multi-head
    RS9 FFT convolution, eliminating fixed-size state, rank-2 transition,
    and scalar gating constraints.
    """

    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], nn.Module],
        n_heads: int,
        head_channels: Sequence[int],
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims
        self.n_heads: int = n_heads
        self.dtype_idx: FPDTypeIdx = dtype_idx

        f_dtype = get_float_dtype(dtype_idx)

        # Multi-head RS9 convolution heads
        channels_list = _normalize_head_channels(d_model, n_heads, head_channels)
        self.heads: nn.ModuleList = nn.ModuleList([
            MultiheadRS9Head(
                d_model=d_model,
                spatial_dims=spatial_dims,
                head_channels=ch,
                dtype_idx=dtype_idx,
            )
            for ch in channels_list
        ])

        # Gate projections (real-valued)
        self.gate_proj: nn.Linear = nn.Linear(d_model, 2 * d_model, dtype=f_dtype)
        self.z_proj: nn.Linear = nn.Linear(d_model, d_model, dtype=f_dtype)

        # Initialize gate bias: alpha ~ 0.73 (residual), beta ~ 0.27 (update)
        with torch.no_grad():
            self.gate_proj.bias[:d_model].fill_(1.0)
            self.gate_proj.bias[d_model:].fill_(-1.0)

        # Post-processing
        self.activation: nn.Module = gen_activation(d_model, eps, dtype_idx)
        self.output_linear: nn.Linear = nn.Linear(d_model, d_model, bias=False, dtype=f_dtype)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        self.norm: nn.RMSNorm = nn.RMSNorm(d_model, eps=eps, dtype=f_dtype)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, d_model, D1, D2, ...) Real input
        Returns:
            (B, d_model, D1, D2, ...) Real output
        """
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        permute_order = [0] + list(range(2, 2 + self.spatial_dims)) + [1]
        inv_permute_order = [0, self.spatial_dims + 1] + list(range(1, 1 + self.spatial_dims))

        # Step 1: Gate computation (channel-last)
        u_cl = u.permute(*permute_order)  # (B, D1, ..., d_model) real
        gate_input = u_cl

        gate_logits = self.gate_proj(gate_input)
        alpha, beta = torch.sigmoid(gate_logits).chunk(2, dim=-1)
        z = self.z_proj(gate_input)

        # Step 2: Multi-head RS9 convolution
        y = torch.zeros_like(u)
        for head in self.heads:
            y = y + head(u)

        # Step 3: Gated delta combination (channel-last)
        y_cl = y.permute(*permute_order)
        combined = alpha * u_cl + beta * y_cl

        # Step 4: Output with gating
        combined = self.activation(combined)
        combined = self.output_linear(combined)
        combined = self.dropout(combined)
        combined = self.norm(combined) * F.silu(z)

        return combined.permute(*inv_permute_order)


class BiaffineGatedDeltaRS9Layer(nn.Module):
    """Biaffine Gated Delta RS9 Layer: RS9-based redesign of Gated DeltaNet
    with biaffine channel coupling (real-valued).

    Uses BiaffineRS9Head for richer input-output channel interactions
    via low-rank biaffine channel mixing.
    """

    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], nn.Module],
        n_heads: int,
        latent_channels: Sequence[int],
        channel_embed_dim: int = 16,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims
        self.n_heads: int = n_heads
        self.dtype_idx: FPDTypeIdx = dtype_idx

        f_dtype = get_float_dtype(dtype_idx)

        # Biaffine RS9 convolution heads
        channels_list = _normalize_head_channels(d_model, n_heads, latent_channels)
        mapper = BiaffineRS9Layer.HeadMapper(d_model, spatial_dims, channel_embed_dim)
        self.heads: nn.ModuleList = nn.ModuleList([
            mapper.mapping(ch, dtype_idx)
            for ch in channels_list
        ])

        # Gate projections (real-valued)
        self.gate_proj: nn.Linear = nn.Linear(d_model, 2 * d_model, dtype=f_dtype)
        self.z_proj: nn.Linear = nn.Linear(d_model, d_model, dtype=f_dtype)

        # Initialize gate bias: alpha ~ 0.73, beta ~ 0.27
        with torch.no_grad():
            self.gate_proj.bias[:d_model].fill_(1.0)
            self.gate_proj.bias[d_model:].fill_(-1.0)

        # Post-processing
        self.activation: nn.Module = gen_activation(d_model, eps, dtype_idx)
        self.output_linear: nn.Linear = nn.Linear(d_model, d_model, bias=False, dtype=f_dtype)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        self.norm: nn.RMSNorm = nn.RMSNorm(d_model, eps=eps, dtype=f_dtype)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, d_model, D1, D2, ...) Real input
        Returns:
            (B, d_model, D1, D2, ...) Real output
        """
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        permute_order = [0] + list(range(2, 2 + self.spatial_dims)) + [1]
        inv_permute_order = [0, self.spatial_dims + 1] + list(range(1, 1 + self.spatial_dims))

        # Step 1: Gate computation
        u_cl = u.permute(*permute_order)
        gate_input = u_cl

        gate_logits = self.gate_proj(gate_input)
        alpha, beta = torch.sigmoid(gate_logits).chunk(2, dim=-1)
        z = self.z_proj(gate_input)

        # Step 2: Multi-head biaffine RS9 convolution
        y = torch.zeros_like(u)
        for head in self.heads:
            y = y + head(u)

        # Step 3: Gated delta combination
        y_cl = y.permute(*permute_order)
        combined = alpha * u_cl + beta * y_cl

        # Step 4: Output with gating
        combined = self.activation(combined)
        combined = self.output_linear(combined)
        combined = self.dropout(combined)
        combined = self.norm(combined) * F.silu(z)

        return combined.permute(*inv_permute_order)
