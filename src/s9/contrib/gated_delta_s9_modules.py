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

from s9.base import ComplexActivationFunctionBase, FPDTypeIdx, get_complex_dtype, get_float_dtype
from s9.modules import ComplexDropout
from s9.multihead_s9_modules import (
    MultiheadS9Head,
    _normalize_head_channels,
)
from s9.biaffine_s9_modules import BiaffineS9Head, BiaffineS9Layer


__all__ = [
    "ComplexRMSNorm",
    "GatedDeltaS9Layer",
    "BiaffineGatedDeltaS9Layer",
]


class ComplexRMSNorm(nn.Module):
    """RMSNorm for complex-valued tensors.

    Normalizes by the RMS of the magnitude, preserving phase.
    Applies a real-valued learnable weight per channel.
    """

    def __init__(self, d_model: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64):
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(
            torch.ones(d_model, dtype=get_float_dtype(dtype_idx))
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., d_model) complex
        rms = torch.sqrt(
            torch.mean(z.real.pow(2) + z.imag.pow(2), dim=-1, keepdim=True) + self.eps
        )
        view_shape = [1] * (z.ndim - 1) + [-1]
        w = self.weight.view(*view_shape)
        return z * (w / rms)


class GatedDeltaS9Layer(nn.Module):
    """Gated Delta S9 Layer: S9-based redesign of Gated DeltaNet (complex-valued, multi-head).

    Replaces Gated DeltaNet's token-level rank-2 recurrence with multi-head
    S9 FFT convolution, eliminating fixed-size state, rank-2 transition,
    and scalar gating constraints.
    """

    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase],
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

        c_dtype = get_complex_dtype(dtype_idx)
        f_dtype = get_float_dtype(dtype_idx)

        # Multi-head S9 convolution heads
        channels_list = _normalize_head_channels(d_model, n_heads, head_channels)
        self.heads: nn.ModuleList = nn.ModuleList([
            MultiheadS9Head(
                d_model=d_model,
                spatial_dims=spatial_dims,
                head_channels=ch,
                dtype_idx=dtype_idx,
            )
            for ch in channels_list
        ])

        # Gate projections (real-valued, applied to magnitude of complex input)
        self.gate_proj: nn.Linear = nn.Linear(d_model, 2 * d_model, dtype=f_dtype)
        self.z_proj: nn.Linear = nn.Linear(d_model, d_model, dtype=f_dtype)

        # Initialize gate bias: alpha ~ 0.73 (residual), beta ~ 0.27 (update)
        with torch.no_grad():
            self.gate_proj.bias[:d_model].fill_(1.0)
            self.gate_proj.bias[d_model:].fill_(-1.0)

        # Post-processing
        self.activation: nn.Module = gen_activation(d_model, eps, dtype_idx)
        self.output_linear: nn.Linear = nn.Linear(d_model, d_model, bias=False, dtype=c_dtype)
        self.dropout: ComplexDropout = ComplexDropout(dropout_p)
        self.norm: ComplexRMSNorm = ComplexRMSNorm(d_model, eps, dtype_idx)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, d_model, D1, D2, ...) Complex input
        Returns:
            (B, d_model, D1, D2, ...) Complex output
        """
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        # Permutation orders
        permute_order = [0] + list(range(2, 2 + self.spatial_dims)) + [1]
        inv_permute_order = [0, self.spatial_dims + 1] + list(range(1, 1 + self.spatial_dims))

        # Step 1: Gate computation (channel-last)
        u_cl = u.permute(*permute_order)  # (B, D1, ..., d_model) complex
        gate_input = u_cl.abs()  # (B, D1, ..., d_model) real

        gate_logits = self.gate_proj(gate_input)  # (B, D1, ..., 2*d_model)
        alpha, beta = torch.sigmoid(gate_logits).chunk(2, dim=-1)  # each (B, D1, ..., d_model)
        z = self.z_proj(gate_input)  # (B, D1, ..., d_model) real

        # Step 2: Multi-head S9 convolution
        y = torch.zeros_like(u)
        for head in self.heads:
            y = y + head(u)

        # Step 3: Gated delta combination (channel-last)
        y_cl = y.permute(*permute_order)  # (B, D1, ..., d_model) complex
        combined = alpha * u_cl + beta * y_cl  # real gates * complex tensors

        # Step 4: Output with gating
        combined = self.activation(combined)

        ydtype = combined.dtype
        if ydtype == get_complex_dtype(32):
            with torch.amp.autocast(device_type=str(combined.device)):
                combined = self.output_linear.to(dtype=torch.complex64)(
                    combined.to(dtype=torch.complex64)
                ).to(dtype=ydtype)
        else:
            combined = self.output_linear(combined)

        combined = self.dropout(combined)
        combined = self.norm(combined) * F.silu(z)

        return combined.permute(*inv_permute_order)


class BiaffineGatedDeltaS9Layer(nn.Module):
    """Biaffine Gated Delta S9 Layer: S9-based redesign of Gated DeltaNet
    with biaffine channel coupling (complex-valued).

    Uses BiaffineS9Head for richer input-output channel interactions
    via low-rank biaffine channel mixing.
    """

    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase],
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

        c_dtype = get_complex_dtype(dtype_idx)
        f_dtype = get_float_dtype(dtype_idx)

        # Biaffine S9 convolution heads
        channels_list = _normalize_head_channels(d_model, n_heads, latent_channels)
        mapper = BiaffineS9Layer.HeadMapper(d_model, spatial_dims, channel_embed_dim)
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
        self.output_linear: nn.Linear = nn.Linear(d_model, d_model, bias=False, dtype=c_dtype)
        self.dropout: ComplexDropout = ComplexDropout(dropout_p)
        self.norm: ComplexRMSNorm = ComplexRMSNorm(d_model, eps, dtype_idx)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, d_model, D1, D2, ...) Complex input
        Returns:
            (B, d_model, D1, D2, ...) Complex output
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
        gate_input = u_cl.abs()

        gate_logits = self.gate_proj(gate_input)
        alpha, beta = torch.sigmoid(gate_logits).chunk(2, dim=-1)
        z = self.z_proj(gate_input)

        # Step 2: Multi-head biaffine S9 convolution
        y = torch.zeros_like(u)
        for head in self.heads:
            y = y + head(u)

        # Step 3: Gated delta combination
        y_cl = y.permute(*permute_order)
        combined = alpha * u_cl + beta * y_cl

        # Step 4: Output with gating
        combined = self.activation(combined)

        ydtype = combined.dtype
        if ydtype == get_complex_dtype(32):
            with torch.amp.autocast(device_type=str(combined.device)):
                combined = self.output_linear.to(dtype=torch.complex64)(
                    combined.to(dtype=torch.complex64)
                ).to(dtype=ydtype)
        else:
            combined = self.output_linear(combined)

        combined = self.dropout(combined)
        combined = self.norm(combined) * F.silu(z)

        return combined.permute(*inv_permute_order)
