"""Symmetric per-tensor quantization with STE for QAT."""

from __future__ import annotations

from typing import Optional

import torch


class _STERound(torch.autograd.Function):
    """Round with straight-through estimator for backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad_output


def symmetric_per_tensor_quantize(
    x: torch.Tensor,
    bits: int,
    is_complex: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor quantization.

    For complex tensors, uses ``max(|x|)`` as the unified scale so that
    phase relationships are preserved (Q-S5 convention).

    Args:
        x: Tensor to quantize.
        bits: Target bit-width.
        is_complex: If True, use magnitude-based scaling.

    Returns:
        ``(x_quantized, scale)`` where ``x_quantized ≈ round(x / scale) * scale``.
    """
    q_max = (1 << (bits - 1)) - 1

    if is_complex:
        abs_max = x.abs().max().clamp(min=1e-8)
    else:
        abs_max = x.abs().max().clamp(min=1e-8)

    scale = abs_max / q_max
    x_scaled = x / scale

    if is_complex and x.is_complex():
        # Round real and imaginary parts separately
        re = _STERound.apply(x_scaled.real).clamp(-q_max, q_max)
        im = _STERound.apply(x_scaled.imag).clamp(-q_max, q_max)
        x_clamped = torch.complex(re, im)
    else:
        x_rounded = _STERound.apply(x_scaled)
        x_clamped = x_rounded.clamp(-q_max, q_max)

    return x_clamped * scale, scale


def fake_quant(
    x: torch.Tensor,
    bits: Optional[int],
    is_complex: bool = False,
) -> torch.Tensor:
    """Fake quantization (quantize + dequantize). No-op if bits is None."""
    if bits is None:
        return x
    q, _ = symmetric_per_tensor_quantize(x, bits, is_complex)
    return q
