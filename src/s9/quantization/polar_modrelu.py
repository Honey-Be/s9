"""INT8-friendly polar variant of StableModReLU."""

from __future__ import annotations

import torch
import torch.nn as nn

from s9.base import ComplexActivationFunctionBase, FPDTypeIdx, FLOAT_DTYPES_DICT
from s9.quantization.quantizers import fake_quant


class PolarStableModReLU(ComplexActivationFunctionBase):
    """StableModReLU variant with polar decomposition + per-component quantization.

    Input complex is decomposed into (r, theta), each quantized to 8-bit
    independently, then recombined. This preserves phase better than
    Cartesian quantization for magnitude-based activations.
    """

    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64,
                 r_bits: int = 8, theta_bits: int = 8):
        super().__init__(features, eps, dtype_idx)
        self.r_bits = r_bits
        self.theta_bits = theta_bits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = z.abs().clamp(min=self.eps)
        theta = torch.angle(z)

        # Quantize polar components
        r_q = fake_quant(r, self.r_bits)
        theta_q = fake_quant(theta, self.theta_bits)

        # ModReLU activation on magnitude
        activated_r = torch.relu(r_q + self.bias)

        # Reconstruct
        return activated_r * torch.exp(1j * theta_q)
