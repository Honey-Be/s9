"""qGELU-style quantized real activations for RS9/ARS9."""

from __future__ import annotations

import torch
import torch.nn as nn

from s9.activations.real.base import RealActivationBase
from s9.quantization.quantizers import fake_quant


class QThASh(RealActivationBase):
    """Quantized ThASh: fake-quant input, apply ThASh, fake-quant output."""

    def __init__(self, input_bits: int = 8, output_bits: int = 8) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = fake_quant(input, self.input_bits)
        y = x / torch.sqrt(1 + x * x)
        return fake_quant(y, self.output_bits)


class QHGLU(RealActivationBase):
    """Quantized HGLU: fake-quant input, apply HGLU, fake-quant output."""

    def __init__(self, k: float, input_bits: int = 8, output_bits: int = 8) -> None:
        super().__init__()
        self.k = k
        self.input_bits = input_bits
        self.output_bits = output_bits

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = fake_quant(input, self.input_bits)
        y = (x + torch.sqrt(self.k + x * x)) / 2
        return fake_quant(y, self.output_bits)
