"""Per-component bit-width configuration for quantized SSM layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class QuantConfig:
    """Bit-budget specification following Q-S5 sensitivity findings.

    Default: A/dt stay fp32, B/C at int8, output_linear at int4, input at int8.
    Set a field to ``None`` to skip quantization for that component.
    """

    # Weight bit-widths
    w_bits_A: Optional[int] = None        # fp32 by default (not quantized)
    w_bits_B: Optional[int] = 8
    w_bits_C: Optional[int] = 8
    w_bits_output: Optional[int] = 4

    # Activation bit-widths
    a_bits_input: Optional[int] = 8
    a_bits_activation: Optional[int] = 8

    # Accumulation mode
    accumulation: Literal["int32", "fp32"] = "int32"

    # Discrete stability check
    stability_epsilon: float = 1e-3
    enforce_stability: bool = True
