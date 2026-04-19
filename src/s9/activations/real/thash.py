import torch
import torch.nn as nn

from s9.activations.real.base import RealActivationBase


def thash(input: torch.Tensor) -> torch.Tensor:
    """
    ThASh(TanhArSinh) activation:
        f(x) = tanh(asinh(x)) = x / sqrt(1 + x^2)

    Element-wise, shape-preserving, parameter-free.
    Range: (-1, 1) for all real inputs.
    """
    return input / torch.sqrt(1 + input * input)


class ThASh(RealActivationBase):
    r"""
    ThASh(TanhArSinh) activation.

    Definition:
        ThASh(x) = tanh(asinh(x)) = x / sqrt(1 + x^2)

    This is designed as a drop-in replacement for built-in PyTorch
    activations such as nn.Tanh() / nn.Softsign() in contexts where:
      - domain  = R
      - codomain = R
      - image   = (-1, 1)

    Shape:
      - Input:  (*)
      - Output: (*), same shape as input
    """

    __constants__ = []

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return thash(input)

    def extra_repr(self) -> str:
        return ""