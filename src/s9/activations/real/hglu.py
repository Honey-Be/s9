import torch
import torch.nn as nn

from s9.activations.real.base import RealActivationBase


def hglu(input: torch.Tensor, k: float) -> torch.Tensor:
    """
    HGLU_k(Hyperbolic Gain Linear Unit with positive hyperparameter k) activation:
        f(x) = (x + sqrt(k + x^2)) / 2

    Element-wise, shape-preserving, parameter-free.
    Range: (0, +inf) for all real inputs.
    """
    if not (k > 0):
        raise Exception("k must be positive!")
    return (input + torch.sqrt(k + input * input)) / 2


class HGLU(RealActivationBase):
    r"""
    HGLU_k(Hyperbolic Gain Linear Unit with positive hyperparameter k) activation

    Definition:
        HGLU_k(x) = (x + sqrt(k + x^2)) / 2

    This is designed as a drop-in replacement for built-in PyTorch
    activations such as nn.Tanh() / nn.Softsign() in contexts where:
      - domain  = R
      - codomain = R
      - image   = (0, +inf)

    Shape:
      - Input:  (*)
      - Output: (*), same shape as input
    """

    __constants__ = []

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k: float = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return hglu(input, self.k)

    def extra_repr(self) -> str:
        return ""