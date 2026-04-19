"""Abstract base for real-valued activation functions."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class RealActivationBase(nn.Module, ABC):
    """Marker base class for real-valued activation functions.

    Unlike ``ComplexActivationFunctionBase``, this does **not** mandate
    a specific ``__init__`` signature because real activations range from
    parameter-free (``ThASh``) to hyperparameter-only (``HGLU``).
    """

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...
