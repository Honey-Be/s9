"""Abstract base classes for SSM layers (non-multihead, single-kernel-per-axis).

Hierarchy::

    SSMLayerBase
      ├── ComplexSSMLayerBase   →  S9Layer
      └── RealSSMLayerBase      →  RS9Layer, ARS9Layer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

from s9._common.kernel_base import SSMKernelBase, InitMode, Discretization
from s9._common.outer_product import build_outer_product_global_kernel
from s9._common.fft_conv import fftn_convolve_nd, rfftn_convolve_nd
from ypsilon_torch import FPDTypeIdx, get_complex_dtype, get_float_dtype


class SSMLayerBase(nn.Module, ABC):
    """Base for single-kernel-per-axis SSM layers (S9Layer / RS9Layer / ARS9Layer)."""

    @abstractmethod
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_kernels(self) -> nn.ModuleList:
        """Return ModuleList of SSMKernelBase for each spatial axis."""
        ...

    @abstractmethod
    def _fft_conv(self, u: torch.Tensor, k_global: torch.Tensor) -> torch.Tensor:
        """Perform N-D FFT convolution (complex or real variant)."""
        ...

    @abstractmethod
    def _postprocess(self, y: torch.Tensor) -> torch.Tensor:
        """Activation → output_linear → dropout (channel-last permute handled)."""
        ...

    # ------------------------------------------------------------------
    # Forward (template method)
    # ------------------------------------------------------------------

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        k_1d_list = [k(length=L) for k, L in zip(self.kernels, spatial_shapes)]
        k_global = build_outer_product_global_kernel(
            k_1d_list, spatial_shapes, u.shape[1],
        )
        y = self._fft_conv(u, k_global)
        return self._postprocess(y)


class ComplexSSMLayerBase(SSMLayerBase, ABC):
    """Complex-domain layer base (fftn/ifftn, ComplexDropout)."""

    pass


class RealSSMLayerBase(SSMLayerBase, ABC):
    """Real-domain layer base (rfftn/irfftn, nn.Dropout)."""

    pass
