from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override, Callable
except Exception:  # pragma: no cover
    from typing_extensions import override, Callable  # type: ignore

from ypsilon_torch import get_float_dtype, FPDTypeIdx
from s9._common.kernel_base import DiagonalSSMKernelBase, InitMode, Discretization
from s9._common.outer_product import build_outer_product_global_kernel
from s9._common.fft_conv import rfftn_convolve_nd

class RS9SSMKernel(DiagonalSSMKernelBase):
    """
    The Core RS9 Kernel (Real Domain, S4ND structure, S7 State Sharing).
    단일 차원에 대한 커널을 생성합니다.
    """
    def __init__(
        self,
        d_model: int,
        N: int = 64,
        L: Optional[int] = None,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ):
        if init_mode == "hippo_n":
            raise ValueError(
                "init_mode='hippo_n' is only valid for complex-A kernels (S9/ARS9). "
                "Use 's4d_real' or 'legacy' for RS9SSMKernel."
            )
        super().__init__(d_model, N, L, dtype_idx, init_mode, discretization)

        f_dtype = get_float_dtype(dtype_idx)

        # A (diagonal) — 안정성을 위해 A < 0 이 되도록 Log 파라미터화
        if init_mode == "s4d_real":
            init_vals = torch.log((torch.arange(N, dtype=f_dtype) + 1) / 2)
            self.log_A = nn.Parameter(init_vals.unsqueeze(0).expand(d_model, -1).clone())
        else:
            self.log_A = nn.Parameter(torch.log(0.5 * torch.ones(d_model, N, dtype=f_dtype)))

        # B and C parameters (real)
        self.B = nn.Parameter(torch.randn(d_model, N, dtype=f_dtype))
        self.C = nn.Parameter(torch.randn(d_model, N, dtype=f_dtype))

    @override
    def _materialize_A(self) -> torch.Tensor:
        return -torch.exp(self.log_A)

    @override
    def _get_B(self) -> torch.Tensor:
        return self.B

    @override
    def _get_C(self) -> torch.Tensor:
        return self.C

    @override
    def _reduce_kernel(self, raw: torch.Tensor) -> torch.Tensor:
        return raw  # already real

class RS9Layer(nn.Module):
    """
    Multidimensional RS9 Layer (Generalized for D dimensions).
    spatial_dims(= D)에 따라 1D, 2D, 3D... 로 확장됩니다.
    """
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], nn.Module],
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ):
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims

        self.kernels: nn.ModuleList = nn.ModuleList([
            RS9SSMKernel(d_model, L=None, dtype_idx=dtype_idx,
                         init_mode=init_mode, discretization=discretization)
            for _ in range(spatial_dims)
        ])

        self.output_linear: nn.Linear = nn.Linear(d_model, d_model, bias=False, dtype=get_float_dtype(dtype_idx))
        self.activation = gen_activation(d_model, eps, dtype_idx)
        self.dropout: nn.Dropout = nn.Dropout(0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, C, D1, D2, ...) Real Input
        """
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        k_1d_list = [k(length=L) for k, L in zip(self.kernels, spatial_shapes)]
        k_global = build_outer_product_global_kernel(k_1d_list, spatial_shapes, u.shape[1])
        y = rfftn_convolve_nd(u, k_global, self.spatial_dims)

        # Pointwise operations (channel-last)
        permute_order = [0] + list(range(2, 2 + self.spatial_dims)) + [1]
        y = y.permute(*permute_order)

        y = self.activation(y)
        y = self.output_linear(y)
        y = self.dropout(y)

        inv_permute_order = [0, self.spatial_dims + 1] + list(range(1, 1 + self.spatial_dims))
        y = y.permute(*inv_permute_order)

        return y