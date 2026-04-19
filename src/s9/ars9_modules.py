"""ARS9 (Advanced RS9): Conjugate-pair complex internal state, real I/O.

The ARS9 kernel stores N/2 complex conjugate pairs of eigenvalues.
Each pair contributes ``2·Re(C_n · B_bar_n · A_bar_n^t)`` to the kernel,
guaranteeing a real-valued output without requiring DOST preprocessing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from typing import override, Callable
except Exception:  # pragma: no cover
    from typing_extensions import override, Callable  # type: ignore

from s9.base import get_float_dtype, FPDTypeIdx
from s9._common.kernel_base import DiagonalSSMKernelBase, InitMode, Discretization
from s9._common.outer_product import build_outer_product_global_kernel
from s9._common.fft_conv import rfftn_convolve_nd


class ARS9SSMKernel(DiagonalSSMKernelBase):
    """ARS9 SSM Kernel — conjugate-pair complex diagonal A, real output.

    Internal state dimension is N (must be even). Parameters are stored for
    N/2 pairs; the conjugate half is implicit.
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
        if N % 2 != 0:
            raise ValueError(f"ARS9SSMKernel requires even N, got {N}")
        if init_mode == "s4d_real":
            raise ValueError(
                "init_mode='s4d_real' is only valid for RS9SSMKernel (real-valued A). "
                "Use 'hippo_n' or 'legacy' for ARS9SSMKernel."
            )
        super().__init__(d_model, N, L, dtype_idx, init_mode, discretization)

        half_N = N // 2
        f_dtype = get_float_dtype(dtype_idx)

        # A diagonal (half-pair): Re(A) < 0 via log parameterisation
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(d_model, half_N, dtype=f_dtype))
        )
        if init_mode == "hippo_n":
            self.A_imag = nn.Parameter(
                torch.pi * (torch.arange(half_N, dtype=f_dtype) + 0.5)
            )
        else:
            self.A_imag = nn.Parameter(
                torch.pi * torch.arange(half_N, dtype=f_dtype) / half_N
            )

        # B (complex, half-pair)
        if init_mode == "hippo_n":
            B_real_init = torch.sqrt(2 * torch.arange(half_N, dtype=f_dtype) + 1).unsqueeze(0).expand(d_model, -1)
            self.B_real = nn.Parameter(B_real_init.clone())
            self.B_imag = nn.Parameter(torch.zeros(d_model, half_N, dtype=f_dtype))
        else:
            self.B_real = nn.Parameter(torch.randn(d_model, half_N, dtype=f_dtype))
            self.B_imag = nn.Parameter(torch.randn(d_model, half_N, dtype=f_dtype))

        # C (complex, half-pair)
        self.C_real = nn.Parameter(torch.randn(d_model, half_N, dtype=f_dtype))
        self.C_imag = nn.Parameter(torch.randn(d_model, half_N, dtype=f_dtype))

    @override
    def _materialize_A(self) -> torch.Tensor:
        return -torch.exp(self.log_A_real) + 1j * self.A_imag

    @override
    def _get_B(self) -> torch.Tensor:
        return torch.complex(self.B_real, self.B_imag)

    @override
    def _get_C(self) -> torch.Tensor:
        return torch.complex(self.C_real, self.C_imag)

    @override
    def _reduce_kernel(self, raw: torch.Tensor) -> torch.Tensor:
        # Conjugate pair: contribution = 2·Re(sum)
        return 2 * raw.real


class ARS9Layer(nn.Module):
    """Multidimensional ARS9 Layer — real I/O, conjugate-pair complex internal dynamics.

    Drop-in replacement for RS9Layer with richer (oscillatory) expressiveness.
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
            ARS9SSMKernel(d_model, L=None, dtype_idx=dtype_idx,
                          init_mode=init_mode, discretization=discretization)
            for _ in range(spatial_dims)
        ])

        self.output_linear: nn.Linear = nn.Linear(
            d_model, d_model, bias=False, dtype=get_float_dtype(dtype_idx)
        )
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
