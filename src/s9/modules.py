import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Literal, Optional, Tuple

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override

from ypsilon_torch import get_complex_dtype, get_float_dtype, FPDTypeIdx
from s9._common.kernel_base import DiagonalSSMKernelBase, InitMode, Discretization
from s9._common.outer_product import build_outer_product_global_kernel
from s9._common.fft_conv import fftn_convolve_nd

from ypsilon_torch.blocks.activations.complex import StableComplexCardioid, StableModReLU
from ypsilon_torch.blocks.regularizations.complex import ComplexDropout

class S9SSMKernel(DiagonalSSMKernelBase):
    """
    The Core S9 Kernel (Complex Domain, S4ND structure, S7 State Sharing).
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
        if init_mode == "s4d_real":
            raise ValueError(
                "init_mode='s4d_real' is only valid for RS9SSMKernel (real-valued A). "
                "Use 'hippo_n' or 'legacy' for S9SSMKernel."
            )
        super().__init__(d_model, N, L, dtype_idx, init_mode, discretization)

        c_dtype = get_complex_dtype(dtype_idx)
        f_dtype = get_float_dtype(dtype_idx)

        # Real and Imag parts of A (diagonal)
        # 안정성을 위해 Real(A) < 0 이 되도록 Log 파라미터화
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, N, dtype=f_dtype)))
        if init_mode == "hippo_n":
            self.A_imag = nn.Parameter(torch.pi * (torch.arange(N).to(f_dtype) + 0.5))
        else:
            self.A_imag = nn.Parameter(torch.pi * torch.arange(N).to(f_dtype) / N)

        # B and C parameters (Complex)
        if init_mode == "hippo_n":
            B_init = torch.sqrt(2 * torch.arange(N).to(f_dtype) + 1).unsqueeze(0).expand(d_model, -1)
            self.B = nn.Parameter(torch.complex(B_init, torch.zeros_like(B_init)))
        else:
            self.B = nn.Parameter(torch.randn(d_model, N, dtype=c_dtype))
        self.C = nn.Parameter(torch.randn(d_model, N, dtype=c_dtype))

    @override
    def _materialize_A(self) -> torch.Tensor:
        return -torch.exp(self.log_A_real) + 1j * self.A_imag

    @override
    def _get_B(self) -> torch.Tensor:
        return self.B

    @override
    def _get_C(self) -> torch.Tensor:
        return self.C

    @override
    def _reduce_kernel(self, raw: torch.Tensor) -> torch.Tensor:
        return raw  # complex output as-is

class S9Layer(nn.Module):
    """
    Multidimensional S9 Layer (Generalized for D dimensions).
    spatial_dims(= D)에 따라 1D, 2D, 3D... 로 확장됩니다.
    """
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase],
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ):
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims

        self.kernels: nn.ModuleList = nn.ModuleList([
            S9SSMKernel(d_model, L=None, dtype_idx=dtype_idx,
                        init_mode=init_mode, discretization=discretization)
            for _ in range(spatial_dims)
        ])

        self.output_linear: nn.Linear = nn.Linear(d_model, d_model, bias=False, dtype=get_complex_dtype(dtype_idx))
        self.activation = gen_activation(d_model, eps, dtype_idx)
        self.dropout: ComplexDropout = ComplexDropout(0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, C, D1, D2, ...) Complex Input
        """
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        k_1d_list = [k(length=L) for k, L in zip(self.kernels, spatial_shapes)]
        k_global = build_outer_product_global_kernel(k_1d_list, spatial_shapes, u.shape[1])
        y = fftn_convolve_nd(u, k_global, self.spatial_dims)

        # Pointwise operations (channel-last)
        permute_order = [0] + list(range(2, 2 + self.spatial_dims)) + [1]
        y = y.permute(*permute_order)

        y = self.activation(y)
        ydtype = y.dtype
        if ydtype == get_complex_dtype(32):
            with torch.amp.autocast(device_type=str(y.device)):
                y = self.output_linear.to(dtype=torch.complex64)(y.to(dtype=torch.complex64)).to(dtype=ydtype)
        else:
            y = self.output_linear(y)
        y = self.dropout(y)

        inv_permute_order = [0, self.spatial_dims + 1] + list(range(1, 1 + self.spatial_dims))
        y = y.permute(*inv_permute_order)

        return y