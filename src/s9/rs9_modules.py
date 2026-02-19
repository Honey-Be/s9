from __future__ import annotations

import math
import itertools
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, final, Optional

import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override, Tuple, Callable
except Exception:  # pragma: no cover
    from typing_extensions import override, Tuple, Callable  # type: ignore

from s9.base import get_float_dtype, FPDTypeIdx

class RS9SSMKernel(nn.Module):
    """
    The Core RS9 Kernel (Real Domain, S4ND structure, S7 State Sharing).
    단일 차원에 대한 커널을 생성합니다.
    """
    def __init__(self, d_model: int, N: int = 64, L: Optional[int] = None, dtype_idx: FPDTypeIdx = 64):
        super().__init__()
        self.d_model = d_model
        self.N = N # State size
        self.L = L # Max sequence length for this dimension

        f_dtype = get_float_dtype(dtype_idx)

        # A (diagonal)
        # 안정성을 위해 A < 0 이 되도록 Log 파라미터화
        self.log_A = nn.Parameter(torch.log(0.5 * torch.ones(d_model, N, dtype=f_dtype)))
        
        # B and C parameters (real)
        self.B = nn.Parameter(torch.randn(d_model, N, dtype=f_dtype))
        self.C = nn.Parameter(torch.randn(d_model, N, dtype=f_dtype))
        
        # Delta (Step size)
        self.log_dt = nn.Parameter(torch.rand(d_model, dtype=f_dtype) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))

    def forward(self, length: int) -> torch.Tensor:
        # 1. 파라미터 구체화
        dt = torch.exp(self.log_dt) # (d_model)
        A = -torch.exp(self.log_A) # (d_model, N)
        
        dt = dt.unsqueeze(-1)
        
        # 2. 이산화 (ZOH approximation)
        A_bar = torch.exp(A * dt) # (d_model, N)
        B_bar = self.B * dt # (d_model, N)
        
        # 3. SSM 커널 계산 (Vandermonde style / Power series)
        range_t = torch.arange(length, device=A.device)
        # (d_model, 1, N) ** (1, L, 1) -> (d_model, L, N)
        A_powers = A_bar.unsqueeze(1) ** range_t.unsqueeze(0).unsqueeze(2) 
        
        # K = Sum over state dimension N
        term = self.C.unsqueeze(1) * B_bar.unsqueeze(1) * A_powers
        K = torch.sum(term, dim=-1) # (d_model, L)
        
        return K

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
    ):
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims
        
        # 각 차원(Dimension)별로 독립적인 RS9 커널 생성
        # 예: 2D 이미지면 [kernel_H, kernel_W]
        self.kernels: nn.ModuleList = nn.ModuleList([
            RS9SSMKernel(d_model, L=None, dtype_idx=dtype_idx)
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
        B = u.shape[0]
        C = u.shape[1]
        spatial_shapes = u.shape[2:] # (D1, D2, ...)
        
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, got {len(spatial_shapes)}")

        # 1. 차원별 1D 커널 생성
        # k_1d_list[i] shape: (C, L_i)
        k_1d_list = [k(length=L) for k, L in zip(self.kernels, spatial_shapes)]
        
        # 2. D차원 Global Kernel 구성 (Outer Product)
        # K_global = k_0 (x) k_1 (x) ... (x) k_{D-1}
        # Broadcasting을 이용해 계산: (C, L0, 1, ..., 1) * (C, 1, L1, 1, ...) ...
        
        k_global = k_1d_list[0] # (C, L0)
        # 첫 번째 차원에 맞게 reshape: (C, L0, 1, 1, ...)
        view_shape = [C] + [1] * self.spatial_dims
        view_shape[1] = spatial_shapes[0]
        k_global = k_global.view(*view_shape)
        
        for i in range(1, self.spatial_dims):
            k_next = k_1d_list[i] # (C, Li)
            
            # 다음 차원을 위한 view shape 생성
            # 예: i=1 (두번째 차원) -> (C, 1, L1, 1, ...)
            next_view_shape = [C] + [1] * self.spatial_dims
            next_view_shape[i+1] = spatial_shapes[i]
            
            k_global = k_global * k_next.view(*next_view_shape)
            
        # k_global shape: (C, D1, D2, ...)
        
        # 3. Multidimensional FFT Convolution
        # Linear convolution을 위한 패딩 계산 (size * 2)
        padded_shapes = [s * 2 for s in spatial_shapes]
        
        # FFT 수행할 차원들 (뒤에서 D개)
        fft_dims = tuple(range(-self.spatial_dims, 0))
        
        # FFT
        u_f = torch.fft.rfftn(u, s=padded_shapes, dim=fft_dims)
        k_f = torch.fft.rfftn(k_global, s=padded_shapes, dim=fft_dims)
        
        # Convolution in Frequency Domain
        y_f = u_f * k_f 
        
        # IFFT
        y = torch.fft.irfftn(y_f, s=padded_shapes, dim=fft_dims)
        
        # Crop (원래 크기로 자르기)
        # 슬라이스 객체 생성: [..., :D1, :D2, ...]
        slices = [slice(None)] * 2 + [slice(0, s) for s in spatial_shapes]
        y = y[tuple(slices)]
        
        # 4. Pointwise Operations
        # Linear/Activation을 위해 채널을 마지막으로 이동: (B, D1, ..., C)
        permute_order = [0] + list(range(2, 2 + self.spatial_dims)) + [1]
        y = y.permute(*permute_order)
        
        y = self.activation(y)
        y = self.output_linear(y)
        y = self.dropout(y)
        
        # 다시 채널을 두 번째로 이동: (B, C, D1, ...)
        inv_permute_order = [0, self.spatial_dims + 1] + list(range(1, 1 + self.spatial_dims))
        y = y.permute(*inv_permute_order)
        
        return y