"""QRS9: Quantized RS9 Layer (real domain)."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ypsilon_torch import FPDTypeIdx, get_float_dtype
from s9.rs9_modules import RS9Layer
from s9._common.kernel_base import InitMode, Discretization
from s9._common.outer_product import build_outer_product_global_kernel
from s9._common.fft_conv import rfftn_convolve_nd
from s9.quantization.bit_budget import QuantConfig
from s9.quantization.quantizers import fake_quant
from s9.quantization.stability import assert_discrete_stability
from s9.quantization.kernel_cache import QuantizedKernelCache
from s9.quantization.qrelu_real import QThASh


class QRS9Layer(nn.Module):
    """Quantized RS9 Layer — wraps RS9Layer with per-component quantization."""

    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], nn.Module],
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
        quant_config: QuantConfig = QuantConfig(),
    ):
        super().__init__()
        self.quant_config = quant_config

        self.base = RS9Layer(
            d_model=d_model,
            spatial_dims=spatial_dims,
            gen_activation=gen_activation,
            eps=eps,
            dtype_idx=dtype_idx,
            init_mode=init_mode,
            discretization=discretization,
        )

        self.activation = QThASh()
        self.kernel_cache = QuantizedKernelCache(bits=quant_config.w_bits_B or 8)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.base.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.base.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        u_q = fake_quant(u, self.quant_config.a_bits_input)

        k_1d_list = [k(length=L) for k, L in zip(self.base.kernels, spatial_shapes)]
        k_global = build_outer_product_global_kernel(k_1d_list, spatial_shapes, u.shape[1])

        if self.quant_config.enforce_stability:
            for kernel in self.base.kernels:
                dt = torch.exp(kernel.log_dt).unsqueeze(-1)
                A = kernel._materialize_A()
                A_bar = torch.exp(A * dt)
                assert_discrete_stability(
                    A_bar, self.quant_config.stability_epsilon,
                    enforce=self.quant_config.enforce_stability,
                )

        if not self.training:
            k_q, _ = self.kernel_cache.get_or_compute(k_global, tuple(spatial_shapes))
        else:
            k_q = fake_quant(k_global, self.quant_config.w_bits_B)
            self.kernel_cache.invalidate()

        y = rfftn_convolve_nd(u_q, k_q, self.base.spatial_dims)

        permute_order = [0] + list(range(2, 2 + self.base.spatial_dims)) + [1]
        y = y.permute(*permute_order)

        y = self.activation(y)
        y = fake_quant(self.base.output_linear(y), self.quant_config.w_bits_output)
        y = self.base.dropout(y)

        inv_permute_order = [0, self.base.spatial_dims + 1] + list(range(1, 1 + self.base.spatial_dims))
        y = y.permute(*inv_permute_order)

        return y
