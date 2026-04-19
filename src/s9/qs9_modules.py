"""QS9: Quantized S9 Layer (complex domain)."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from s9.base import ComplexActivationFunctionBase, FPDTypeIdx, get_complex_dtype
from s9.modules import S9Layer
from s9._common.kernel_base import InitMode, Discretization
from s9._common.outer_product import build_outer_product_global_kernel
from s9._common.fft_conv import fftn_convolve_nd
from s9.quantization.bit_budget import QuantConfig
from s9.quantization.quantizers import fake_quant
from s9.quantization.stability import assert_discrete_stability
from s9.quantization.kernel_cache import QuantizedKernelCache
from s9.quantization.polar_modrelu import PolarStableModReLU


class QS9Layer(nn.Module):
    """Quantized S9 Layer — wraps S9Layer with per-component quantization."""

    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase],
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
        quant_config: QuantConfig = QuantConfig(),
    ):
        super().__init__()
        self.quant_config = quant_config

        # Build the underlying S9Layer
        self.base = S9Layer(
            d_model=d_model,
            spatial_dims=spatial_dims,
            gen_activation=gen_activation,
            eps=eps,
            dtype_idx=dtype_idx,
            init_mode=init_mode,
            discretization=discretization,
        )

        # Replace activation with polar-quantized variant
        self.activation = PolarStableModReLU(d_model, eps, dtype_idx)
        self.kernel_cache = QuantizedKernelCache(bits=quant_config.w_bits_B or 8)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.base.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.base.spatial_dims} spatial dims, "
                f"got {len(spatial_shapes)}"
            )

        # Quantize input
        u_q = fake_quant(u, self.quant_config.a_bits_input, is_complex=True)

        # Generate kernel (fp32)
        k_1d_list = [k(length=L) for k, L in zip(self.base.kernels, spatial_shapes)]
        k_global = build_outer_product_global_kernel(k_1d_list, spatial_shapes, u.shape[1])

        # Stability check
        if self.quant_config.enforce_stability:
            for kernel in self.base.kernels:
                dt = torch.exp(kernel.log_dt).unsqueeze(-1)
                A = kernel._materialize_A()
                A_bar = torch.exp(A * dt)
                assert_discrete_stability(
                    A_bar, self.quant_config.stability_epsilon,
                    enforce=self.quant_config.enforce_stability,
                )

        # Cache quantized kernel in eval mode
        if not self.training:
            k_q, _ = self.kernel_cache.get_or_compute(k_global, tuple(spatial_shapes))
        else:
            k_q = fake_quant(k_global, self.quant_config.w_bits_B, is_complex=True)
            self.kernel_cache.invalidate()

        # FFT convolution
        y = fftn_convolve_nd(u_q, k_q, self.base.spatial_dims)

        # Pointwise with quantized activation
        permute_order = [0] + list(range(2, 2 + self.base.spatial_dims)) + [1]
        y = y.permute(*permute_order)

        y = self.activation(y)

        ydtype = y.dtype
        if ydtype == get_complex_dtype(32):
            with torch.amp.autocast(device_type=str(y.device)):
                y = self.base.output_linear.to(dtype=torch.complex64)(
                    y.to(dtype=torch.complex64)
                ).to(dtype=ydtype)
        else:
            y = fake_quant(
                self.base.output_linear(y),
                self.quant_config.w_bits_output,
                is_complex=True,
            )
        y = self.base.dropout(y)

        inv_permute_order = [0, self.base.spatial_dims + 1] + list(range(1, 1 + self.base.spatial_dims))
        y = y.permute(*inv_permute_order)

        return y
