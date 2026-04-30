"""HSS Block: full transformer-like block with SASS core — DESIGN-H9 §5.

Each HSS block applies:
1. ComplexLayerNorm (pre-norm)
2. Dual projection: (U, V) = (Z @ W_u, Z @ W_v)
3. SASS on U
4. Output gating: O = (SASS_out @ W_y + b_y) * V
5. Output projection: Z_out = O @ W_o
6. Residual add
7. ComplexLayerNorm + ComplexFFN + residual

Shape contract: ``(B, d_prime, H, W)`` complex in and out.
No spatial mixing — all weights are channel-only.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from h9.components import ComplexFFN, ComplexLayerNorm
from h9.sass import SASS

from ypsilon_torch import FPDTypeIdx, get_complex_dtype, get_float_dtype
from ypsilon_torch.blocks.activations.complex import StableModReLU

class HSSBlock(nn.Module):
    """One HSS block (residual SASS + residual FFN).

    Parameters
    ----------
    d_model : int
        Pre-DOST channel dimension. Block operates on
        ``d_prime = d_model * (n_per_axis ** spatial_dims)``.
    n_per_axis : int
        Warped DOST band count per spatial axis.
    spatial_dims : int
        Spatial dimensionality. Default 2.
    gen_activation : type[nn.Module] | None
        Factory for the FFN activation (complex → complex).
        Default ``StableModReLU``.
    gen_gate_activation : type[nn.Module] | None
        Factory for the SAGU magnitude-gate activation (real → real).
        Default ``nn.Sigmoid``.
    d_ff_mult : int
        FFN hidden dim multiplier. Default 4.
    init_mode : Literal["gaussian"]
        Initialization scheme. Default ``"gaussian"``.
    dropout : float
        FFN dropout. Default ``0.0``.
    eps : float
        Numerical epsilon. Default ``1e-8``.
    dtype_idx : Literal[32, 64, 128]
        Precision selector. Default 64.
    """

    def __init__(
        self,
        d_model: int,
        n_per_axis: int,
        spatial_dims: int = 2,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase] = StableModReLU,
        gen_gate_activation: Callable[[], nn.Module] = nn.Sigmoid,
        d_ff_mult: int = 4,
        init_mode: Literal["gaussian"] = "gaussian",
        dropout: float = 0.0,
        eps: float = 1e-8,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_per_axis = n_per_axis
        self.spatial_dims = spatial_dims
        self.d_prime = d_model * (n_per_axis ** spatial_dims)

        # Pre-norm layers
        self.norm_1 = ComplexLayerNorm(self.d_prime, eps=1e-5, dtype_idx=dtype_idx)
        self.norm_2 = ComplexLayerNorm(self.d_prime, eps=1e-5, dtype_idx=dtype_idx)

        # Dual projection: W_u, W_v complex (d_prime, d_prime)
        self.W_u = nn.Parameter(torch.empty(self.d_prime, self.d_prime, dtype=get_complex_dtype(dtype_idx)))
        self.b_u = nn.Parameter(torch.empty(self.d_prime, dtype=get_complex_dtype(dtype_idx)))
        self.W_v = nn.Parameter(torch.empty(self.d_prime, self.d_prime, dtype=get_complex_dtype(dtype_idx)))
        self.b_v = nn.Parameter(torch.empty(self.d_prime, dtype=get_complex_dtype(dtype_idx)))

        # SASS core
        self.sass = SASS(
            self.d_prime,
            gen_gate_activation=gen_gate_activation,
            eps=eps,
            init_mode=init_mode,
            dtype_idx=dtype_idx
        )

        # Output gating + projection: W_y, W_o complex (d_prime, d_prime)
        self.W_y = nn.Parameter(torch.empty(self.d_prime, self.d_prime, dtype=get_complex_dtype(dtype_idx)))
        self.b_y = nn.Parameter(torch.empty(self.d_prime, dtype=get_complex_dtype(dtype_idx)))
        self.W_o = nn.Parameter(torch.empty(self.d_prime, self.d_prime, dtype=get_complex_dtype(dtype_idx)))

        # FFN
        self.ffn = ComplexFFN(
            self.d_prime,
            d_ff=d_ff_mult * self.d_prime,
            gen_activation=gen_activation,
            dropout=dropout,
            dtype_idx=dtype_idx
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-like init for dual projection and output gate/proj; zero biases."""
        std = 1.0 / math.sqrt(self.d_prime)
        for p in [
            self.W_u, self.W_v, self.W_y, self.W_o
        ]:
            nn.init.normal_(p.real, mean=0.0, std=std)
            nn.init.normal_(p.imag, mean=0.0, std=std)
        for p in [
            self.b_u, self.b_v, self.b_y
        ]:
            nn.init.zeros_(p)

    def forward(self, Z: Tensor) -> Tensor:
        """Apply one HSS block with residual connections.

        Parameters
        ----------
        Z : Tensor
            Complex tensor of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            Same shape, complex.
        """
        # Residual 1: SASS branch
        Z_pre = self.norm_1(Z)
        U = torch.einsum("bchw,cd->bdhw", Z_pre, self.W_u) + self.b_u.view(1, -1, 1, 1)
        V = torch.einsum("bchw,cd->bdhw", Z_pre, self.W_v) + self.b_v.view(1, -1, 1, 1)

        sass_out = self.sass(U)

        O = (torch.einsum("bchw,cd->bdhw", sass_out, self.W_y) + self.b_y.view(1, -1, 1, 1)) * V
        Z_out = torch.einsum("bchw,cd->bdhw", O, self.W_o)
        Z = Z + Z_out

        # Residual 2: FFN branch
        Z = Z + self.ffn(self.norm_2(Z))

        return Z
