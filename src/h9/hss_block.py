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
        Factory for the FFN activation. Default ``StableModReLU``.
    d_ff_mult : int
        FFN hidden dim multiplier. Default 4.
    init_mode : Literal["gaussian"]
        Initialization scheme. Default ``"gaussian"``.
    dropout : float
        FFN dropout. Default ``0.0``.
    eps : float
        Numerical epsilon. Default ``1e-8``.
    dtype_idx : Literal[32, 64]
        Precision selector. Default 64.
    """

    def __init__(
        self,
        d_model: int,
        n_per_axis: int,
        spatial_dims: int = 2,
        gen_activation: type[nn.Module] | None = None,
        d_ff_mult: int = 4,
        init_mode: Literal["gaussian"] = "gaussian",
        dropout: float = 0.0,
        eps: float = 1e-8,
        dtype_idx: Literal[32, 64] = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_per_axis = n_per_axis
        self.spatial_dims = spatial_dims
        self.d_prime = d_model * (n_per_axis ** spatial_dims)

        # Pre-norm layers
        self.norm_1 = ComplexLayerNorm(self.d_prime, eps=1e-5)
        self.norm_2 = ComplexLayerNorm(self.d_prime, eps=1e-5)

        # Dual projection: W_u, W_v complex (d_prime, d_prime)
        self.W_u_re = nn.Parameter(torch.empty(self.d_prime, self.d_prime))
        self.W_u_im = nn.Parameter(torch.empty(self.d_prime, self.d_prime))
        self.b_u_re = nn.Parameter(torch.empty(self.d_prime))
        self.b_u_im = nn.Parameter(torch.empty(self.d_prime))
        self.W_v_re = nn.Parameter(torch.empty(self.d_prime, self.d_prime))
        self.W_v_im = nn.Parameter(torch.empty(self.d_prime, self.d_prime))
        self.b_v_re = nn.Parameter(torch.empty(self.d_prime))
        self.b_v_im = nn.Parameter(torch.empty(self.d_prime))

        # SASS core
        self.sass = SASS(self.d_prime, eps=eps, init_mode=init_mode)

        # Output gating + projection: W_y, W_o complex (d_prime, d_prime)
        self.W_y_re = nn.Parameter(torch.empty(self.d_prime, self.d_prime))
        self.W_y_im = nn.Parameter(torch.empty(self.d_prime, self.d_prime))
        self.b_y_re = nn.Parameter(torch.empty(self.d_prime))
        self.b_y_im = nn.Parameter(torch.empty(self.d_prime))
        self.W_o_re = nn.Parameter(torch.empty(self.d_prime, self.d_prime))
        self.W_o_im = nn.Parameter(torch.empty(self.d_prime, self.d_prime))

        # FFN
        self.ffn = ComplexFFN(
            self.d_prime,
            d_ff=d_ff_mult * self.d_prime,
            gen_activation=gen_activation,
            dropout=dropout,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-like init for dual projection and output gate/proj; zero biases."""
        std = 1.0 / math.sqrt(self.d_prime)
        for p in [
            self.W_u_re, self.W_u_im, self.W_v_re, self.W_v_im,
            self.W_y_re, self.W_y_im, self.W_o_re, self.W_o_im,
        ]:
            nn.init.normal_(p, mean=0.0, std=std)
        for p in [
            self.b_u_re, self.b_u_im, self.b_v_re, self.b_v_im,
            self.b_y_re, self.b_y_im,
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
        W_u = torch.complex(self.W_u_re, self.W_u_im)
        b_u = torch.complex(self.b_u_re, self.b_u_im)
        W_v = torch.complex(self.W_v_re, self.W_v_im)
        b_v = torch.complex(self.b_v_re, self.b_v_im)
        U = torch.einsum("bchw,cd->bdhw", Z_pre, W_u) + b_u.view(1, -1, 1, 1)
        V = torch.einsum("bchw,cd->bdhw", Z_pre, W_v) + b_v.view(1, -1, 1, 1)

        sass_out = self.sass(U)

        W_y = torch.complex(self.W_y_re, self.W_y_im)
        b_y = torch.complex(self.b_y_re, self.b_y_im)
        W_o = torch.complex(self.W_o_re, self.W_o_im)
        O = (torch.einsum("bchw,cd->bdhw", sass_out, W_y) + b_y.view(1, -1, 1, 1)) * V
        Z_out = torch.einsum("bchw,cd->bdhw", O, W_o)
        Z = Z + Z_out

        # Residual 2: FFN branch
        Z = Z + self.ffn(self.norm_2(Z))

        return Z
