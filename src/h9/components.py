"""Shared utility modules: ComplexLayerNorm, ComplexFFN.

See ``DESIGN-H9.md`` §5.3 (ComplexLayerNorm) and §5.4 (ComplexFFN).

No equivalent ComplexLayerNorm exists in ``s9.modules``; this is a
self-contained implementation for h9.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ComplexLayerNorm(nn.Module):
    """LayerNorm for complex tensors, normalizing by magnitude statistics.

    Operates over the channel dimension. Computes mean and variance from
    ``|Z|`` (real-valued), then divides the complex ``Z`` by
    ``sqrt(var + eps)``. Applies learnable complex affine ``gamma * Z + beta``.

    Parameters
    ----------
    d_prime : int
        Channel dimension.
    eps : float
        Stability epsilon for variance. Default ``1e-5``.
    affine : bool
        If True, apply learnable ``gamma`` and ``beta``. Default True.

    Shape contract
    --------------
    Input  : ``(B, d_prime, H, W)`` complex.
    Output : same shape, complex.
    """

    def __init__(
        self,
        d_prime: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma_re = nn.Parameter(torch.empty(d_prime))
            self.gamma_im = nn.Parameter(torch.empty(d_prime))
            self.beta_re = nn.Parameter(torch.empty(d_prime))
            self.beta_im = nn.Parameter(torch.empty(d_prime))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize affine parameters: gamma=1, beta=0 (both complex)."""
        if self.affine:
            nn.init.ones_(self.gamma_re)
            nn.init.zeros_(self.gamma_im)
            nn.init.zeros_(self.beta_re)
            nn.init.zeros_(self.beta_im)

    def forward(self, Z: Tensor) -> Tensor:
        """Normalize and apply affine.

        Parameters
        ----------
        Z : Tensor
            Complex tensor of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            Same shape, complex.
        """
        m = Z.abs()  # real, (B, d_prime, H, W)
        mean = m.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = ((m - mean) ** 2).mean(dim=1, keepdim=True)
        scale = (var + self.eps).rsqrt()  # (B, 1, H, W) real
        Z_norm = Z * scale  # complex * real -> complex
        if self.affine:
            gamma = torch.complex(self.gamma_re, self.gamma_im).view(1, -1, 1, 1)
            beta = torch.complex(self.beta_re, self.beta_im).view(1, -1, 1, 1)
            return gamma * Z_norm + beta
        return Z_norm


class ComplexFFN(nn.Module):
    """Two-layer complex MLP, channel-only (DESIGN-H9 §5.4).

    Parameters
    ----------
    d_prime : int
        Channel dimension (input and output).
    d_ff : int | None
        Hidden dimension. Default ``4 * d_prime``.
    gen_activation : type[nn.Module] | None
        Factory for the inter-layer activation (s9 convention).
        Default: ``s9.activations.complex.StableModReLU``.
    dropout : float
        Dropout probability applied after activation. Default ``0.0``.

    Shape contract
    --------------
    Input  : ``(B, d_prime, H, W)`` complex.
    Output : same shape, complex.
    """

    def __init__(
        self,
        d_prime: int,
        d_ff: int | None = None,
        gen_activation: type[nn.Module] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.d_ff = d_ff if d_ff is not None else 4 * d_prime

        if gen_activation is None:
            from s9.activations.complex.stable_modrelu import StableModReLU
            gen_activation = StableModReLU

        # Complex linear layers via separate real/imag Parameters
        self.W_1_re = nn.Parameter(torch.empty(self.d_ff, d_prime))
        self.W_1_im = nn.Parameter(torch.empty(self.d_ff, d_prime))
        self.b_1_re = nn.Parameter(torch.empty(self.d_ff))
        self.b_1_im = nn.Parameter(torch.empty(self.d_ff))
        self.W_2_re = nn.Parameter(torch.empty(d_prime, self.d_ff))
        self.W_2_im = nn.Parameter(torch.empty(d_prime, self.d_ff))
        self.b_2_re = nn.Parameter(torch.empty(d_prime))
        self.b_2_im = nn.Parameter(torch.empty(d_prime))

        # StableModReLU expects (features, eps, dtype_idx)
        self.activation = gen_activation(self.d_ff)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-like init for both linear layers."""
        std1 = 1.0 / math.sqrt(self.d_prime)
        std2 = 1.0 / math.sqrt(self.d_ff)
        nn.init.normal_(self.W_1_re, mean=0.0, std=std1)
        nn.init.normal_(self.W_1_im, mean=0.0, std=std1)
        nn.init.zeros_(self.b_1_re)
        nn.init.zeros_(self.b_1_im)
        nn.init.normal_(self.W_2_re, mean=0.0, std=std2)
        nn.init.normal_(self.W_2_im, mean=0.0, std=std2)
        nn.init.zeros_(self.b_2_re)
        nn.init.zeros_(self.b_2_im)

    def forward(self, Z: Tensor) -> Tensor:
        """Apply two-layer complex FFN.

        Parameters
        ----------
        Z : Tensor
            Complex tensor of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            Same shape, complex.
        """
        W_1 = torch.complex(self.W_1_re, self.W_1_im)
        b_1 = torch.complex(self.b_1_re, self.b_1_im)
        W_2 = torch.complex(self.W_2_re, self.W_2_im)
        b_2 = torch.complex(self.b_2_re, self.b_2_im)
        h = torch.einsum("bchw,fc->bfhw", Z, W_1) + b_1.view(1, -1, 1, 1)
        # StableModReLU expects channel-last (B, ..., C); h is (B, C, H, W)
        h = self.activation(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.dropout(h)
        out = torch.einsum("bfhw,cf->bchw", h, W_2) + b_2.view(1, -1, 1, 1)
        return out
