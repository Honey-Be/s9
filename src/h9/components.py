"""Shared utility modules: ComplexLayerNorm, ComplexFFN.

See ``DESIGN-H9.md`` §5.3 (ComplexLayerNorm) and §5.4 (ComplexFFN).

No equivalent ComplexLayerNorm exists in ``s9.modules``; this is a
self-contained implementation for h9.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from ypsilon_torch import get_complex_dtype, FPDTypeIdx
from ypsilon_torch.blocks.activations import ComplexActivationFunctionBase
from ypsilon_torch.blocks.activations.complex import StableModReLU

from typing import Callable


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
        dtype_idx: FPDTypeIdx = 64
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.empty(d_prime, dtype=get_complex_dtype(dtype_idx)))
            self.beta = nn.Parameter(torch.empty(d_prime, dtype=get_complex_dtype(dtype_idx)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize affine parameters: gamma=1, beta=0 (both complex)."""
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

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
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
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
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase] = StableModReLU,
        dropout: float = 0.0,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.d_ff = d_ff if d_ff is not None else 4 * d_prime

        # Complex linear layers via separate real/imag Parameters
        self.W_1 = nn.Parameter(torch.empty(self.d_ff, d_prime, dtype=get_complex_dtype(dtype_idx)))
        self.b_1 = nn.Parameter(torch.empty(self.d_ff, dtype=get_complex_dtype(dtype_idx)))
        self.W_2 = nn.Parameter(torch.empty(d_prime, self.d_ff, dtype=get_complex_dtype(dtype_idx)))
        self.b_2 = nn.Parameter(torch.empty(d_prime, dtype=get_complex_dtype(dtype_idx)))

        # StableModReLU expects (features, eps, dtype_idx)
        self.activation = gen_activation(self.d_ff, eps, dtype_idx)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-like init for both linear layers."""
        std1 = 1.0 / math.sqrt(self.d_prime)
        std2 = 1.0 / math.sqrt(self.d_ff)
        nn.init.normal_(self.W_1.real, mean=0.0, std=std1)
        nn.init.normal_(self.W_1.imag, mean=0.0, std=std1)
        nn.init.zeros_(self.b_1)
        nn.init.normal_(self.W_2.real, mean=0.0, std=std2)
        nn.init.normal_(self.W_2.imag, mean=0.0, std=std2)
        nn.init.zeros_(self.b_2)

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
        h = torch.einsum("bchw,fc->bfhw", Z, self.W_1) + self.b_1.view(1, -1, 1, 1)
        # StableModReLU expects channel-last (B, ..., C); h is (B, C, H, W)
        h = self.activation(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.dropout(h)
        out = torch.einsum("bfhw,cf->bchw", h, self.W_2) + self.b_2.view(1, -1, 1, 1)
        return out
