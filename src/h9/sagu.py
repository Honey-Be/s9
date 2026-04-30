"""DOST-Domain Spectral Adaptive Gating Unit (SAGU) — DESIGN-H9 §6.4.

Adapts HAMSA's SAGU to the DOST coefficient domain with 2D spatial structure.
All mixing is channel-only; spatial dims are broadcast.

Mathematical specification::

    SAGU(U_tilde) = (U_tilde W_1) * gate_act(|U_tilde| W_2 + b_2)

where ``W_1 in C^{D' x D'}`` (complex linear branch),
``W_2 in R^{D' x D'}`` (real, from magnitudes), ``b_2 in R^{D'}``,
and ``gate_act`` is a configurable real-valued activation (default sigmoid).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from typing import Callable
from ypsilon_torch import FPDTypeIdx, get_complex_dtype, get_float_dtype

class DOSTDomainSAGU(nn.Module):
    """SAGU adapted for DOST coefficient domain.

    Parameters
    ----------
    d_prime : int
        Channel dimension after Warped DOST.
    gen_gate_activation : type[nn.Module] | None
        Factory for the magnitude-gate activation. Must accept real input and
        produce real output. Default ``nn.Sigmoid``.
    eps : float
        Numerical epsilon. Default ``1e-8``.

    Shape contract
    --------------
    Input  : ``U_tilde`` of shape ``(B, d_prime, H, W)`` complex.
    Output : same shape, complex.

    Notes
    -----
    The gate activation receives **real-valued** input (post-linear magnitudes)
    and must return real output. Complex-valued activations must NOT be used here.
    """

    def __init__(
        self,
        d_prime: int,
        gen_gate_activation: Callable[[], nn.Module] = nn.Sigmoid,
        eps: float = 1e-8,
        dtype_idx: FPDTypeIdx = 64
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.eps = eps

        self.gate_activation = gen_gate_activation()

        c_dtype = get_complex_dtype(dtype_idx)
        r_dtype = get_float_dtype(dtype_idx)

        # Complex linear branch (separate real/imag)
        self.W_1 = nn.Parameter(torch.empty(d_prime, d_prime, dtype=c_dtype))
        # Real magnitude gate
        self.W_2 = nn.Parameter(torch.empty(d_prime, d_prime, dtype=r_dtype))
        self.b_2 = nn.Parameter(torch.empty(d_prime, dtype=r_dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-like init."""
        std = 1.0 / math.sqrt(self.d_prime)
        nn.init.normal_(self.W_1.real, mean=0.0, std=std)
        nn.init.normal_(self.W_1.imag, mean=0.0, std=std)
        nn.init.normal_(self.W_2, mean=0.0, std=std)
        nn.init.zeros_(self.b_2)

    def forward(self, U_tilde: Tensor) -> Tensor:
        """Apply DOST-domain SAGU.

        Parameters
        ----------
        U_tilde : Tensor
            Complex input of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            Same shape, complex.
        """
        # Complex linear branch
        W_1 = self.W_1
        linear_out = torch.einsum("bchw,cd->bdhw", U_tilde, W_1)

        # Magnitude gate (real)
        m = U_tilde.abs()
        gate = self.gate_activation(
            torch.einsum("bchw,cd->bdhw", m, self.W_2)
            + self.b_2.view(1, -1, 1, 1)
        )

        return linear_out * gate
