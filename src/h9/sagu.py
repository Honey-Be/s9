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
        gen_gate_activation: type[nn.Module] | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.eps = eps

        if gen_gate_activation is None:
            gen_gate_activation = nn.Sigmoid
        self.gate_activation = gen_gate_activation()

        # Complex linear branch (separate real/imag)
        self.W_1_re = nn.Parameter(torch.empty(d_prime, d_prime))
        self.W_1_im = nn.Parameter(torch.empty(d_prime, d_prime))
        # Real magnitude gate
        self.W_2 = nn.Parameter(torch.empty(d_prime, d_prime))
        self.b_2 = nn.Parameter(torch.empty(d_prime))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-like init."""
        std = 1.0 / math.sqrt(self.d_prime)
        nn.init.normal_(self.W_1_re, mean=0.0, std=std)
        nn.init.normal_(self.W_1_im, mean=0.0, std=std)
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
        W_1 = torch.complex(self.W_1_re, self.W_1_im)
        linear_out = torch.einsum("bchw,cd->bdhw", U_tilde, W_1)

        # Magnitude gate (real)
        m = U_tilde.abs()
        gate = self.gate_activation(
            torch.einsum("bchw,cd->bdhw", m, self.W_2)
            + self.b_2.view(1, -1, 1, 1)
        )

        return linear_out * gate
