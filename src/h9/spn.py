"""Phase-Aware SpectralPulseNet (SPN) — DESIGN-H9 §6.1.

Core innovation of h9: input-dependent gating in the DOST coefficient domain
that uses BOTH magnitude AND phase information, with sin/cos decomposition to
avoid wrap-around discontinuities at the ``-pi``-to-``pi`` boundary.

Mathematical specification::

    m       = |U|
    cos_t   = Re(U) / (m + eps)
    sin_t   = Im(U) / (m + eps)
    g       = sigmoid(W_m m + W_p cos_t + W_p' sin_t + b_g)

All matmuls are channel-only (einsum ``"bchw,cd->bdhw"``); the gate ``g``
is real-valued in ``[0, 1]^{B x D' x H x W}``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


class PhaseAwareSPN(nn.Module):
    """Phase-aware SpectralPulseNet gate generator.

    Parameters
    ----------
    d_prime : int
        Channel dimension after Warped DOST.
    eps : float
        Stability epsilon for magnitude denominator. Default ``1e-8``.
    init_mode : str
        Initialization scheme. Default ``"gaussian"``.

    Shape contract
    --------------
    Input  : ``U`` of shape ``(B, d_prime, H, W)`` complex.
    Output : ``g`` of shape ``(B, d_prime, H, W)`` real in ``[0, 1]``.
    """

    def __init__(
        self,
        d_prime: int,
        eps: float = 1e-8,
        init_mode: Literal["gaussian"] = "gaussian",
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.eps = eps
        self.init_mode = init_mode

        self.W_m = nn.Parameter(torch.empty(d_prime, d_prime))
        self.W_p = nn.Parameter(torch.empty(d_prime, d_prime))
        self.W_p_prime = nn.Parameter(torch.empty(d_prime, d_prime))
        self.b_g = nn.Parameter(torch.empty(d_prime))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-like Gaussian init; bias zero."""
        std = 1.0 / math.sqrt(self.d_prime)
        nn.init.normal_(self.W_m, mean=0.0, std=std)
        nn.init.normal_(self.W_p, mean=0.0, std=std)
        nn.init.normal_(self.W_p_prime, mean=0.0, std=std)
        nn.init.zeros_(self.b_g)

    def forward(self, U: Tensor) -> Tensor:
        """Compute the phase-aware gate.

        Parameters
        ----------
        U : Tensor
            Complex input of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            Real-valued gate ``g`` of shape ``(B, d_prime, H, W)`` in ``[0, 1]``.
        """
        m = U.abs()  # real, (B, D', H, W)
        m_safe = m.clamp(min=self.eps)
        cos_t = U.real / m_safe
        sin_t = U.imag / m_safe

        # Channel-only mixing via einsum, broadcast over (H, W)
        out = (
            torch.einsum("bchw,cd->bdhw", m, self.W_m)
            + torch.einsum("bchw,cd->bdhw", cos_t, self.W_p)
            + torch.einsum("bchw,cd->bdhw", sin_t, self.W_p_prime)
            + self.b_g.view(1, -1, 1, 1)
        )
        return torch.sigmoid(out)
