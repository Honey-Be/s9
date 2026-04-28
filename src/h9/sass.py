"""SASS: Spectral Adaptive State Space — DESIGN-H9 §6.

Composition of the three core spectral innovations::

    g = SPN(U)              # real, (B, D', H, W) in [0, 1]
    K = SpectralKernel       # complex, (D',)
    U_tilde = g * U * K      # broadcast K over spatial
    out = SAGU(U_tilde)

The SPN output ``g`` is the attribution capture point.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from h9.sagu import DOSTDomainSAGU
from h9.spectral_kernel import SpectralKernel
from h9.spn import PhaseAwareSPN


class SASS(nn.Module):
    """Spectral Adaptive State Space operator.

    Parameters
    ----------
    d_prime : int
        Channel dimension after Warped DOST.
    eps : float
        Numerical epsilon shared with submodules.
    init_mode : str
        Initialization scheme. Default ``"gaussian"``.

    Shape contract
    --------------
    Input  : ``U`` of shape ``(B, d_prime, H, W)`` complex.
    Output : same shape, complex.

    Attribution
    -----------
    Set ``self.capture_gate = True`` to store ``last_gate`` and
    ``last_pre_sagu`` after each forward. Default False.
    """

    def __init__(
        self,
        d_prime: int,
        eps: float = 1e-8,
        init_mode: Literal["gaussian"] = "gaussian",
    ) -> None:
        super().__init__()
        self.spn = PhaseAwareSPN(d_prime, eps=eps, init_mode=init_mode)
        self.kernel = SpectralKernel(d_prime, init_mode=init_mode)
        self.sagu = DOSTDomainSAGU(d_prime, eps=eps)

        self.capture_gate: bool = False
        self.last_gate: Tensor | None = None
        self.last_pre_sagu: Tensor | None = None

    def forward(self, U: Tensor) -> Tensor:
        """Apply SASS.

        Parameters
        ----------
        U : Tensor
            Complex input of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            Same shape, complex.
        """
        g = self.spn(U)                    # (B, D', H, W) real in [0, 1]
        U_kernel = self.kernel(U)          # (B, D', H, W) complex
        U_tilde = g * U_kernel             # gate * kernel-modulated input

        if self.capture_gate:
            self.last_gate = g.detach()
            self.last_pre_sagu = (g * U).detach()  # for |g·U|^2 attribution

        out = self.sagu(U_tilde)
        return out
