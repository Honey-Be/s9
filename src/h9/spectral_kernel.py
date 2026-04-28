"""Spectral kernel module (DESIGN-H9 §6.2).

A single learnable complex-valued vector of shape ``(D',)``, broadcast over
spatial dimensions during multiplication. This is HAMSA's "simplified kernel
parameterization": replaces the (A, B, C) state-space matrices with one
learnable complex kernel, eliminating discretization instability.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


class SpectralKernel(nn.Module):
    """Per-band learnable complex kernel.

    Parameters
    ----------
    d_prime : int
        Channel dimension after Warped DOST.
    init_mode : str
        Initialization scheme. Default ``"gaussian"``.
    init_sigma : float
        Standard deviation for Gaussian init. Default ``1.0``.
    learnable_sigma : bool
        If True, ``sigma`` is itself an ``nn.Parameter``. Default True.

    Shape contract
    --------------
    Input  : ``U`` of shape ``(B, d_prime, H, W)`` complex.
    Output : ``U * K`` of shape ``(B, d_prime, H, W)`` complex.
    """

    def __init__(
        self,
        d_prime: int,
        init_mode: Literal["gaussian"] = "gaussian",
        init_sigma: float = 1.0,
        learnable_sigma: bool = True,
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.init_mode = init_mode
        self.init_sigma = init_sigma
        self.learnable_sigma = learnable_sigma

        self.psi_re = nn.Parameter(torch.empty(d_prime))
        self.psi_im = nn.Parameter(torch.empty(d_prime))
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.empty(()))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize psi_re, psi_im (and sigma if learnable)."""
        nn.init.normal_(self.psi_re, mean=0.0, std=self.init_sigma)
        nn.init.normal_(self.psi_im, mean=0.0, std=self.init_sigma)
        if self.learnable_sigma:
            nn.init.ones_(self.sigma)

    def get_kernel(self) -> Tensor:
        """Construct the complex kernel ``K = psi_re + j * psi_im``.

        Returns
        -------
        Tensor
            Complex tensor of shape ``(d_prime,)``.
        """
        K = torch.complex(self.psi_re, self.psi_im)
        if self.learnable_sigma:
            K = self.sigma * K
        return K

    def forward(self, U: Tensor) -> Tensor:
        """Multiply input by the spectral kernel, broadcast over spatial dims.

        Parameters
        ----------
        U : Tensor
            Complex input of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            ``U * K`` of shape ``(B, d_prime, H, W)``.
        """
        K = self.get_kernel()  # (d_prime,)
        return U * K.view(1, -1, 1, 1)
