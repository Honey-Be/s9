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
from ypsilon_torch import FPDTypeIdx, get_complex_dtype, get_float_dtype

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
        dtype_idx: FPDTypeIdx = 64
    ) -> None:
        super().__init__()
        self.d_prime = d_prime
        self.init_mode = init_mode
        self.init_sigma = init_sigma
        self.learnable_sigma = learnable_sigma

        c_dtype = get_complex_dtype(dtype_idx)
        r_dtype = get_float_dtype(dtype_idx)

        self.psi = nn.Parameter(torch.empty(d_prime, dtype=c_dtype))
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.empty((), dtype=r_dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize psi_re, psi_im (and sigma if learnable)."""
        nn.init.normal_(self.psi.real, mean=0.0, std=self.init_sigma)
        nn.init.normal_(self.psi.imag, mean=0.0, std=self.init_sigma)
        if self.learnable_sigma:
            nn.init.ones_(self.sigma)

    def get_kernel(self) -> Tensor:
        """Construct the complex kernel ``K = psi_re + j * psi_im``.

        Returns
        -------
        Tensor
            Complex tensor of shape ``(d_prime,)``.
        """
        K = self.psi
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
