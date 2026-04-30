"""Abstract base classes for SSM kernels.

Hierarchy::

    SSMKernelBase
      └── DiagonalSSMKernelBase          # Vandermonde power-series kernel
              ├── S9SSMKernel            # complex-full
              ├── RS9SSMKernel           # real-only
              └── ARS9SSMKernel          # conjugate-pair, real output
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn

from ypsilon_torch import FPDTypeIdx, get_float_dtype

InitMode = Literal["legacy", "hippo_n", "s4d_real"]
Discretization = Literal["zoh", "approx"]


class SSMKernelBase(nn.Module, ABC):
    """Root ABC for all SSM kernel modules.

    Every kernel converts learned parameters into a 1-D convolution kernel of
    shape ``(channels, length)`` for one spatial axis.
    """

    @abstractmethod
    def __init__(self, d_model: int, N: int = 64, L: Optional[int] = None,
                 dtype_idx: FPDTypeIdx = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.L = L

    @abstractmethod
    def forward(self, length: int) -> torch.Tensor:
        """Return the 1-D kernel of shape ``(d_model, length)``."""
        ...


class DiagonalSSMKernelBase(SSMKernelBase, ABC):
    """Diagonal-A SSM kernel using Vandermonde power-series.

    Subclasses must implement:
        * ``_init_state_params`` — create A/B/C as ``nn.Parameter``
        * ``_materialize_A``    — return the continuous-time A diagonal
        * ``_compute_kernel``   — Vandermonde sum (may need 2·Re trick for ARS9)

    Common across all subclasses:
        * ``log_dt`` parameter and its initialization
        * Discretization dispatch (``"zoh"`` exact vs ``"approx"`` legacy)
    """

    @abstractmethod
    def __init__(
        self,
        d_model: int,
        N: int = 64,
        L: Optional[int] = None,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ) -> None:
        super().__init__(d_model, N, L, dtype_idx)
        self.init_mode: InitMode = init_mode
        self.discretization: Discretization = discretization

        f_dtype = get_float_dtype(dtype_idx)
        self.log_dt = nn.Parameter(
            torch.rand(d_model, dtype=f_dtype) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _materialize_A(self) -> torch.Tensor:
        """Return continuous-time diagonal A of shape ``(d_model, N_eff)``."""
        ...

    @abstractmethod
    def _get_B(self) -> torch.Tensor:
        """Return B parameter of shape ``(d_model, N_eff)``."""
        ...

    @abstractmethod
    def _get_C(self) -> torch.Tensor:
        """Return C parameter of shape ``(d_model, N_eff)``."""
        ...

    @abstractmethod
    def _reduce_kernel(self, raw: torch.Tensor) -> torch.Tensor:
        """Post-process the raw Vandermonde sum. Identity for S9/RS9, ``2·Re`` for ARS9."""
        ...

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def _discretize(self, A: torch.Tensor, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretize ``(A, B)`` → ``(A_bar, B_bar)``.

        ``A_bar = exp(A·dt)`` in both modes; ``B_bar`` differs.
        """
        A_bar = torch.exp(A * dt)

        if self.discretization == "approx":
            B_bar = self._get_B() * dt
        else:
            # Exact ZOH: B_bar = (exp(A·dt) - 1) / A · B
            A_dt = A * dt
            # Taylor fallback for |A·dt| < 1e-4 to avoid 0/0
            small = A_dt.abs() < 1e-4
            # safe divisor (won't actually be used where small==True)
            safe_A = torch.where(small, torch.ones_like(A), A)
            exact = (A_bar - 1) / safe_A
            taylor = dt * (1 + A_dt / 2)
            factor = torch.where(small, taylor, exact)
            B_bar = factor * self._get_B()

        return A_bar, B_bar

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt).unsqueeze(-1)       # (d_model, 1)
        A = self._materialize_A()                        # (d_model, N_eff)
        A_bar, B_bar = self._discretize(A, dt)

        # Vandermonde power series
        range_t = torch.arange(length, device=A.device)
        # (d_model, 1, N_eff) ** (1, L, 1) → (d_model, L, N_eff)
        A_powers = A_bar.unsqueeze(1) ** range_t.unsqueeze(0).unsqueeze(2)
        raw = torch.sum(self._get_C().unsqueeze(1) * B_bar.unsqueeze(1) * A_powers, dim=-1)

        return self._reduce_kernel(raw)                  # (d_model, L)
