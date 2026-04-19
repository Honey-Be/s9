"""Checkpoint migration between ``approx`` and ``zoh`` discretizations.

The ``approx`` discretization uses ``B_bar = B * dt``, while ``zoh`` uses
``B_bar = (exp(A*dt) - 1) / A * B``.  To preserve the *effective kernel*
after switching discretization, ``B`` must be rescaled by::

    B_new = B_old * (A * dt) / (exp(A * dt) - 1)        [approx → zoh]
    B_new = B_old * (exp(A * dt) - 1) / (A * dt)         [zoh → approx]

This transform is exact (lossless) and per-channel-per-state.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Literal, Optional

import torch


# ---- kernel family detection patterns ----------------------------------------

# S9SSMKernel:  has log_A_real, A_imag, B (complex)
# RS9SSMKernel: has log_A, B (real)
# ARS9SSMKernel: has log_A_real, A_imag, B_real, B_imag

_KERNEL_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.)?kernels\.(?P<idx>\d+)\."
    r"(?P<param>log_A_real|log_A|A_imag|B|B_real|B_imag|C|C_real|C_imag|log_dt)$"
)


def _detect_family(keys: set[str], prefix: str) -> Literal["S9", "RS9", "ARS9"]:
    """Detect kernel family from the parameter names under *prefix*."""
    has_log_A_real = f"{prefix}log_A_real" in keys
    has_log_A = f"{prefix}log_A" in keys
    has_B_real = f"{prefix}B_real" in keys

    if has_log_A_real and has_B_real:
        return "ARS9"
    elif has_log_A_real:
        return "S9"
    elif has_log_A:
        return "RS9"
    else:
        raise ValueError(f"Cannot detect kernel family at prefix '{prefix}'")


def _compute_factor(A: torch.Tensor, dt: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute ``A*dt / (exp(A*dt) - 1)`` with Taylor fallback for small ``|A*dt|``."""
    A_dt = A * dt
    small = A_dt.abs() < eps
    safe_A_dt = torch.where(small, torch.ones_like(A_dt), A_dt)
    exact = safe_A_dt / (torch.exp(safe_A_dt) - 1)
    taylor = 1 / (1 + A_dt / 2)  # 1st-order Taylor of x/(e^x - 1)
    return torch.where(small, taylor, exact)


def _group_kernel_instances(state_dict: dict[str, torch.Tensor]) -> dict[str, set[str]]:
    """Group state_dict keys by kernel instance prefix.

    Returns:
        Mapping from kernel prefix (e.g. ``"layers.0.kernels.0."``) to the
        set of parameter short-names present for that kernel.
    """
    groups: dict[str, set[str]] = {}
    for key in state_dict:
        m = _KERNEL_KEY_RE.match(key)
        if m:
            prefix = f"{m.group('prefix') or ''}kernels.{m.group('idx')}."
            groups.setdefault(prefix, set()).add(m.group("param"))
    return groups


def _migrate_kernel(
    sd: dict[str, torch.Tensor],
    prefix: str,
    family: Literal["S9", "RS9", "ARS9"],
    direction: Literal["to_zoh", "from_zoh"],
) -> None:
    """In-place rescale ``B`` parameters for one kernel instance."""
    # Reconstruct continuous-time A
    if family == "RS9":
        A = -torch.exp(sd[f"{prefix}log_A"])
    else:  # S9 or ARS9
        A = -torch.exp(sd[f"{prefix}log_A_real"]) + 1j * sd[f"{prefix}A_imag"]

    dt = torch.exp(sd[f"{prefix}log_dt"]).unsqueeze(-1)
    factor = _compute_factor(A, dt)

    if direction == "from_zoh":
        factor = 1.0 / factor

    if family == "ARS9":
        B_complex = torch.complex(sd[f"{prefix}B_real"], sd[f"{prefix}B_imag"])
        B_new = B_complex * factor
        sd[f"{prefix}B_real"] = B_new.real.clone()
        sd[f"{prefix}B_imag"] = B_new.imag.clone()
    else:
        sd[f"{prefix}B"] = sd[f"{prefix}B"] * factor


def migrate_state_dict_zoh(
    state_dict: dict[str, torch.Tensor],
    kernel_family: Optional[Literal["S9", "RS9", "ARS9", "auto"]] = "auto",
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    """Migrate a state_dict from ``approx`` discretization to ``zoh``.

    Args:
        state_dict: Original checkpoint (not mutated).
        kernel_family: Override family detection (``"auto"`` detects per kernel).
        strict: If True, raise on unrecognised kernel prefixes.

    Returns:
        New state_dict with rescaled ``B`` parameters.
    """
    sd = deepcopy(state_dict)
    groups = _group_kernel_instances(sd)
    all_keys = set(sd.keys())

    for prefix, params in groups.items():
        if kernel_family == "auto":
            family = _detect_family(all_keys, prefix)
        else:
            family = kernel_family  # type: ignore[assignment]
        _migrate_kernel(sd, prefix, family, "to_zoh")

    return sd


def migrate_state_dict_from_zoh(
    state_dict: dict[str, torch.Tensor],
    kernel_family: Optional[Literal["S9", "RS9", "ARS9", "auto"]] = "auto",
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    """Migrate a state_dict from ``zoh`` discretization to ``approx``.

    Inverse of :func:`migrate_state_dict_zoh`.
    """
    sd = deepcopy(state_dict)
    groups = _group_kernel_instances(sd)
    all_keys = set(sd.keys())

    for prefix, params in groups.items():
        if kernel_family == "auto":
            family = _detect_family(all_keys, prefix)
        else:
            family = kernel_family  # type: ignore[assignment]
        _migrate_kernel(sd, prefix, family, "from_zoh")

    return sd
