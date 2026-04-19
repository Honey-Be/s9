"""Post-quantization discrete-time stability check."""

from __future__ import annotations

import warnings

import torch


def assert_discrete_stability(
    A_bar: torch.Tensor,
    epsilon: float = 1e-3,
    enforce: bool = True,
) -> None:
    """Check that all discrete eigenvalues satisfy ``|A_bar| < 1 - epsilon``.

    Args:
        A_bar: Discrete-time diagonal state matrix.
        epsilon: Stability margin.
        enforce: If True, raise ``RuntimeError`` on violation; else ``RuntimeWarning``.
    """
    max_abs = A_bar.abs().max().item()
    threshold = 1.0 - epsilon

    if max_abs >= threshold:
        msg = (
            f"Discrete stability violation: max|A_bar| = {max_abs:.6f} "
            f">= {threshold:.6f} (threshold = 1 - {epsilon})"
        )
        if enforce:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
