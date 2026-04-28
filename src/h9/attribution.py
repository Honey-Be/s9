"""Dual attribution: non-intrusive capture of (spatial x spectral) maps.

Captures, for each HSS block, the tensor ``A = |g * U|^2`` where ``g`` is
the SPN gate and ``U`` is the input. Three aggregation views:

* ``spatial_view``  : sum over channels -> (B, H, W) heatmap
* ``spectral_view`` : sum over (H, W)   -> (B, D') profile
* ``joint_view``    : reshape D' -> (D, n, ..., n), sum over D -> (B, n, n, H, W)

See ``DESIGN-H9.md`` §8.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor, nn

from h9.sass import SASS


class DualAttribution:
    """Manager for dual attribution capture across an h9 model.

    Usage
    -----
    >>> attr = DualAttribution(model)
    >>> attr.enable()
    >>> _ = model(x)
    >>> captured = attr.collect()
    >>> attr.disable()
    >>> heatmap = spatial_view(captured[name])

    Parameters
    ----------
    model : nn.Module
        The h9 model containing ``SASS`` submodules.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._sass_modules: list[tuple[str, SASS]] = list(self._iter_sass_modules())

    def _iter_sass_modules(self) -> Iterator[tuple[str, SASS]]:
        """Iterate (qualified_name, sass_module) pairs in forward order."""
        for name, mod in self._model.named_modules():
            if isinstance(mod, SASS):
                yield name, mod

    def enable(self) -> None:
        """Turn on capture for all SASS modules in the model."""
        for _, sass in self._sass_modules:
            sass.capture_gate = True
            sass.last_gate = None
            sass.last_pre_sagu = None

    def disable(self) -> None:
        """Turn off capture and clear stored tensors."""
        for _, sass in self._sass_modules:
            sass.capture_gate = False
            sass.last_gate = None
            sass.last_pre_sagu = None

    def collect(self) -> dict[str, Tensor]:
        """Return ``|g * U|^2`` per SASS module as a dict.

        Returns
        -------
        dict[str, Tensor]
            Keys are SASS module qualified names, values are real tensors
            of shape ``(B, d_prime, H, W)``.

        Raises
        ------
        RuntimeError
            If capture is enabled but forward has not been called.
        """
        out: dict[str, Tensor] = {}
        for name, sass in self._sass_modules:
            if sass.last_pre_sagu is None:
                raise RuntimeError(
                    f"Attribution not captured for {name}; call forward first."
                )
            out[name] = sass.last_pre_sagu.abs() ** 2  # real, (B, D', H, W)
        return out


def spatial_view(A: Tensor) -> Tensor:
    """Sum attribution over channels to produce a spatial heatmap.

    Parameters
    ----------
    A : Tensor
        Attribution tensor of shape ``(B, d_prime, H, W)``, real-valued.

    Returns
    -------
    Tensor
        Heatmap of shape ``(B, H, W)``.
    """
    return A.sum(dim=1)


def spectral_view(A: Tensor) -> Tensor:
    """Sum attribution over spatial dims to produce a per-channel profile.

    Parameters
    ----------
    A : Tensor
        Attribution tensor of shape ``(B, d_prime, H, W)``, real-valued.

    Returns
    -------
    Tensor
        Profile of shape ``(B, d_prime)``.
    """
    return A.sum(dim=(-2, -1))


def joint_view(A: Tensor, n: int, d: int, spatial_dims: int = 2) -> Tensor:
    """Reshape channel dim and sum over original channels for dual attribution.

    Parameters
    ----------
    A : Tensor
        Attribution tensor of shape ``(B, d_prime, H, W)``.
    n : int
        Warped DOST ``n_per_axis``.
    d : int
        Original channel dim before DOST expansion.
    spatial_dims : int
        Spatial dimensionality. Default 2.

    Returns
    -------
    Tensor
        Joint dual attribution of shape ``(B, n, n, H, W)`` for ``spatial_dims=2``.

    Raises
    ------
    ValueError
        If ``A.shape[1] != d * n ** spatial_dims``.
    """
    B, d_prime = A.shape[0], A.shape[1]
    spatial_shape = A.shape[2:]
    expected = d * n ** spatial_dims
    if d_prime != expected:
        raise ValueError(f"d_prime={d_prime} != d * n^{spatial_dims} = {expected}")

    # Reshape (B, d_prime, *spatial) -> (B, d, n, ..., n, *spatial) -> sum over d
    freq_shape = (n,) * spatial_dims
    A_reshaped = A.view(B, d, *freq_shape, *spatial_shape)
    return A_reshaped.sum(dim=1)
