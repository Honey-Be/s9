"""S9 classifier example using the calibrated, resolution-invariant WarpedDOST.

Place at: ``src/s9/examples_warped.py`` (or merge into ``examples.py``).

Key difference from ``S9ClassifierModelExample``
------------------------------------------------
The standard ``DOST``-based example uses lazy ``input_proj`` initialization
because the channel expansion depends on resolution. With ``WarpedDOST``
the expansion is fixed at ``in_channels * n_per_axis ** D``, so we
initialize ``input_proj`` eagerly. Together with the refit-per-N
calibration workflow, **a single trained model applies to any resolution**
(subject to ``n_per_axis <= N // 2 + 2`` per axis).

Calibration
-----------
* One-shot:  ``model.calibrate(x_calib)``
* Streaming: ``f = model.fitter; for c in chunks: f.accumulate(c); f.finalize()``

Calibration uses only forward statistics (no labels). Calling ``forward``
on a spatial shape that has not been calibrated for raises ``RuntimeError``
from :class:`WarpedDOST` — this is intentional to keep the workflow explicit.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ypsilon_torch.blocks.activations.complex import StableModReLU
from ypsilon_torch import FPDTypeIdx, get_complex_dtype, get_float_dtype
from s9.modules import S9Layer
from ypsilon_torch.blocks.transforms.real_complex.warped_dost import WarpedDOST, WarpedDOSTFitter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_pointwise_complex_proj(
    c_in: int, c_out: int, D: int, dtype_idx: FPDTypeIdx, device: torch.device
) -> tuple[nn.Module, bool]:
    """1x1 convolution as channel projection in the complex domain.

    Returns ``(proj, is_high_dim)`` where ``is_high_dim`` is True for D > 3
    (in which case ``proj`` is an ``nn.Linear`` to be applied channel-last).
    """
    c_dtype = get_complex_dtype(dtype_idx)
    if D == 1:
        return nn.Conv1d(c_in, c_out, kernel_size=1, dtype=c_dtype).to(device), False
    if D == 2:
        return nn.Conv2d(c_in, c_out, kernel_size=1, dtype=c_dtype).to(device), False
    if D == 3:
        return nn.Conv3d(c_in, c_out, kernel_size=1, dtype=c_dtype).to(device), False
    return nn.Linear(c_in, c_out, bias=True, dtype=c_dtype).to(device), True


def _apply_complex_proj(
    proj: nn.Module,
    is_high_dim: bool,
    x: torch.Tensor,
    D: int,
    dtype_idx: FPDTypeIdx,
) -> torch.Tensor:
    def _do(t: torch.Tensor) -> torch.Tensor:
        if t.dtype == get_complex_dtype(32):
            with torch.amp.autocast(device_type=str(t.device)):
                return (
                    proj.to(dtype=torch.complex64)(t.to(dtype=torch.complex64))
                    .to(dtype=t.dtype)
                )
        return proj(t)

    if is_high_dim:
        permute_order = [0] + list(range(2, 2 + D)) + [1]
        x_p = x.permute(*permute_order)
        x_p = _do(x_p)
        inv_order = [0, D + 1] + list(range(1, 1 + D))
        return x_p.permute(*inv_order)
    return _do(x)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class WarpedS9ClassifierModelExample(nn.Module):
    """S9 classifier built on the calibrated WarpedDOST.

    The transform's channel expansion is fixed at
    ``in_channels * n_per_axis ** D``, so ``input_proj`` is initialized
    eagerly and the same instance can be forwarded with arbitrary spatial
    shapes (after calibration for each shape).
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        n_per_axis: int = 6,
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.d_model: int = d_model
        self.D: int = len(spatial_shape)
        self.spatial_shape = spatial_shape
        self.dtype_idx: FPDTypeIdx = dtype_idx
        self.n_per_axis: int = n_per_axis

        self.transform: WarpedDOST = WarpedDOST(D=self.D, n_per_axis=n_per_axis)

        # Channel expansion depends only on (in_channels, n_per_axis, D),
        # so input_proj can be initialized eagerly.
        c_expanded = in_channels * (n_per_axis ** self.D)
        device = torch.device("cpu")
        self.input_proj, self._is_high_dim_proj = _make_pointwise_complex_proj(
            c_expanded, d_model, self.D, dtype_idx, device
        )

        self.layers: nn.ModuleList = nn.ModuleList([
            S9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=StableModReLU,
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])

        self.norm: nn.Module = nn.LayerNorm(d_model, dtype=get_float_dtype(dtype_idx))
        self.classifier: nn.Linear = nn.Linear(
            d_model, num_classes, dtype=get_float_dtype(dtype_idx)
        )

    # ----------------------------------------------------- calibration API
    def calibrate(self, x_calib: torch.Tensor) -> None:
        """One-shot calibration on an unlabeled real-valued batch.

        ``x_calib`` shape: ``(B, in_channels, *spatial)``. Equivalent to
        ``self.fitter.accumulate(x_calib).finalize()``. Use ``self.fitter``
        directly for streaming calibration when the calibration set does
        not fit in memory at once.
        """
        self.transform.fit(x_calib)

    @property
    def fitter(self) -> WarpedDOSTFitter:
        """Begin a new streaming-calibration session for the underlying transform.

        Equivalent to ``self.transform.fitter``; provided as a top-level
        property for ergonomic parity with :meth:`calibrate`.
        """
        return self.transform.fitter

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 + self.D:
            raise ValueError(
                f"expected input with {self.D} spatial dims, got shape {tuple(x.shape)}"
            )

        x_t = self.transform.transform(x, dtype_idx=self.dtype_idx)
        x_proj = _apply_complex_proj(
            self.input_proj, self._is_high_dim_proj, x_t, self.D, self.dtype_idx
        )

        for layer in self.layers:
            residual = x_proj
            x_proj = layer(x_proj) + residual

        reduce_dims = list(range(-self.D, 0))
        if x_proj.dtype == get_complex_dtype(32):
            with torch.amp.autocast(device_type=str(x_proj.device)):
                pooled = (
                    x_proj.to(torch.complex64).mean(dim=reduce_dims).to(x_proj.dtype)
                )
        else:
            pooled = x_proj.mean(dim=reduce_dims)

        x_mag = torch.abs(pooled)
        x_norm = self.norm(x_mag)
        return self.classifier(x_norm)