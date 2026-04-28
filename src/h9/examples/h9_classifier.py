"""End-to-end H9 classifier example.

Parallel to ``s9.examples.S9ClassifierModelExample`` and
``s9.examples_warped.WarpedS9ClassifierModelExample``, but using h9 blocks
and the h9 design constraints (no patch/positional embedding).

See ``DESIGN-H9.md`` §2 for the full pipeline diagram.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from h9.hss_block import HSSBlock
from s9.transforms.warped_dost import WarpedDOST


class _H9ClassifierFitter:
    """Streaming fitter that mirrors the WarpedDOST.fitter API at the
    classifier level. See ``s9.examples_warped.WarpedS9ClassifierModelExample``
    for the parallel pattern.
    """

    def __init__(self, model: H9ClassifierModelExample) -> None:
        self._model = model
        self._dost_fitter = model.dost.fitter

    def accumulate(self, x: Tensor) -> None:
        """Accumulate calibration statistics from a real-valued input batch.

        Parameters
        ----------
        x : Tensor
            Real input of shape ``(B, in_channels, H, W)``.
        """
        with torch.no_grad():
            u = self._model.stem(x)
            self._dost_fitter.accumulate(u)

    def finalize(self) -> None:
        """Commit the accumulated boundaries to the model's WarpedDOST."""
        self._dost_fitter.finalize()


class H9ClassifierModelExample(nn.Module):
    """End-to-end H9 image classifier.

    Parameters
    ----------
    in_channels : int
        Input image channels (e.g., 3 for RGB).
    d_model : int
        Channel dim after stem. The HSS blocks operate on
        ``d_prime = d_model * n_per_axis ** spatial_dims``.
    n_layers : int
        Number of HSS blocks.
    num_classes : int
        Output classes for the classifier head.
    n_per_axis : int
        Warped DOST band count per spatial axis. Default 2.
    spatial_dims : int
        Spatial dimensionality. Default 2 (images).
    use_stem : bool
        If True (default), apply a 1x1 Conv2D mapping ``in_channels -> d_model``
        before DOST. If False and ``in_channels == d_model``, skip stem.
    gen_activation : type[nn.Module] | None
        Factory for FFN activations. Default StableModReLU.
    d_ff_mult : int
        FFN expansion multiplier. Default 4.
    init_mode : Literal["gaussian"]
        Initialization scheme. Default "gaussian".
    dropout : float
        FFN dropout. Default 0.0.
    eps : float
        Numerical epsilon shared across submodules. Default 1e-8.
    dtype_idx : Literal[32, 64]
        Precision selector following s9 convention. Default 64.

    Attributes
    ----------
    stem : nn.Module
        1x1 Conv2D or Identity.
    dost : WarpedDOST
        The non-learnable DOST preprocessor.
    hss_blocks : nn.ModuleList[HSSBlock]
        Stack of HSS blocks.
    head : nn.Module
        GAP + Linear classifier.

    Calibration
    -----------
    Before any forward pass, calibrate the DOST boundaries:

    >>> # one-shot
    >>> model.calibrate(calib_batch)
    >>> # or streaming
    >>> f = model.fitter
    >>> for chunk in calib_loader:
    >>>     f.accumulate(chunk)
    >>> f.finalize()

    Resolution invariance
    ---------------------
    The same trained weights work at any spatial resolution after re-calibrating
    DOST boundaries on a calibration batch at the new resolution. See
    ``DESIGN-H9.md`` §9.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        num_classes: int,
        n_per_axis: int = 2,
        spatial_dims: int = 2,
        use_stem: bool = True,
        gen_activation: type[nn.Module] | None = None,
        d_ff_mult: int = 4,
        init_mode: Literal["gaussian"] = "gaussian",
        dropout: float = 0.0,
        eps: float = 1e-8,
        dtype_idx: Literal[32, 64] = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_per_axis = n_per_axis
        self.spatial_dims = spatial_dims
        self.d_prime = d_model * (n_per_axis ** spatial_dims)
        self.num_classes = num_classes

        # Stem: 1x1 Conv2D, channel mapping only, no spatial change.
        if use_stem or in_channels != d_model:
            self.stem: nn.Module = nn.Conv2d(
                in_channels, d_model, kernel_size=1, stride=1, padding=0
            )
        else:
            self.stem = nn.Identity()

        # Warped DOST (non-learnable; calibrate before use)
        # TODO(h9): Verify the WarpedDOST constructor signature against the
        # current s9.transforms.warped_dost. The README example uses:
        #   WarpedDOST(D=2, n_per_axis=6)
        # where D=spatial_dims and n_per_axis=n. Adjust if the API has changed.
        self.dost = WarpedDOST(D=spatial_dims, n_per_axis=n_per_axis)

        # HSS blocks
        self.hss_blocks = nn.ModuleList(
            [
                HSSBlock(
                    d_model=d_model,
                    n_per_axis=n_per_axis,
                    spatial_dims=spatial_dims,
                    gen_activation=gen_activation,
                    d_ff_mult=d_ff_mult,
                    init_mode=init_mode,
                    dropout=dropout,
                    eps=eps,
                    dtype_idx=dtype_idx,
                )
                for _ in range(n_layers)
            ]
        )

        # Classifier head: GAP + Linear (operates on real, post-IDOST features)
        self.head = nn.Linear(d_model, num_classes)

        # Attribution flag (default off for zero training overhead)
        self.attribution_enabled: bool = False

    def calibrate(self, x: Tensor) -> None:
        """One-shot DOST calibration on a real-valued batch.

        Parameters
        ----------
        x : Tensor
            Real input of shape ``(B, in_channels, H, W)``. Used to determine
            DOST boundaries via equal-power quantile.
        """
        with torch.no_grad():
            u = self.stem(x)
            self.dost.fit(u)

    @property
    def fitter(self) -> _H9ClassifierFitter:
        """Streaming calibration fitter (parallel to WarpedDOST.fitter)."""
        return _H9ClassifierFitter(self)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Real input of shape ``(B, in_channels, H, W)``. The same model
            handles arbitrary ``(H, W)`` provided DOST has been calibrated
            for that resolution.

        Returns
        -------
        Tensor
            Logits of shape ``(B, num_classes)``.

        Raises
        ------
        RuntimeError
            If DOST has not been calibrated for the input's spatial shape.
        """
        u = self.stem(x)                              # (B, D, H, W) real
        z = self.dost(u)                              # (B, D', H, W) complex
        for block in self.hss_blocks:
            z = block(z)                              # shape preserved
        # Inverse DOST
        inv = self.dost.get_inverse_transform()
        y = inv(z)                                    # (B, D, H, W) real
        # Head: GAP + Linear
        pooled = y.mean(dim=(-2, -1))                 # (B, D)
        logits = self.head(pooled)                    # (B, num_classes)
        return logits
