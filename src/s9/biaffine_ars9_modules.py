from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

try:
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from ypsilon_torch import FPDTypeIdx, get_float_dtype
from s9.multihead_ars9_modules import HeadMapperBase, MultiheadARS9HeadBase, MultiheadARS9LayerBase
from s9._common.kernel_base import InitMode, Discretization
from s9.biaffine_rs9_modules import RealBiaffineChannelMixer

from collections.abc import Sequence


__all__ = [
    "BiaffineARS9Head",
    "BiaffineARS9Layer",
]


class BiaffineARS9Head(MultiheadARS9HeadBase):
    """One real-valued biaffine head on top of the multi-head ARS9 scaffold."""

    @override
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        latent_channels: int,
        channel_embed_dim: int,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ) -> None:
        super().__init__(
            d_model=d_model,
            spatial_dims=spatial_dims,
            channels=latent_channels,
            dtype_idx=dtype_idx,
            init_mode=init_mode,
            discretization=discretization,
        )
        self.mixer: RealBiaffineChannelMixer = RealBiaffineChannelMixer(
            d_model=d_model,
            latent_channels=latent_channels,
            channel_embed_dim=channel_embed_dim,
            dtype_idx=dtype_idx,
        )

    class _Prepare(MultiheadARS9HeadBase._Prepare):
        @override
        def __init__(self, head_ref: "BiaffineARS9Head") -> None:
            super().__init__(head_ref)
            self.head_ref: BiaffineARS9Head = head_ref
            self.phi, self.psi = self.head_ref.mixer()

        @override
        def preprocess(self, u: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bc...,cr->br...", u, self.psi)

        @override
        def postprocess(self, y: torch.Tensor) -> torch.Tensor:
            return torch.einsum("br...,cr->bc...", y, self.phi)

    @override
    def _prepare(self) -> "BiaffineARS9Head._Prepare":
        return BiaffineARS9Head._Prepare(head_ref=self)


class BiaffineARS9Layer(MultiheadARS9LayerBase[BiaffineARS9Head]):
    """Biaffine generalization of MultiheadARS9Layer.

    Hierarchy:
        ARS9Layer -> MultiheadARS9Layer -> BiaffineARS9Layer
    """

    class HeadMapper(HeadMapperBase[BiaffineARS9Head]):
        def __init__(self, d_model: int, spatial_dims: int, channel_embed_dim: int,
                     init_mode: InitMode = "legacy",
                     discretization: Discretization = "zoh") -> None:
            super().__init__()
            self.d_model: int = d_model
            self.spatial_dims: int = spatial_dims
            self.channel_embed_dim: int = channel_embed_dim
            self.init_mode: InitMode = init_mode
            self.discretization: Discretization = discretization

        @override
        def mapping(self, ch: int, dtype_idx: FPDTypeIdx) -> BiaffineARS9Head:
            return BiaffineARS9Head(
                d_model=self.d_model,
                spatial_dims=self.spatial_dims,
                latent_channels=ch,
                channel_embed_dim=self.channel_embed_dim,
                dtype_idx=dtype_idx,
                init_mode=self.init_mode,
                discretization=self.discretization,
            )

    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], nn.Module],
        latent_channels: Sequence[int],
        n_heads: int = 1,
        channel_embed_dim: int = 16,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ) -> None:
        super().__init__(
            d_model,
            spatial_dims,
            gen_activation,
            n_heads,
            BiaffineARS9Layer.HeadMapper(d_model, spatial_dims, channel_embed_dim,
                                         init_mode, discretization),
            eps,
            dtype_idx,
            latent_channels,
            init_mode,
            discretization,
        )
