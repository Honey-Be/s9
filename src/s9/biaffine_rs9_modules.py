from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

try:
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from ypsilon_torch import FPDTypeIdx, get_float_dtype
from s9.multihead_rs9_modules import HeadMapperBase, MultiheadRS9HeadBase, MultiheadRS9LayerBase
from s9._common.kernel_base import InitMode, Discretization

from collections.abc import Sequence


__all__ = [
    "RealBiaffineChannelMixer",
    "BiaffineRS9Head",
    "BiaffineRS9Layer",
]


class RealBiaffineChannelMixer(nn.Module):
    """Low-rank real-valued biaffine channel mixer.

    This module parameterizes two channel-side factor matrices
        phi in R^{d_model x latent_channels}
        psi in R^{d_model x latent_channels}
    through augmented channel embeddings so that affine terms are absorbed
    into the same factorization.
    """

    def __init__(
        self,
        d_model: int,
        latent_channels: int,
        channel_embed_dim: int,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        f_dtype = get_float_dtype(dtype_idx)
        self.d_model: int = d_model
        self.latent_channels: int = latent_channels
        self.channel_embed_dim: int = channel_embed_dim

        self.query_channel_embedding: nn.Parameter = nn.Parameter(
            torch.randn(d_model, channel_embed_dim, dtype=f_dtype)
        )
        self.key_channel_embedding: nn.Parameter = nn.Parameter(
            torch.randn(d_model, channel_embed_dim, dtype=f_dtype)
        )
        self.query_factor: nn.Parameter = nn.Parameter(
            torch.randn(channel_embed_dim + 1, latent_channels, dtype=f_dtype)
        )
        self.key_factor: nn.Parameter = nn.Parameter(
            torch.randn(channel_embed_dim + 1, latent_channels, dtype=f_dtype)
        )

    def compute_output_coefficients(self) -> torch.Tensor:
        ones = torch.ones(
            self.d_model,
            1,
            dtype=self.query_channel_embedding.dtype,
            device=self.query_channel_embedding.device,
        )
        query_aug = torch.cat([self.query_channel_embedding, ones], dim=-1)
        return query_aug @ self.query_factor

    def compute_input_coefficients(self) -> torch.Tensor:
        ones = torch.ones(
            self.d_model,
            1,
            dtype=self.key_channel_embedding.dtype,
            device=self.key_channel_embedding.device,
        )
        key_aug = torch.cat([self.key_channel_embedding, ones], dim=-1)
        return key_aug @ self.key_factor

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.compute_output_coefficients()
        psi = self.compute_input_coefficients()
        return phi, psi


class BiaffineRS9Head(MultiheadRS9HeadBase):
    """One real-valued biaffine head on top of the multi-head RS9 scaffold."""

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

    class _Prepare(MultiheadRS9HeadBase._Prepare):
        @override
        def __init__(self, head_ref: "BiaffineRS9Head") -> None:
            super().__init__(head_ref)
            self.head_ref: BiaffineRS9Head = head_ref
            self.phi, self.psi = self.head_ref.mixer()

        @override
        def preprocess(self, u: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bc...,cr->br...", u, self.psi)

        @override
        def postprocess(self, y: torch.Tensor) -> torch.Tensor:
            return torch.einsum("br...,cr->bc...", y, self.phi)

    @override
    def _prepare(self) -> "BiaffineRS9Head._Prepare":
        return BiaffineRS9Head._Prepare(head_ref=self)


class BiaffineRS9Layer(MultiheadRS9LayerBase[BiaffineRS9Head]):
    """Biaffine generalization of MultiheadRS9Layer.

    Hierarchy:
        RS9Layer -> MultiheadRS9Layer -> BiaffineRS9Layer
    """

    class HeadMapper(HeadMapperBase[BiaffineRS9Head]):
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
        def mapping(self, ch: int, dtype_idx: FPDTypeIdx) -> BiaffineRS9Head:
            return BiaffineRS9Head(
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
            BiaffineRS9Layer.HeadMapper(d_model, spatial_dims, channel_embed_dim,
                                        init_mode, discretization),
            eps,
            dtype_idx,
            latent_channels,
            init_mode,
            discretization,
        )
