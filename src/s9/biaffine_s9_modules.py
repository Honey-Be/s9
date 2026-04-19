from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

try:
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from s9._common.kernel_base import InitMode, Discretization
from s9.base import ComplexActivationFunctionBase, FPDTypeIdx, get_complex_dtype
from s9.multihead_s9_modules import HeadMapperBase, MultiheadS9HeadBase, MultiheadS9LayerBase

from collections.abc import Sequence

__all__ = [
    "ComplexBiaffineChannelMixer",
    "BiaffineS9Head",
    "BiaffineS9Layer",
]


class ComplexBiaffineChannelMixer(nn.Module):
    """Low-rank complex-valued biaffine channel mixer.

    This module parameterizes two channel-side factor matrices
        phi in C^{d_model x latent_channels}
        psi in C^{d_model x latent_channels}
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
        c_dtype = get_complex_dtype(dtype_idx)
        self.d_model: int = d_model
        self.latent_channels: int = latent_channels
        self.channel_embed_dim: int = channel_embed_dim

        self.query_channel_embedding: nn.Parameter = nn.Parameter(
            torch.randn(d_model, channel_embed_dim, dtype=c_dtype)
        )
        self.key_channel_embedding: nn.Parameter = nn.Parameter(
            torch.randn(d_model, channel_embed_dim, dtype=c_dtype)
        )
        self.query_factor: nn.Parameter = nn.Parameter(
            torch.randn(channel_embed_dim + 1, latent_channels, dtype=c_dtype)
        )
        self.key_factor: nn.Parameter = nn.Parameter(
            torch.randn(channel_embed_dim + 1, latent_channels, dtype=c_dtype)
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


class BiaffineS9Head(MultiheadS9HeadBase):
    """One complex-valued biaffine head on top of the multi-head S9 scaffold."""

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
        self.mixer: ComplexBiaffineChannelMixer = ComplexBiaffineChannelMixer(
            d_model=d_model,
            latent_channels=latent_channels,
            channel_embed_dim=channel_embed_dim,
            dtype_idx=dtype_idx,
        )

    class _Prepare(MultiheadS9HeadBase._Prepare):
        @override
        def __init__(self, head_ref: "BiaffineS9Head") -> None:
            super().__init__(head_ref)
            self.head_ref: BiaffineS9Head = head_ref
            self.phi, self.psi = self.head_ref.mixer()

        @override
        def preprocess(self, u: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bc...,cr->br...", u, self.psi)

        @override
        def postprocess(self, y: torch.Tensor) -> torch.Tensor:
            return torch.einsum("br...,cr->bc...", y, self.phi)

    @override
    def _prepare(self) -> "BiaffineS9Head._Prepare":
        return BiaffineS9Head._Prepare(head_ref=self)


class BiaffineS9Layer(MultiheadS9LayerBase[BiaffineS9Head]):
    """Biaffine generalization of MultiheadS9Layer.

    Hierarchy:
        S9Layer -> MultiheadS9Layer -> BiaffineS9Layer
    """

    class HeadMapper(HeadMapperBase[BiaffineS9Head]):
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
        def mapping(self, ch: int, dtype_idx: FPDTypeIdx) -> BiaffineS9Head:
            return BiaffineS9Head(
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
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase],
        latent_channels: Sequence[int],
        n_heads: int = 1,
        channel_embed_dim: int = 16,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
    ) -> None:
        super().__init__(
            d_model=d_model,
            spatial_dims=spatial_dims,
            gen_activation=gen_activation,
            n_heads=n_heads,
            mapper=BiaffineS9Layer.HeadMapper(d_model, spatial_dims, channel_embed_dim,
                                              init_mode, discretization),
            channels=latent_channels,
            eps=eps,
            dtype_idx=dtype_idx,
            init_mode=init_mode,
            discretization=discretization,
        )
