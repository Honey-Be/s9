"""Abstract base for biaffine channel mixers (complex and real)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BiaffineChannelMixerBase(nn.Module, ABC):
    """Low-rank biaffine channel mixer.

    Parameterizes two channel-side factor matrices (phi, psi) through
    augmented channel embeddings so that affine terms are absorbed into
    the same factorization.

    Subclasses set the dtype (complex vs real) for all parameters.
    """

    @abstractmethod
    def __init__(
        self,
        d_model: int,
        latent_channels: int,
        channel_embed_dim: int,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.latent_channels: int = latent_channels
        self.channel_embed_dim: int = channel_embed_dim

    def _compute_coefficients(
        self,
        channel_embedding: torch.Tensor,
        factor: torch.Tensor,
    ) -> torch.Tensor:
        ones = torch.ones(
            self.d_model, 1,
            dtype=channel_embedding.dtype,
            device=channel_embedding.device,
        )
        aug = torch.cat([channel_embedding, ones], dim=-1)
        return aug @ factor

    @abstractmethod
    def compute_output_coefficients(self) -> torch.Tensor:
        ...

    @abstractmethod
    def compute_input_coefficients(self) -> torch.Tensor:
        ...

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.compute_output_coefficients()
        psi = self.compute_input_coefficients()
        return phi, psi
