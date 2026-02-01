import torch
import torch.nn as nn

from functools import reduce

from s9.base import NonLearnableProcessorBase

from typing import override, final
from abc import ABC, abstractmethod


class _DOSTBase(NonLearnableProcessorBase, ABC):
    @final
    @staticmethod
    def __get_dyadic_partitions(self, N: int) -> list[tuple[int, int]]:
        """
        Generates indices for dyadic partitioning of the frequency spectrum.
        Fundamental to the Discrete Orthogonal Stockwell Transform.
        """
        partitions: list[tuple[int, int]] = []
        N_eff: int
        if N % 2 != 0:
            # Handle odd size by padding or simplifying (simplified here for robustness)
            N_eff = N - 1
        else:
            N_eff = N
            
        # DC component
        partitions.append((0, 1))
        
        # Positive frequencies dyadic split
        k: int = 1
        while k < N_eff // 2:
            end = min(2 * k, N_eff // 2)
            partitions.append((k, end))
            k *= 2
            
        # Negative frequencies (Nyquist and beyond)
        # Simplified: Treat remaining high freqs as one block or mirror structure
        # For M-DOST in deep learning, we usually focus on the positive spectrum 
        # and let the complex weights handle the rest, but to be strict:
        if k < N_eff:
            partitions.append((k, N_eff))

        return partitions

    @final
    @staticmethod
    def _map_bands(S: tuple[int, ...]):
        for d, size in enumerate(S):
            partitions: list[tuple[int, int]] = self._get_dyadic_partitions(size)
            for start, end in partitions:
                yield (d, start, end)
    
    @final
    @staticmethod
    def _build_band_metadata(S: tuple[int, ...], device: torch.device):
        band: list[tuple[int, int, int]] = list((d, start, end) for (d, start, end) in self._map_bands(S))
        return (
            torch.tensor(band[:, 0], dtype=torch.int64, device=device),
            torch.tensor(band[:, 1], dtype=torch.int64, device=device),
            torch.tensor(band[:, 2], dtype=torch.int64, device=device),
        )

    @final
    def _build_mask(
        self,
        band_dim: torch.Tensor,
        band_start: torch.Tensor,
        band_end: torch.Tensor,
    ):
        """
        Returns:
            mask: (num_bands, *S) bool
        """
        device = band_dim.device
        D = len(self.S)
        B = band_dim.shape[0]

        # coordinate grid: (*S, D)
        coords = torch.meshgrid(
            *[torch.arange(s, device=device) for s in S],
            indexing="ij"
        )
        grid = torch.stack(coords, dim=-1)  # (*S, D)

        # expand for bands
        grid = grid.unsqueeze(0)            # (B, *S, D)
        bd = band_dim.view(B, *([1] * D))
        bs = band_start.view(B, *([1] * D))
        be = band_end.view(B, *([1] * D))

        in_band = (grid[..., bd] >= bs) & (grid[..., bd] < be)

        # AND across spatial dimensions
        return in_band.all(dim=-1)

    
    @abstractmethod
    @override
    def __init__(self, *spatial_shape: *tuple[int, ...], device: torch.device):
        super().__init__()
        self.S: tuple[int, ...] = spatial_shape
        self.D: int = len(spatial_shape)

        band_dim, band_start, band_end = self._build_band_metadata(self.S, device)

        # register band metadata
        self.register_buffer("band_dim", band_dim, persistent=False)
        self.register_buffer("band_start", band_start, persistent=False)
        self.register_buffer("band_end", band_end, persistent=False)

        mask = self._build_mask(
            self.S,
            band_dim,
            band_start,
            band_end,
        )

        # (num_bands, *S)
        self.register_buffer("mask", mask, persistent=False)
        self.num_bands: int = mask.shape[0]

class DOST(_DOSTBase):
    @override
    def __init__(self, spatial_shape: tuple[int, ...], device: torch.device):
        super().__init__(spatial_shape, device)
    
    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (x.dtype in [torch.float64, torch.float32, torch.float16, torch.int32, torch.int16, torch.int8]) and (len(x.shape) >= self.D + 2)

    def _convert_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float64 or x.dtype == torch.int32:
            return x.to(torch.complex128)
        elif x.dtype == torch.float32 or x.dtype == torch.int16:
            return x.to(torch.complex64)
        elif x.dtype == torch.float16 or x.dtype == torch.int8:
            return x.to(torch.complex32)
        else:
            raise RuntimeError("Cannot convert to complex dtype :(")
    
    @override
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, *self.S) - Real domain
        Returns:
            z: Output tensor of shape (B, C', *self.S) - Complex domain
               Where C' is expanded by the number of frequency bands.
        """

        dims = tuple(range(2, self.D + 2))

        x_c = self._convert_to_complex(x)
        f_x = torch.fft.fftn(x_c, dim=dims).unsqueeze(2)  # (B, C, 1, *S)

        mask = self.mask.unsqueeze(0).unsqueeze(0)

        band_freq = f_x * mask
        band_time = torch.fft.ifftn(band_freq, dim=dims)

        B, C = x.shape[:2]

        return band_time.reshape(B, C * self.num_bands, *self.S)


class IDOST(_DOSTBase):
    @override
    def __init__(self, spatial_shape: tuple[int, ...], device: torch.device):
        super().__init__(spatial_shape, device)
    
    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return (z.dtype.is_complex) and (len(z.shape) >= self.D + 2)

    @override
    def transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C', *self.S) - Complex domain
               Where C' is expanded by the number of frequency bands.
        Returns:
            z: Output tensor of shape (B, C, *self.S) - Real domain
        """

        dims = tuple(range(2, self.D + 2))

        B, C_expanded = z.shape[:2]
        if C_expanded % self.num_bands != 0:
            raise RuntimeError("Invalid DOST band structure")

        C = C_expanded // self.num_bands

        z = z.view(B, C, self.num_bands, *self.S)

        z_f = torch.fft.fftn(z, dim=dims)

        mask = self.mask.unsqueeze(0).unsqueeze(0)
        f_recon = torch.sum(z_f * mask, dim=2)

        recon = torch.fft.ifftn(f_recon, dim=dims)
        return recon.real