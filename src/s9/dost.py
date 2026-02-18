from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, final, Optional

import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from s9.base import NonLearnableProcessorBase, FPDTypeIdx, COMPLEX_DTYPES_DICT, FLOAT_DTYPES_DICT


def get_dyadic_partitions(N: int) -> list[tuple[int, int]]:
    """Dyadic partitions over the FFT index axis [0, N).

    This is a pragmatic partitioning used for (M-)DOST-style banding.
    We treat the spectrum as indices; negative frequencies are represented
    in the upper half of the FFT output.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    N_eff = N if (N % 2 == 0) else (N - 1)
    parts: list[tuple[int, int]] = [(0, 1)]

    k = 1
    half = N_eff // 2
    while k < half:
        end = min(2 * k, half)
        parts.append((k, end))
        k *= 2

    # Remainder (Nyquist + negative frequencies) as a final block.
    if half < N_eff:
        parts.append((half, N_eff))
    # If we reduced an odd N, include the last index.
    if N_eff != N:
        parts.append((N_eff, N))
    return parts


@lru_cache(maxsize=256)
def _band_slices_cached(spatial_shape: tuple[int, ...]) -> tuple[tuple[slice, ...], ...]:
    per_dim = [get_dyadic_partitions(s) for s in spatial_shape]
    bands = list(itertools.product(*per_dim))
    out: list[tuple[slice, ...]] = []
    for band in bands:
        out.append(tuple(slice(int(st), int(en)) for (st, en) in band))
    return tuple(out)


class _DOSTBase(NonLearnableProcessorBase, ABC):
    """Common base for DOST/IDOST.

    v0.x backport notes:
    - Band metadata (start/end and slice tuples) is fixed at init time.
    - Dense `mask` is built once using slice assignment (no meshgrid).
    - IDOST supports a `strategy` selector; currently auto->sparse.
    """

    @final
    @override
    def __init__(self, spatial_shape: tuple[int, ...]):
        super().__init__()
        self.S: tuple[int, ...] = tuple(int(s) for s in spatial_shape)
        self.D: int = len(self.S)

        # Precompute band slices (Python, cached by spatial_shape).
        band_slices = _band_slices_cached(self.S)
        self._band_slices: tuple[tuple[slice, ...], ...] = band_slices
        self.num_bands: int = len(band_slices)

        # Build band_start/band_end as (BANDS, D) int64 tensors (buffers).
        bs = torch.empty((self.num_bands, self.D), dtype=torch.int64)
        be = torch.empty((self.num_bands, self.D), dtype=torch.int64)
        for b, sl in enumerate(band_slices):
            for d, s in enumerate(sl):
                bs[b, d] = int(s.start or 0)
                be[b, d] = int(s.stop)
        self.register_buffer("band_start", bs, persistent=False)
        self.register_buffer("band_end", be, persistent=False)

        # Dense mask: (BANDS, *S) bool
        # Built by slice assignment (no coordinate grids).
        mask = torch.zeros((self.num_bands, *self.S), dtype=torch.bool)
        for b, sl in enumerate(band_slices):
            mask[(b,) + sl] = True
        self.register_buffer("mask", mask, persistent=False)

    @final
    def _fft_dims(self) -> tuple[int, ...]:
        # For tensors shaped (B, C, *S)
        return tuple(range(2, 2 + self.D))


class DOST(_DOSTBase):
    """Discrete Orthogonal Stockwell Transform.

    Input:  (B, C, *S) real
    Output: (B, C * BANDS, *S) complex
    """

    @override
    def __init__(self, spatial_shape: tuple[int, ...]):
        super().__init__(spatial_shape)

    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (not x.dtype.is_complex) and (x.ndim >= self.D + 2)

    def _to_complex(self, x: torch.Tensor, dtype_idx: Optional[FPDTypeIdx] = None) -> torch.Tensor:
        # v0.x policy: float16 is experimental; promote to complex64.
        if dtype_idx is None:
            if x.dtype == torch.float64:
                return x.to(torch.complex128)
            if x.dtype == torch.float16:
                return x.to(torch.complexe32)
            return x.to(torch.float32).to(torch.complex64)
        else:
            return x.to(FLOAT_DTYPES_DICT[dtype_idx]).to(COMPLEX_DTYPES_DICT[dtype_idx])

    @override
    def transform(self, x: torch.Tensor, dtype_idx: Optional[FPDTypeIdx] = None) -> torch.Tensor:
        dims = self._fft_dims()
        x_c = self._to_complex(x, dtype_idx)

        # (B,C,*S) -> (B,C,1,*S)
        f_x = torch.fft.fftn(x_c, dim=dims).unsqueeze(2)

        # Dense path: single ifftn over (B,C,BANDS,*S)
        mask = self.mask.to(device=f_x.device)
        band_freq = f_x * mask.unsqueeze(0).unsqueeze(0)
        band_time = torch.fft.ifftn(band_freq, dim=dims)

        B, C = x.shape[:2]
        return band_time.reshape(B, C * self.num_bands, *self.S)


class IDOST(_DOSTBase):
    """Inverse DOST.

    Input:  (B, C * BANDS, *S) complex
    Output: (B, C, *S) real

    `strategy`:
      - "sparse": slice-accumulate (preferred)
      - "dense": dense mask multiply + sum (reference)
      - "auto": currently maps to "sparse" (future: heuristic)
    """

    @override
    def __init__(self, spatial_shape: tuple[int, ...]):
        super().__init__(spatial_shape)

    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return z.dtype.is_complex and (z.ndim >= self.D + 2)

    def transform(
        self,
        z: torch.Tensor,
        *,
        strategy: Literal["auto", "dense", "sparse"] = "auto",
        band_block: int = 64,
        dtype_idx: Optional[FPDTypeIdx] = None
    ) -> torch.Tensor:
        dims = self._fft_dims()
        B, C_expanded = z.shape[:2]
        if C_expanded % self.num_bands != 0:
            raise RuntimeError("Invalid DOST band structure")
        C = C_expanded // self.num_bands
        z = z.view(B, C, self.num_bands, *self.S)

        if strategy == "auto":
            strategy = "sparse"  # v0.x mapping

        if strategy == "dense":
            z_f = torch.fft.fftn(z, dim=dims)
            mask = self.mask.to(device=z.device)
            f_recon = torch.sum(z_f * mask.unsqueeze(0).unsqueeze(0), dim=2)
            recon = torch.fft.ifftn(f_recon, dim=dims)
            return recon.real

        # sparse: slice-accumulate
        f_recon = torch.zeros((B, C, *self.S), device=z.device, dtype=z.dtype)
        blk = int(band_block)
        if blk <= 0:
            blk = self.num_bands
        blk = min(blk, self.num_bands)

        for b0 in range(0, self.num_bands, blk):
            b1 = min(b0 + blk, self.num_bands)
            z_blk = z[:, :, b0:b1].contiguous()
            zf_blk = torch.fft.fftn(z_blk, dim=dims)  # (B,C,blk,*S)
            for i, band in enumerate(range(b0, b1)):
                sl = self._band_slices[band]
                f_recon[(...,) + sl] += zf_blk[:, :, i][(...,) + sl]

        recon = torch.fft.ifftn(f_recon, dim=dims)

        if dtype_idx is None:
            return recon.real
        else:
            return recon.real.to(FLOAT_DTYPES_DICT[dtype_idx])
