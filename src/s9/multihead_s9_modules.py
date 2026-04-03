from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, TypeVar, final

import torch
import torch.nn as nn

try:
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from s9.base import ComplexActivationFunctionBase, FPDTypeIdx, get_complex_dtype
from s9.modules import S9SSMKernel, ComplexDropout

from collections.abc import Sequence

__all__ = [
    "_normalize_head_channels",
    "_build_outer_product_global_kernel",
    "_fftn_convolve_nd",
    "_apply_channel_last_linear_and_activation",
    "HeadMapperBase",
    "MultiheadS9HeadBase",
    "MultiheadS9Head",
    "MultiheadS9LayerBase",
    "MultiheadS9Layer",
]


def _normalize_head_channels(d_model: int, n_heads: int, head_channels: Sequence[int]) -> list[int]:
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")

    if len(head_channels) == 0:
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads} when head_channels is omitted"
            )
        return [d_model // n_heads] * n_heads

    if len(head_channels) == 1:
        ch = int(head_channels[0])
        if ch <= 0:
            raise ValueError(f"head_channels must be positive, got {head_channels[0]}")
        return [ch] * n_heads

    if len(head_channels) != n_heads:
        raise ValueError(f"Expected {n_heads} head channel sizes, got {len(head_channels)}")

    channels_list = [int(ch) for ch in head_channels]
    if any(ch <= 0 for ch in channels_list):
        raise ValueError("All head channel sizes must be positive")
    return channels_list



def _build_outer_product_global_kernel(
    kernels_1d: Sequence[torch.Tensor],
    spatial_shapes: Sequence[int],
    channels: int,
) -> torch.Tensor:
    if len(kernels_1d) != len(spatial_shapes):
        raise ValueError(
            f"kernels_1d and spatial_shapes must have the same length, got {len(kernels_1d)} and {len(spatial_shapes)}"
        )

    k_global = kernels_1d[0]
    view_shape = [channels] + [1] * len(spatial_shapes)
    view_shape[1] = spatial_shapes[0]
    k_global = k_global.view(*view_shape)

    for dim_idx in range(1, len(spatial_shapes)):
        next_view_shape = [channels] + [1] * len(spatial_shapes)
        next_view_shape[dim_idx + 1] = spatial_shapes[dim_idx]
        k_global = k_global * kernels_1d[dim_idx].view(*next_view_shape)

    return k_global



def _fftn_convolve_nd(u: torch.Tensor, k_global: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    spatial_shapes = u.shape[2:]
    padded_shapes = [size * 2 for size in spatial_shapes]
    fft_dims = tuple(range(-spatial_dims, 0))

    u_f = torch.fft.fftn(u, s=padded_shapes, dim=fft_dims)
    k_f = torch.fft.fftn(k_global, s=padded_shapes, dim=fft_dims)
    y_f = u_f * k_f
    y = torch.fft.ifftn(y_f, s=padded_shapes, dim=fft_dims)

    slices = [slice(None)] * 2 + [slice(0, size) for size in spatial_shapes]
    return y[tuple(slices)]



def _apply_channel_last_linear_and_activation(
    y: torch.Tensor,
    spatial_dims: int,
    activation: nn.Module,
    output_linear: nn.Linear,
    dropout: ComplexDropout,
) -> torch.Tensor:
    permute_order = [0] + list(range(2, 2 + spatial_dims)) + [1]
    inv_permute_order = [0, spatial_dims + 1] + list(range(1, 1 + spatial_dims))

    y = y.permute(*permute_order)
    y = activation(y)
    y = output_linear(y)
    y = dropout(y)
    return y.permute(*inv_permute_order)


class MultiheadS9HeadBase(nn.Module, ABC):
    class _Prepare(ABC):
        @abstractmethod
        def __init__(self, head_ref: "MultiheadS9HeadBase") -> None:
            super().__init__()
            self.head_ref: MultiheadS9HeadBase = head_ref

        @abstractmethod
        def preprocess(self, u: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @abstractmethod
        def postprocess(self, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

    @abstractmethod
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        channels: int,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims
        self.channels: int = channels
        self.dtype_idx: FPDTypeIdx = dtype_idx
        self.kernels: nn.ModuleList = nn.ModuleList(
            [S9SSMKernel(channels, L=None, dtype_idx=dtype_idx) for _ in range(spatial_dims)]
        )

    @abstractmethod
    def _prepare(self) -> "MultiheadS9HeadBase._Prepare":
        raise NotImplementedError

    @final
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        spatial_shapes: tuple[int, ...] = tuple(u.shape[2:])
        preparation = self._prepare()

        latent_input: torch.Tensor = preparation.preprocess(u)
        kernels_1d = [kernel(length=length) for kernel, length in zip(self.kernels, spatial_shapes)]
        k_global = _build_outer_product_global_kernel(kernels_1d, spatial_shapes, self.channels)
        y_latent = _fftn_convolve_nd(latent_input, k_global, self.spatial_dims)

        return preparation.postprocess(y_latent)


class MultiheadS9Head(MultiheadS9HeadBase):
    """One complex-valued head for MultiheadS9Layer."""

    @override
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        head_channels: int,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__(d_model=d_model, spatial_dims=spatial_dims, channels=head_channels, dtype_idx=dtype_idx)
        c_dtype = get_complex_dtype(dtype_idx)
        self.input_linear: nn.Linear = nn.Linear(d_model, head_channels, bias=False, dtype=c_dtype)
        self.output_linear: nn.Linear = nn.Linear(head_channels, d_model, bias=False, dtype=c_dtype)
        self.permute_order: list[int] = [0] + list(range(2, 2 + self.spatial_dims)) + [1]
        self.inv_permute_order: list[int] = [0, self.spatial_dims + 1] + list(range(1, 1 + self.spatial_dims))

    class _Prepare(MultiheadS9HeadBase._Prepare):
        @override
        def __init__(self, head_ref: "MultiheadS9Head") -> None:
            super().__init__(head_ref)
            self.head_ref: MultiheadS9Head = head_ref

        @override
        def preprocess(self, u: torch.Tensor) -> torch.Tensor:
            u_channel_last = u.permute(*self.head_ref.permute_order)
            latent = self.head_ref.input_linear(u_channel_last)
            return latent.permute(*self.head_ref.inv_permute_order)

        @override
        def postprocess(self, y: torch.Tensor) -> torch.Tensor:
            y_channel_last = y.permute(*self.head_ref.permute_order)
            out = self.head_ref.output_linear(y_channel_last)
            return out.permute(*self.head_ref.inv_permute_order)

    @override
    def _prepare(self) -> "MultiheadS9Head._Prepare":
        return MultiheadS9Head._Prepare(head_ref=self)


H = TypeVar("H", bound=MultiheadS9HeadBase)


class HeadMapperBase(Generic[H], ABC):
    @abstractmethod
    def mapping(self, ch: int, dtype_idx: FPDTypeIdx) -> H:
        raise NotImplementedError


class MultiheadS9LayerBase(nn.Module, Generic[H], ABC):
    """Multi-head generalization of S9Layer.

    Relationship to the existing hierarchy:
        S9Layer -> MultiheadS9Layer -> BiaffineS9Layer
    """

    @staticmethod
    def _gen_heads(
        mapper: HeadMapperBase[H],
        d_model: int,
        n_heads: int,
        dtype_idx: FPDTypeIdx,
        head_channels: Sequence[int],
    ) -> list[H]:
        channels_list = _normalize_head_channels(d_model, n_heads, head_channels)
        return [mapper.mapping(ch, dtype_idx) for ch in channels_list]

    @abstractmethod
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase],
        n_heads: int,
        mapper: HeadMapperBase[H],
        channels: Sequence[int],
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.spatial_dims: int = spatial_dims
        self.n_heads: int = n_heads

        self.heads: nn.ModuleList = nn.ModuleList(
            self._gen_heads(mapper, d_model, n_heads, dtype_idx, channels)
        )
        self.activation: nn.Module = gen_activation(d_model, eps, dtype_idx)
        self.output_linear: nn.Linear = nn.Linear(
            d_model, d_model, bias=False, dtype=get_complex_dtype(dtype_idx)
        )
        self.dropout: ComplexDropout = ComplexDropout(0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        spatial_shapes = u.shape[2:]
        if len(spatial_shapes) != self.spatial_dims:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.spatial_dims} spatial dims, got {len(spatial_shapes)}"
            )

        y = torch.zeros_like(u)
        for head in self.heads:
            y = y + head(u)

        return _apply_channel_last_linear_and_activation(
            y=y,
            spatial_dims=self.spatial_dims,
            activation=self.activation,
            output_linear=self.output_linear,
            dropout=self.dropout,
        )


class MultiheadS9Layer(MultiheadS9LayerBase[MultiheadS9Head]):
    """Multi-head generalization of S9Layer.

    Relationship to the existing hierarchy:
        S9Layer -> MultiheadS9Layer -> BiaffineS9Layer
    """

    class HeadMapper(HeadMapperBase[MultiheadS9Head]):
        def __init__(self, d_model: int, spatial_dims: int) -> None:
            super().__init__()
            self.d_model: int = d_model
            self.spatial_dims: int = spatial_dims

        @override
        def mapping(self, ch: int, dtype_idx: FPDTypeIdx) -> MultiheadS9Head:
            return MultiheadS9Head(
                d_model=self.d_model,
                spatial_dims=self.spatial_dims,
                head_channels=ch,
                dtype_idx=dtype_idx,
            )

    @override
    def __init__(
        self,
        d_model: int,
        spatial_dims: int,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase],
        n_heads: int,
        head_channels: Sequence[int],
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__(
            d_model=d_model,
            spatial_dims=spatial_dims,
            gen_activation=gen_activation,
            n_heads=n_heads,
            mapper=MultiheadS9Layer.HeadMapper(d_model, spatial_dims),
            channels=head_channels,
            eps=eps,
            dtype_idx=dtype_idx,
        )
