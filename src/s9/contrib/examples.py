import torch
import torch.nn as nn
from typing import Tuple

from ypsilon_torch import FPDTypeIdx, get_complex_dtype, get_float_dtype
from ypsilon_torch.blocks.transforms.real_complex.dost import DOST
from ypsilon_torch.blocks.activations.complex import StableModReLU
from ypsilon_torch.blocks.activations.real import ThASh

from s9.contrib.gated_delta_s9_modules import GatedDeltaS9Layer, BiaffineGatedDeltaS9Layer
from s9.contrib.gated_delta_rs9_modules import GatedDeltaRS9Layer, BiaffineGatedDeltaRS9Layer


def _make_rs9_activation(d_model: int, eps: float, dtype_idx: FPDTypeIdx) -> nn.Module:
    """Factory for real-valued activation compatible with gen_activation signature."""
    return ThASh()


class _S9ClassifierBase(nn.Module):
    """Shared base for S9-based (complex) gated delta classifier examples."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype_idx: FPDTypeIdx = dtype_idx
        self.spatial_shape = spatial_shape
        self.D = len(spatial_shape)
        self.d_model = d_model

        self.dost = DOST(self.D)
        self.input_proj = None

        self.norm = nn.LayerNorm(d_model, dtype=get_float_dtype(dtype_idx))
        self.classifier = nn.Linear(d_model, num_classes, dtype=get_float_dtype(dtype_idx))

    def _init_input_proj(self, c_expanded: int, device: torch.device) -> None:
        c_dtype = get_complex_dtype(self.dtype_idx)
        if self.D == 1:
            conv_cls = nn.Conv1d
        elif self.D == 2:
            conv_cls = nn.Conv2d
        elif self.D == 3:
            conv_cls = nn.Conv3d
        else:
            conv_cls = None

        if conv_cls is not None:
            self.input_proj = conv_cls(c_expanded, self.d_model, kernel_size=1, dtype=c_dtype).to(device)
        else:
            self.input_proj = nn.Linear(c_expanded, self.d_model, bias=True, dtype=c_dtype).to(device)
            self.is_high_dim_proj = True

    def _project_input(self, x_dost: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'is_high_dim_proj') and self.is_high_dim_proj:
            permute_order = [0] + list(range(2, 2 + self.D)) + [1]
            x = x_dost.permute(*permute_order)
            x = self.input_proj(x)
            inv_order = [0, self.D + 1] + list(range(1, 1 + self.D))
            return x.permute(*inv_order)
        else:
            return self.input_proj(x_dost)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dost = self.dost(x)

        if self.input_proj is None:
            self._init_input_proj(x_dost.shape[1], x.device)

        x = self._project_input(x_dost)

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual

        reduce_dims = list(range(-self.D, 0))
        x = x.mean(dim=reduce_dims)
        x_mag = torch.abs(x)
        x_final = self.norm(x_mag)
        return self.classifier(x_final)


class GatedDeltaS9ClassifierExample(_S9ClassifierBase):
    """Example classifier using GatedDeltaS9Layer backbone.

    Flow: Input (Real) -> DOST -> Complex -> Projection
          -> Stack of Gated Delta S9 Layers -> Magnitude Pooling -> Classifier
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__(in_channels, d_model, num_classes, spatial_shape, dtype_idx)
        self.layers = nn.ModuleList([
            GatedDeltaS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=StableModReLU,
                n_heads=n_heads,
                head_channels=(in_channels,),
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])


class BiaffineGatedDeltaS9ClassifierExample(_S9ClassifierBase):
    """Example classifier using BiaffineGatedDeltaS9Layer backbone.

    Flow: Input (Real) -> DOST -> Complex -> Projection
          -> Stack of Biaffine Gated Delta S9 Layers -> Magnitude Pooling -> Classifier
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        channel_embed_dim: int = 16,
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__(in_channels, d_model, num_classes, spatial_shape, dtype_idx)
        self.layers = nn.ModuleList([
            BiaffineGatedDeltaS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=StableModReLU,
                n_heads=n_heads,
                latent_channels=(in_channels,),
                channel_embed_dim=channel_embed_dim,
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])


class _RS9ClassifierBase(nn.Module):
    """Shared base for RS9-based (real) gated delta classifier examples."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype_idx: FPDTypeIdx = dtype_idx
        self.spatial_shape = spatial_shape
        self.D = len(spatial_shape)
        self.d_model = d_model
        self.in_channels = in_channels

        self.input_proj = None

        self.norm = nn.LayerNorm(d_model, dtype=get_float_dtype(dtype_idx))
        self.classifier = nn.Linear(d_model, num_classes, dtype=get_float_dtype(dtype_idx))

    def _init_input_proj(self, device: torch.device) -> None:
        f_dtype = get_float_dtype(self.dtype_idx)
        if self.D == 1:
            conv_cls = nn.Conv1d
        elif self.D == 2:
            conv_cls = nn.Conv2d
        elif self.D == 3:
            conv_cls = nn.Conv3d
        else:
            conv_cls = None

        if conv_cls is not None:
            self.input_proj = conv_cls(self.in_channels, self.d_model, kernel_size=1, dtype=f_dtype).to(device)
        else:
            self.input_proj = nn.Linear(self.in_channels, self.d_model, bias=True, dtype=f_dtype).to(device)
            self.is_high_dim_proj = True

    def _project_input(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'is_high_dim_proj') and self.is_high_dim_proj:
            permute_order = [0] + list(range(2, 2 + self.D)) + [1]
            xp = x.permute(*permute_order)
            xp = self.input_proj(xp)
            inv_order = [0, self.D + 1] + list(range(1, 1 + self.D))
            return xp.permute(*inv_order)
        else:
            return self.input_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_proj is None:
            self._init_input_proj(x.device)

        x = self._project_input(x)

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual

        reduce_dims = list(range(-self.D, 0))
        x = x.mean(dim=reduce_dims)
        x_final = self.norm(x)
        return self.classifier(x_final)


class GatedDeltaRS9ClassifierExample(_RS9ClassifierBase):
    """Example classifier using GatedDeltaRS9Layer backbone.

    Flow: Input (Real) -> Projection -> Stack of Gated Delta RS9 Layers
          -> Global Pooling -> Classifier
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__(in_channels, d_model, num_classes, spatial_shape, dtype_idx)
        self.layers = nn.ModuleList([
            GatedDeltaRS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=_make_rs9_activation,
                n_heads=n_heads,
                head_channels=(in_channels,),
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])


class BiaffineGatedDeltaRS9ClassifierExample(_RS9ClassifierBase):
    """Example classifier using BiaffineGatedDeltaRS9Layer backbone.

    Flow: Input (Real) -> Projection -> Stack of Biaffine Gated Delta RS9 Layers
          -> Global Pooling -> Classifier
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        channel_embed_dim: int = 16,
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__(in_channels, d_model, num_classes, spatial_shape, dtype_idx)
        self.layers = nn.ModuleList([
            BiaffineGatedDeltaRS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=_make_rs9_activation,
                n_heads=n_heads,
                latent_channels=(in_channels,),
                channel_embed_dim=channel_embed_dim,
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])


# ---------------------------------------------------------------------------
# ARS9 Gated Delta classifier examples
# ---------------------------------------------------------------------------

from s9.contrib.gated_delta_ars9_modules import GatedDeltaARS9Layer, BiaffineGatedDeltaARS9Layer


class GatedDeltaARS9ClassifierExample(_RS9ClassifierBase):
    """Example classifier using GatedDeltaARS9Layer backbone.

    Flow: Input (Real) -> Projection -> Stack of Gated Delta ARS9 Layers
          -> Global Pooling -> Classifier
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__(in_channels, d_model, num_classes, spatial_shape, dtype_idx)
        self.layers = nn.ModuleList([
            GatedDeltaARS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=_make_rs9_activation,
                n_heads=n_heads,
                head_channels=(in_channels,),
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])


class BiaffineGatedDeltaARS9ClassifierExample(_RS9ClassifierBase):
    """Example classifier using BiaffineGatedDeltaARS9Layer backbone.

    Flow: Input (Real) -> Projection -> Stack of Biaffine Gated Delta ARS9 Layers
          -> Global Pooling -> Classifier
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        channel_embed_dim: int = 16,
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__(in_channels, d_model, num_classes, spatial_shape, dtype_idx)
        self.layers = nn.ModuleList([
            BiaffineGatedDeltaARS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=_make_rs9_activation,
                n_heads=n_heads,
                latent_channels=(in_channels,),
                channel_embed_dim=channel_embed_dim,
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])
