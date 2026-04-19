import torch
import torch.nn as nn
from typing import Tuple

from s9.base import FPDTypeIdx, get_complex_dtype, get_float_dtype
from s9.transforms.dost import DOST
from s9.modules import S9Layer, StableModReLU
from s9.multihead_s9_modules import MultiheadS9Layer
from s9.biaffine_s9_modules import BiaffineS9Layer
from s9.ars9_modules import ARS9Layer
from s9.multihead_ars9_modules import MultiheadARS9Layer
from s9.biaffine_ars9_modules import BiaffineARS9Layer
from s9.activations.real.thash import ThASh

class S9ClassifierModelExample(nn.Module):
    """
    N-Dimensional S9 Model Architecture.
    입력 데이터의 공간 차원(spatial_shape)에 따라 1D, 2D, 3D 등으로 자동 확장됩니다.
    
    Flow:
    Input (Real) -> ND-DOST -> Complex Features 
    -> Complex Linear Projection -> Stack of ND-S9 Layers -> Magnitude Pooling -> Classifier
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6
    ):
        super().__init__()
        
        self.dtype_idx: FPDTypeIdx = dtype_idx

        self.spatial_shape = spatial_shape
        self.D = len(spatial_shape) # Dimension (1, 2, 3, ...)
        
        # 1. Non-learnable Preprocessor (Multidimensional DOST)
        self.dost = DOST(self.D)
        
        self.input_proj = None 
        self.d_model = d_model
        
        # 2. S9 Layers (Complex Domain, N-Dimensional)
        self.layers = nn.ModuleList(
            [
                S9Layer(
                    d_model=d_model,
                    spatial_dims=self.D,
                    dtype_idx=self.dtype_idx,
                    gen_activation=StableModReLU,
                    eps=eps
                )
                for _ in range(n_layers)
            ]
        )
        
        # 3. Output Head
        self.norm = nn.LayerNorm(d_model, dtype = get_float_dtype(self.dtype_idx))
        self.classifier = nn.Linear(d_model, num_classes, dtype = get_float_dtype(self.dtype_idx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D1, D2, ...) Real valued tensor
        """
        # 1. Preprocessing (DOST)
        # x_dost: (B, C_expanded, D1, D2, ...)
        x_dost = self.dost(x) 
        
        # Initialize input projection lazily
        if self.input_proj is None:
            c_expanded = x_dost.shape[1]
            device = x.device
            # Conv1d, Conv2d, Conv3d 중 차원에 맞는 것 선택
            if self.D == 1:
                conv_cls = nn.Conv1d
            elif self.D == 2:
                conv_cls = nn.Conv2d
            elif self.D == 3:
                conv_cls = nn.Conv3d
            else:
                # 4차원 이상은 nn.Linear로 처리 (채널 축을 마지막으로 보낸 뒤)하거나 Custom 구현 필요
                # 여기서는 1x1 Conv 효과를 내기 위해 단순 Linear Projection 사용 권장하나,
                # 코드 일관성을 위해 weight를 직접 생성하여 matmul로 처리하는 방식 사용
                # 간단하게 구현하기 위해 ConvND 대신 pointwise linear로 처리하는 헬퍼 사용
                pass

            if self.D <= 3:
                self.input_proj = conv_cls(c_expanded, self.d_model, kernel_size=1, dtype=get_complex_dtype(self.dtype_idx)).to(device)
            else:
                # Fallback for >3D: 1x1 convolution is effectively MatMul over channel dim
                # (B, C_in, ...) -> permute -> (..., C_in) -> Linear -> (..., C_out) -> permute
                self.input_proj = nn.Linear(c_expanded, self.d_model, bias=True, dtype=get_complex_dtype(self.dtype_idx)).to(device)
                self.is_high_dim_proj = True

        
        # Projection logic
        if hasattr(self, 'is_high_dim_proj') and self.is_high_dim_proj:
            # Permute channels to last: (B, D1.., C)
            permute_order = [0] + list(range(2, 2 + self.D)) + [1]
            x = x_dost.permute(*permute_order)
            xdtype = x.dtype
            if xdtype == get_complex_dtype(32):
                with torch.amp.autocast(device_type=str(x.device)):
                    x = self.input_proj.to(dtype=torch.complex64)(x.to(dtype=torch.complex64)).to(dtype=xdtype)
            else:
                x = self.input_proj(x)
            # Permute back: (B, C, D1..)
            inv_order = [0, self.D + 1] + list(range(1, 1 + self.D))
            x = x.permute(*inv_order)
        else:
            xdtype = x_dost.dtype
            if xdtype == get_complex_dtype(32):
                with torch.amp.autocast(device_type=str(x_dost.device)):
                    x = self.input_proj.to(dtype=torch.complex64)(x_dost.to(dtype=torch.complex64)).to(dtype=xdtype)
            else:
                x = self.input_proj(x_dost)
        
        # 2. S9 Backbone
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
            
        # 3. Global Pooling & Classification
        # 모든 공간 차원(Spatial dimensions)에 대해 평균
        # dims to reduce: range(-D, 0)
        reduce_dims = list(range(-self.D, 0))
        xdtype = x.dtype
        if xdtype == get_complex_dtype(32):
            with torch.amp.autocast(device_type=str(x.device)):
                x = x.to(torch.complex64).mean(dim=reduce_dims).to(xdtype) # (B, d_model)
        else:
            x = x.mean(dim=reduce_dims) # (B, d_model)
        
        x_mag = torch.abs(x) 
        
        x_final = self.norm(x_mag)
        logits = self.classifier(x_final)
        
        return logits


class MultiheadS9ClassifierModelExample(nn.Module):
    """
    N-Dimensional S9 Model Architecture.
    입력 데이터의 공간 차원(spatial_shape)에 따라 1D, 2D, 3D 등으로 자동 확장됩니다.
    
    Flow:
    Input (Real) -> ND-DOST -> Complex Features 
    -> Complex Linear Projection -> Stack of ND-S9 Layers -> Magnitude Pooling -> Classifier
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
        eps: float = 1e-6
    ):
        super().__init__()
        
        self.dtype_idx: FPDTypeIdx = dtype_idx

        self.spatial_shape = spatial_shape
        self.D = len(spatial_shape) # Dimension (1, 2, 3, ...)
        
        # 1. Non-learnable Preprocessor (Multidimensional DOST)
        self.dost = DOST(self.D)
        
        self.input_proj = None 
        self.d_model = d_model
        
        # 2. S9 Layers (Complex Domain, N-Dimensional)
        self.layers = nn.ModuleList(
            [
                MultiheadS9Layer(
                    d_model=d_model,
                    spatial_dims=self.D,
                    dtype_idx=self.dtype_idx,
                    gen_activation=StableModReLU,
                    head_channels=(in_channels,),
                    n_heads=n_heads,
                    eps=eps
                )
                for _ in range(n_layers)
            ]
        )
        
        # 3. Output Head
        self.norm = nn.LayerNorm(d_model, dtype = get_float_dtype(self.dtype_idx))
        self.classifier = nn.Linear(d_model, num_classes, dtype = get_float_dtype(self.dtype_idx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D1, D2, ...) Real valued tensor
        """
        # 1. Preprocessing (DOST)
        # x_dost: (B, C_expanded, D1, D2, ...)
        x_dost = self.dost(x) 
        
        # Initialize input projection lazily
        if self.input_proj is None:
            c_expanded = x_dost.shape[1]
            device = x.device
            # Conv1d, Conv2d, Conv3d 중 차원에 맞는 것 선택
            if self.D == 1:
                conv_cls = nn.Conv1d
            elif self.D == 2:
                conv_cls = nn.Conv2d
            elif self.D == 3:
                conv_cls = nn.Conv3d
            else:
                # 4차원 이상은 nn.Linear로 처리 (채널 축을 마지막으로 보낸 뒤)하거나 Custom 구현 필요
                # 여기서는 1x1 Conv 효과를 내기 위해 단순 Linear Projection 사용 권장하나,
                # 코드 일관성을 위해 weight를 직접 생성하여 matmul로 처리하는 방식 사용
                # 간단하게 구현하기 위해 ConvND 대신 pointwise linear로 처리하는 헬퍼 사용
                pass

            if self.D <= 3:
                self.input_proj = conv_cls(c_expanded, self.d_model, kernel_size=1, dtype=get_complex_dtype(self.dtype_idx)).to(device)
            else:
                # Fallback for >3D: 1x1 convolution is effectively MatMul over channel dim
                # (B, C_in, ...) -> permute -> (..., C_in) -> Linear -> (..., C_out) -> permute
                self.input_proj = nn.Linear(c_expanded, self.d_model, bias=True, dtype=get_complex_dtype(self.dtype_idx)).to(device)
                self.is_high_dim_proj = True

        # Projection logic
        if hasattr(self, 'is_high_dim_proj') and self.is_high_dim_proj:
            # Permute channels to last: (B, D1.., C)
            permute_order = [0] + list(range(2, 2 + self.D)) + [1]
            x = x_dost.permute(*permute_order)
            x = self.input_proj(x)
            # Permute back: (B, C, D1..)
            inv_order = [0, self.D + 1] + list(range(1, 1 + self.D))
            x = x.permute(*inv_order)
        else:
            x = self.input_proj(x_dost)
        
        # 2. S9 Backbone
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
            
        # 3. Global Pooling & Classification
        # 모든 공간 차원(Spatial dimensions)에 대해 평균
        # dims to reduce: range(-D, 0)
        reduce_dims = list(range(-self.D, 0))
        x = x.mean(dim=reduce_dims) # (B, d_model)
        
        x_mag = torch.abs(x) 
        
        x_final = self.norm(x_mag)
        logits = self.classifier(x_final)
        
        return logits


class BiaffineS9ClassifierModelExample(nn.Module):
    """
    N-Dimensional S9 Model Architecture.
    입력 데이터의 공간 차원(spatial_shape)에 따라 1D, 2D, 3D 등으로 자동 확장됩니다.
    
    Flow:
    Input (Real) -> ND-DOST -> Complex Features 
    -> Complex Linear Projection -> Stack of ND-S9 Layers -> Magnitude Pooling -> Classifier
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
        channel_embed_dim: int = 16
    ):
        super().__init__()
        
        self.dtype_idx: FPDTypeIdx = dtype_idx

        self.spatial_shape = spatial_shape
        self.D = len(spatial_shape) # Dimension (1, 2, 3, ...)
        
        # 1. Non-learnable Preprocessor (Multidimensional DOST)
        self.dost = DOST(self.D)
        
        self.input_proj = None 
        self.d_model = d_model
        
        # 2. S9 Layers (Complex Domain, N-Dimensional)
        self.layers = nn.ModuleList(
            [
                BiaffineS9Layer(
                    d_model=d_model,
                    spatial_dims=self.D,
                    dtype_idx=self.dtype_idx,
                    gen_activation=StableModReLU,
                    latent_channels=(in_channels,),
                    n_heads=n_heads,
                    eps=eps,
                    channel_embed_dim=channel_embed_dim
                )
                for _ in range(n_layers)
            ]
        )
        
        # 3. Output Head
        self.norm = nn.LayerNorm(d_model, dtype = get_float_dtype(self.dtype_idx))
        self.classifier = nn.Linear(d_model, num_classes, dtype = get_float_dtype(self.dtype_idx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D1, D2, ...) Real valued tensor
        """
        # 1. Preprocessing (DOST)
        # x_dost: (B, C_expanded, D1, D2, ...)
        x_dost = self.dost(x) 
        
        # Initialize input projection lazily
        if self.input_proj is None:
            c_expanded = x_dost.shape[1]
            device = x.device
            # Conv1d, Conv2d, Conv3d 중 차원에 맞는 것 선택
            if self.D == 1:
                conv_cls = nn.Conv1d
            elif self.D == 2:
                conv_cls = nn.Conv2d
            elif self.D == 3:
                conv_cls = nn.Conv3d
            else:
                # 4차원 이상은 nn.Linear로 처리 (채널 축을 마지막으로 보낸 뒤)하거나 Custom 구현 필요
                # 여기서는 1x1 Conv 효과를 내기 위해 단순 Linear Projection 사용 권장하나,
                # 코드 일관성을 위해 weight를 직접 생성하여 matmul로 처리하는 방식 사용
                # 간단하게 구현하기 위해 ConvND 대신 pointwise linear로 처리하는 헬퍼 사용
                pass

            if self.D <= 3:
                self.input_proj = conv_cls(c_expanded, self.d_model, kernel_size=1, dtype=get_complex_dtype(self.dtype_idx)).to(device)
            else:
                # Fallback for >3D: 1x1 convolution is effectively MatMul over channel dim
                # (B, C_in, ...) -> permute -> (..., C_in) -> Linear -> (..., C_out) -> permute
                self.input_proj = nn.Linear(c_expanded, self.d_model, bias=True, dtype=get_complex_dtype(self.dtype_idx)).to(device)
                self.is_high_dim_proj = True

        # Projection logic
        if hasattr(self, 'is_high_dim_proj') and self.is_high_dim_proj:
            # Permute channels to last: (B, D1.., C)
            permute_order = [0] + list(range(2, 2 + self.D)) + [1]
            x = x_dost.permute(*permute_order)
            x = self.input_proj(x)
            # Permute back: (B, C, D1..)
            inv_order = [0, self.D + 1] + list(range(1, 1 + self.D))
            x = x.permute(*inv_order)
        else:
            x = self.input_proj(x_dost)
        
        # 2. S9 Backbone
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
            
        # 3. Global Pooling & Classification
        # 모든 공간 차원(Spatial dimensions)에 대해 평균
        # dims to reduce: range(-D, 0)
        reduce_dims = list(range(-self.D, 0))
        x = x.mean(dim=reduce_dims) # (B, d_model)
        
        x_mag = torch.abs(x) 
        
        x_final = self.norm(x_mag)
        logits = self.classifier(x_final)
        
        return logits

# ---------------------------------------------------------------------------
# ARS9 classifier examples (real I/O, no DOST, conjugate-pair complex internal)
# ---------------------------------------------------------------------------

def _make_ars9_activation(d_model: int, eps: float, dtype_idx: FPDTypeIdx) -> nn.Module:
    """Factory for real-valued activation compatible with gen_activation signature."""
    return ThASh()


class _ARS9ClassifierBase(nn.Module):
    """Shared base for ARS9-based (real I/O, conjugate-pair) classifier examples."""

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
            self.input_proj = conv_cls(
                self.in_channels, self.d_model, kernel_size=1, dtype=f_dtype
            ).to(device)
        else:
            self.input_proj = nn.Linear(
                self.in_channels, self.d_model, bias=True, dtype=f_dtype
            ).to(device)
            self.is_high_dim_proj = True

    def _project_input(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "is_high_dim_proj") and self.is_high_dim_proj:
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


class ARS9ClassifierModelExample(_ARS9ClassifierBase):
    """ARS9 classifier: Real input -> Projection -> ARS9 backbone -> Classifier."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int,
        num_classes: int,
        spatial_shape: Tuple[int, ...],
        dtype_idx: FPDTypeIdx = 64,
        eps: float = 1e-6,
    ):
        super().__init__(in_channels, d_model, num_classes, spatial_shape, dtype_idx)
        self.layers = nn.ModuleList([
            ARS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=_make_ars9_activation,
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])


class MultiheadARS9ClassifierModelExample(_ARS9ClassifierBase):
    """Multi-head ARS9 classifier: Real input -> MultiheadARS9 -> Classifier."""

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
            MultiheadARS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=_make_ars9_activation,
                n_heads=n_heads,
                head_channels=(in_channels,),
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])


class BiaffineARS9ClassifierModelExample(_ARS9ClassifierBase):
    """Biaffine ARS9 classifier: Real input -> BiaffineARS9 -> Classifier."""

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
            BiaffineARS9Layer(
                d_model=d_model,
                spatial_dims=self.D,
                gen_activation=_make_ars9_activation,
                latent_channels=(in_channels,),
                n_heads=n_heads,
                channel_embed_dim=channel_embed_dim,
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])
