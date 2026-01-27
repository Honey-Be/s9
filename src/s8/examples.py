import torch
import torch.nn as nn
from typing import Tuple
from s8.dost import DOST
from s8.modules import S8Layer, StableModReLU
from typing import Tuple, List, Literal

class S8ClassifierModelExample(nn.Module):
    """
    N-Dimensional S8 Model Architecture.
    입력 데이터의 공간 차원(spatial_shape)에 따라 1D, 2D, 3D 등으로 자동 확장됩니다.
    
    Flow:
    Input (Real) -> ND-DOST -> Complex Features 
    -> Complex Linear Projection -> Stack of ND-S8 Layers -> Magnitude Pooling -> Classifier
    """
    def __init__(self, in_channels: int, d_model: int, n_layers: int, num_classes: int, spatial_shape: Tuple[int, ...], dtype_idx: Literal[32, 64, 128] = 64):
        super().__init__()
        
        self.spatial_shape = spatial_shape
        self.D = len(spatial_shape) # Dimension (1, 2, 3, ...)
        
        # 1. Non-learnable Preprocessor (Multidimensional DOST)
        self.dost = DOST(D=self.D)
        
        self.input_proj = None 
        self.d_model = d_model
        
        # 2. S8 Layers (Complex Domain, N-Dimensional)
        self.layers = nn.ModuleList([
            S8Layer[StableModReLU](d_model=d_model, spatial_shapes=self.spatial_shape, dtype_idx=dtype_idx, gen_activation=StableModReLU)
            for _ in range(n_layers)
        ])
        
        # 3. Output Head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

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
                self.input_proj = conv_cls(c_expanded, self.d_model, kernel_size=1, dtype=torch.complex64).to(device)
            else:
                # Fallback for >3D: 1x1 convolution is effectively MatMul over channel dim
                # (B, C_in, ...) -> permute -> (..., C_in) -> Linear -> (..., C_out) -> permute
                self.input_proj = nn.Linear(c_expanded, self.d_model, bias=True, dtype=torch.complex64).to(device)
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
        
        # 2. S8 Backbone
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