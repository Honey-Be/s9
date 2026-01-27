import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import final

COMPLEX_DTYPES_DICT: dict[Literal[32, 64, 128], torch.dtype] = {
    32: torch.complex32,
    64: torch.complex64,
    128: torch.complex128
}

FLOAT_DTYPES_DICT: dict[Literal[32, 64, 128], torch.dtype] = {
    32: torch.float16,
    64: torch.float32,
    128: torch.float64
}

class NonLearnableProcessorBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """실제 변환 로직을 구현하는 추상 메서드"""
        pass

    def is_valid_input(self, x: torch.Tensor) -> bool:
        """
        입력 텐서의 유효성을 검사하는 메서드.
        기본적으로 True를 반환하며, 필요시 하위 클래스에서 오버라이드.
        """
        return True

    @final
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch 모듈의 forward 메서드.
        입력 검증 후 transform을 호출합니다.
        """
        if not self.is_valid_input(x):
            raise ValueError(f"Invalid input shape or dtype: {x.shape}, {x.dtype}")
        return self.transform(x)

class ComplexActivationFunctionBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: Literal[32, 64, 128] = 64):
        super().__init__()

        self.bias: nn.Parameter = nn.Parameter(torch.zeros(features, dtype=FLOAT_DTYPES_DICT[dtype_idx]))
        self.eps: float = eps
    
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

    