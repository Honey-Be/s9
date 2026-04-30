import torch
import torch.nn as nn
import torch.optim as optim
from s9.examples import BiaffineS9ClassifierModelExample
from ypsilon_torch.blocks.transforms.real_complex.surfacelet import FastSurfaceletTransform3D, InverseFastSurfaceletTransform3D
import pytest

from typing import Literal

from ypsilon_torch import get_float_dtype, get_complex_dtype, FPDTypeIdx

from . import SPATIAL_SHAPES

@pytest.mark.parametrize('dtype_idx', [64,128])
def test_surfacelet(dtype_idx: FPDTypeIdx):
    print(f"=== Fast Surfacelet Transform / Inverse Fast Surfacelet Transform Correctness Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    BATCH_SIZE = 2
    CHANNELS = 3
    spatial_shape = SPATIAL_SHAPES[2]
    x = torch.rand([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=get_float_dtype(dtype_idx))
    fsurft = FastSurfaceletTransform3D(dtype_idx=dtype_idx)
    ifsurft = fsurft.get_inverse_transform()
    y = fsurft(x)
    x2 = ifsurft(y)
    torch.testing.assert_close(x, x2, check_stride=False)