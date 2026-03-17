import torch
import torch.nn as nn
import torch.optim as optim
from s9.examples import BiaffineS9ClassifierModelExample
from s9.transforms.curvelet import FastCurveletTransform2D, InverseFastCurveletTransform2D
import pytest

from typing import Literal

from s9.base import get_float_dtype, get_complex_dtype, FPDTypeIdx

from . import SPATIAL_SHAPES

@pytest.mark.parametrize('dtype_idx', [64,128])
def test_curvelet(dtype_idx: FPDTypeIdx):
    print(f"=== Fast Curvelet Transform / Inverse Fast Curvelet Transform Correctness Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    BATCH_SIZE = 2
    CHANNELS = 3
    spatial_shape = SPATIAL_SHAPES[1]
    x = torch.rand([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=get_float_dtype(dtype_idx))
    fcurvt = FastCurveletTransform2D(dtype_idx=dtype_idx)
    ifcurvt = fcurvt.get_inverse_transform()
    y = fcurvt(x)
    x2 = ifcurvt(y)
    torch.testing.assert_close(x, x2, check_stride=False)
