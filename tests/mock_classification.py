import torch
import torch.nn as nn
import torch.optim as optim
from s8.examples import S8ClassifierModelExample
from s8.dost import DOST, IDOST
import pytest

from typing import Literal

from s8.base import FLOAT_DTYPES_DICT

SPATIAL_SHAPES: list[list[int]] = [
    [128],
    [32, 32],
    [8, 16, 16],
    [8, 8, 8, 8]
]

@pytest.mark.parametrize('D', [1,2,3,4])
@pytest.mark.parametrize('dtype_idx', [32,64,128])
def test_s8(D: int, dtype_idx: Literal[32, 64, 128]):
    print(f"=== S8 {D}D Classifier Model Example Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Settings
    BATCH_SIZE = 2
    CHANNELS = 3
    NUM_CLASSES = 5
    D_MODEL = 32
    LAYERS = 1

    spatial_shape = SPATIAL_SHAPES[D-1]
    model = S8ClassifierModelExample(
        in_channels=CHANNELS, d_model=D_MODEL, n_layers=LAYERS,
        num_classes=NUM_CLASSES, spatial_shape=spatial_shape,
        dtype_idx=dtype_idx
    ).to(device)

    input = torch.randn([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(device)
    output = model(input)
    print(f"   Input: {input.shape} -> Output: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print(f"   -> {D}D Pass Successful!")