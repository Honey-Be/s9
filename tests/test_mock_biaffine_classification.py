import torch
import torch.nn as nn
import torch.optim as optim
from s9.examples import BiaffineS9ClassifierModelExample
from ypsilon_torch.blocks.transforms.real_complex.dost import DOST, IDOST
import pytest

from typing import Literal

from ypsilon_torch import FLOAT_DTYPES_DICT, FPDTypeIdx

from . import SPATIAL_SHAPES

@pytest.mark.parametrize('D', [1,2,3,4])
@pytest.mark.parametrize('dtype_idx', [64,128])
@pytest.mark.parametrize('H', [1,2,4,8])
def test_mock_biaffine_classification(D: int, dtype_idx: FPDTypeIdx, H: int):
    print(f"=== S9 {D}D Classifier Model Example Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Settings
    BATCH_SIZE = 2
    CHANNELS = 3
    NUM_CLASSES = 5
    D_MODEL = 32
    LAYERS = 1

    spatial_shape = SPATIAL_SHAPES[D-1]
    model = BiaffineS9ClassifierModelExample(
        in_channels=CHANNELS, d_model=D_MODEL, n_layers=LAYERS,
        num_classes=NUM_CLASSES, spatial_shape=spatial_shape,
        dtype_idx=dtype_idx, n_heads=H
    ).to(device)

    input = torch.randn([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(device=device)
    output = model(input)
    print(f"   Input: {input.shape} -> Output: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print(f"   -> {D}D Pass Successful!")