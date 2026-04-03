import torch
import torch.nn as nn
import pytest

from s9.base import FLOAT_DTYPES_DICT, FPDTypeIdx

from s9.contrib.examples import (
    GatedDeltaS9ClassifierExample,
    BiaffineGatedDeltaS9ClassifierExample,
    GatedDeltaRS9ClassifierExample,
    BiaffineGatedDeltaRS9ClassifierExample,
)

from . import SPATIAL_SHAPES


BATCH_SIZE = 2
CHANNELS = 3
NUM_CLASSES = 5
D_MODEL = 32
LAYERS = 1


@pytest.mark.parametrize('D', [1, 2, 3, 4])
@pytest.mark.parametrize('dtype_idx', [64, 128])
@pytest.mark.parametrize('H', [1, 2, 4])
def test_mock_gated_delta_s9_classification(D: int, dtype_idx: FPDTypeIdx, H: int):
    print(f"=== Gated Delta S9 {D}D Classifier Test (H={H}, dtype_idx={dtype_idx}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_shape = SPATIAL_SHAPES[D - 1]
    model = GatedDeltaS9ClassifierExample(
        in_channels=CHANNELS, d_model=D_MODEL, n_layers=LAYERS,
        num_classes=NUM_CLASSES, spatial_shape=spatial_shape,
        dtype_idx=dtype_idx, n_heads=H
    ).to(device)

    input = torch.randn([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(device)
    output = model(input)
    print(f"   Input: {input.shape} -> Output: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print(f"   -> Pass!")


@pytest.mark.parametrize('D', [1, 2, 3, 4])
@pytest.mark.parametrize('dtype_idx', [64, 128])
@pytest.mark.parametrize('H', [1, 2, 4])
def test_mock_biaffine_gated_delta_s9_classification(D: int, dtype_idx: FPDTypeIdx, H: int):
    print(f"=== Biaffine Gated Delta S9 {D}D Classifier Test (H={H}, dtype_idx={dtype_idx}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_shape = SPATIAL_SHAPES[D - 1]
    model = BiaffineGatedDeltaS9ClassifierExample(
        in_channels=CHANNELS, d_model=D_MODEL, n_layers=LAYERS,
        num_classes=NUM_CLASSES, spatial_shape=spatial_shape,
        dtype_idx=dtype_idx, n_heads=H
    ).to(device)

    input = torch.randn([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(device)
    output = model(input)
    print(f"   Input: {input.shape} -> Output: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print(f"   -> Pass!")


@pytest.mark.parametrize('D', [1, 2, 3, 4])
@pytest.mark.parametrize('dtype_idx', [64, 128])
@pytest.mark.parametrize('H', [1, 2, 4])
def test_mock_gated_delta_rs9_classification(D: int, dtype_idx: FPDTypeIdx, H: int):
    print(f"=== Gated Delta RS9 {D}D Classifier Test (H={H}, dtype_idx={dtype_idx}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_shape = SPATIAL_SHAPES[D - 1]
    model = GatedDeltaRS9ClassifierExample(
        in_channels=CHANNELS, d_model=D_MODEL, n_layers=LAYERS,
        num_classes=NUM_CLASSES, spatial_shape=spatial_shape,
        dtype_idx=dtype_idx, n_heads=H
    ).to(device)

    input = torch.randn([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(device)
    output = model(input)
    print(f"   Input: {input.shape} -> Output: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print(f"   -> Pass!")


@pytest.mark.parametrize('D', [1, 2, 3, 4])
@pytest.mark.parametrize('dtype_idx', [64, 128])
@pytest.mark.parametrize('H', [1, 2, 4])
def test_mock_biaffine_gated_delta_rs9_classification(D: int, dtype_idx: FPDTypeIdx, H: int):
    print(f"=== Biaffine Gated Delta RS9 {D}D Classifier Test (H={H}, dtype_idx={dtype_idx}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_shape = SPATIAL_SHAPES[D - 1]
    model = BiaffineGatedDeltaRS9ClassifierExample(
        in_channels=CHANNELS, d_model=D_MODEL, n_layers=LAYERS,
        num_classes=NUM_CLASSES, spatial_shape=spatial_shape,
        dtype_idx=dtype_idx, n_heads=H
    ).to(device)

    input = torch.randn([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(device)
    output = model(input)
    print(f"   Input: {input.shape} -> Output: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print(f"   -> Pass!")
