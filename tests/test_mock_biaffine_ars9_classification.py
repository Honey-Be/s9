import torch
import pytest

from ypsilon_torch import FLOAT_DTYPES_DICT, FPDTypeIdx
from s9.examples import BiaffineARS9ClassifierModelExample

from . import SPATIAL_SHAPES


@pytest.mark.parametrize('D', [1, 2, 3, 4])
@pytest.mark.parametrize('dtype_idx', [64, 128])
@pytest.mark.parametrize('H', [1, 2, 4, 8])
def test_mock_biaffine_ars9_classification(D: int, dtype_idx: FPDTypeIdx, H: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 2
    CHANNELS = 3
    NUM_CLASSES = 5
    D_MODEL = 32
    LAYERS = 1

    spatial_shape = SPATIAL_SHAPES[D - 1]
    model = BiaffineARS9ClassifierModelExample(
        in_channels=CHANNELS, d_model=D_MODEL, n_layers=LAYERS,
        n_heads=H, num_classes=NUM_CLASSES, spatial_shape=spatial_shape,
        dtype_idx=dtype_idx
    ).to(device)

    x = torch.randn([BATCH_SIZE, CHANNELS] + spatial_shape,
                     dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(device=device)
    output = model(x)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    assert not output.is_complex()
