import torch
import torch.nn as nn
import torch.optim as optim
from s9.examples import BiaffineS9ClassifierModelExample
from ypsilon_torch.blocks.transforms.real_complex.dost import DOST, IDOST
from ypsilon_torch.blocks.transforms.real_complex.sst import SynchronizedGenericSST

import pytest

from typing import Literal

from ypsilon_torch import get_float_dtype, get_complex_dtype, FPDTypeIdx


from . import SPATIAL_SHAPES

_sync_cache: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

@pytest.mark.parametrize('D', [1,2,3,4])
@pytest.mark.parametrize('dtype_idx', [64,128])
def test_dost_sst(D: int, dtype_idx: FPDTypeIdx):
    print(f"=== DOST + SST / IDOST + ISST Correctness Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    BATCH_SIZE = 2
    CHANNELS = 3
    spatial_shape = SPATIAL_SHAPES[D-1]
    x = torch.rand([BATCH_SIZE, CHANNELS] + spatial_shape, dtype=get_float_dtype(dtype_idx))
    dost = DOST(D)
    sst = SynchronizedGenericSST[DOST, IDOST](dost, dtype_idx)
    isst = sst.get_inverse_transform()
    cache_key=f"DOST{D}D_sst_{dtype_idx}"

    ys = sst.transform(x, cache_key = cache_key, sync_cache = _sync_cache)
    (x2, ) = isst.transform(*ys, cache_key = cache_key, sync_cache = _sync_cache)
    torch.testing.assert_close(x, x2, check_stride=False)