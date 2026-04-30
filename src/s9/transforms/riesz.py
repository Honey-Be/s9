import torch
import torch.nn as nn
from s9.base import NonLearnableProcessorBase, FPDTypeIdx, get_complex_dtype, get_float_dtype
from s9.transforms.base import InvertibleTransformsBase

try:
    from typing import override
except ImportError:
    from typing_extension import override

from ypsilon_torch.blocks.transforms.real_complex.riesz import *