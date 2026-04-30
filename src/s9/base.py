import warnings

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import final, Literal, Generic, TypedDict, Unpack, TypeVar, Protocol, Self

try:
    from typing import override
except:
    from typing_extensions import override

from ypsilon_torch import NonLearnableProcessorBase, NonLearnableSynchronizedProcessorBase, FPDTypeIdx, FLOAT_DTYPES_DICT, COMPLEX_DTYPES_DICT, EXPERIMENTAL_DTYPE_IDXS, get_complex_dtype, get_float_dtype

from ypsilon_torch.blocks.activations import ComplexActivationFunctionBase, RealActivationFunctionBase