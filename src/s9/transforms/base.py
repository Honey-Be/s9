from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, final, TypeVar, Self, Generic

import torch
import torch.nn as nn

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from s9.base import NonLearnableProcessorBase, FPDTypeIdx, COMPLEX_DTYPES_DICT, FLOAT_DTYPES_DICT

class InvertibleTransformsBase[I: InvertibleTransformsBase[Self]](NonLearnableProcessorBase, ABC):
    @abstractmethod
    def get_inverse_transform(self) -> I:
        raise NotImplementedError()