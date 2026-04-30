from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, final, Self

import torch
import torch.nn as nn

try:
    from typing import override
except ImportError:
    from typing_extensions import override


from collections.abc import Sequence

from ypsilon_torch.blocks.transforms.real_complex.dost import *