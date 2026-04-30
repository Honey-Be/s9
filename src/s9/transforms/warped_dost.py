from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod

import torch

try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from ypsilon_torch.blocks.transforms.real_complex.warped_dost import *