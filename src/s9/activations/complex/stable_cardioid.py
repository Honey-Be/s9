import math
import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override

from ypsilon_torch.blocks.activations.complex import StableComplexCardioid