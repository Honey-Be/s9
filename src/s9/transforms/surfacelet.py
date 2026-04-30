import math
import torch
from typing import Dict, List, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ypsilon_torch.blocks.transforms.real_complex.surfacelet import *