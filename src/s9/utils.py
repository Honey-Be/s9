import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_dropout(z: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    if (not training) or p == 0.0:
        return z
    if p == 1.0:
        return torch.zeros_like(z)
    q: float = 1.0 - p
    # real-valued mask with the same complex-element shape
    if z.is_complex():
        mask: torch.Tensor = F.dropout(torch.ones_like(z.real), p=p, training=training)
    else:
        mask: torch.Tensor = F.dropout(torch.ones_like(z), p=p, training=training)
    return z * mask