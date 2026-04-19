"""Quantized kernel cache for eval-mode inference."""

from __future__ import annotations

from typing import Optional

import torch

from s9.quantization.quantizers import symmetric_per_tensor_quantize


class QuantizedKernelCache:
    """Caches a quantized global kernel for reuse during inference.

    When the layer is in eval mode, the kernel is computed once per spatial
    shape, quantized to int8, and stored. Subsequent calls reuse the cached
    quantized kernel.
    """

    def __init__(self, bits: int = 8) -> None:
        self.bits: int = bits
        self._cache_key: Optional[tuple[int, ...]] = None
        self._k_quantized: Optional[torch.Tensor] = None
        self._scale: Optional[torch.Tensor] = None

    def get_or_compute(
        self,
        k_global: torch.Tensor,
        spatial_shapes: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(k_quantized, scale)`` from cache or compute fresh."""
        if self._cache_key == spatial_shapes and self._k_quantized is not None:
            return self._k_quantized, self._scale  # type: ignore[return-value]

        is_complex = k_global.is_complex()
        k_q, scale = symmetric_per_tensor_quantize(k_global, self.bits, is_complex)
        self._cache_key = spatial_shapes
        self._k_quantized = k_q.detach()
        self._scale = scale.detach()
        return k_q, scale

    def invalidate(self) -> None:
        self._cache_key = None
        self._k_quantized = None
        self._scale = None
