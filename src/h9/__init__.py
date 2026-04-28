"""h9: HAMSA improvements inspired by s9 package.

Re-formulation of HAMSA (arXiv:2604.14724) using s9's Warped DOST preprocessor
and complex-domain multidimensional infrastructure. No patch embedding, no
positional embedding, no tokenization.

See ``DESIGN-H9.md`` (in repo root) for the architecture specification.
See ``CLAUDE.md`` (in repo root) for implementation conventions.

Public API
----------

Core building blocks:
    * :class:`PhaseAwareSPN` -- Phase-aware SpectralPulseNet (DESIGN-H9 §6.1)
    * :class:`SpectralKernel` -- Per-band learnable complex kernel (DESIGN-H9 §6.2)
    * :class:`DOSTDomainSAGU` -- SAGU adapted for DOST coefficient domain (DESIGN-H9 §6.4)
    * :class:`SASS` -- Spectral Adaptive State Space (DESIGN-H9 §6)

Block-level:
    * :class:`HSSBlock` -- Full HSS block (DESIGN-H9 §5)

Utilities:
    * :class:`ComplexLayerNorm` -- Magnitude-based LN (DESIGN-H9 §5.3)
    * :class:`ComplexFFN` -- Channel-only complex MLP (DESIGN-H9 §5.4)

Attribution (eval-only):
    * :class:`DualAttribution` -- Non-intrusive attribution capture (DESIGN-H9 §8)
    * :func:`spatial_view`, :func:`spectral_view`, :func:`joint_view` --
      Aggregation utilities for captured attribution tensors.
"""

from __future__ import annotations

from h9.attribution import (
    DualAttribution,
    joint_view,
    spatial_view,
    spectral_view,
)
from h9.components import ComplexFFN, ComplexLayerNorm
from h9.hss_block import HSSBlock
from h9.sagu import DOSTDomainSAGU
from h9.sass import SASS
from h9.spectral_kernel import SpectralKernel
from h9.spn import PhaseAwareSPN

__all__ = [
    # Core SASS components
    "PhaseAwareSPN",
    "SpectralKernel",
    "DOSTDomainSAGU",
    "SASS",
    # Block
    "HSSBlock",
    # Utilities
    "ComplexLayerNorm",
    "ComplexFFN",
    # Attribution
    "DualAttribution",
    "spatial_view",
    "spectral_view",
    "joint_view",
]
