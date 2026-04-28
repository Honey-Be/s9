# CLAUDE.md — h9 Branch Implementation Guide

> This file is read by Claude Code at the start of each session.
> It captures the **conventions, gotchas, and workflow** specific to this branch.

## Branch context

- **Branch name**: `h9`
- **Parent**: forked from `main` after v0.5.0 of s9
- **Scope**: implement H9 (HAMSA + Warped DOST) per `DESIGN-H9.md`
- **Phase**: Phase 1+2 only (no MoE, no hierarchical, no video — see DESIGN-H9.md §12)

## First read order

1. `DESIGN-H9.md` — **source of truth** for math and shapes
2. `pyproject.toml` — extras and dependencies
2. `README-H9.md` — quickstart and API surface
3. This file (CLAUDE.md) — conventions, dev workflow
4. Existing `README.md`, `README-WARPED-DOST.md` — to understand s9 patterns
5. Existing `src/s9/modules/`, `src/s9/transforms/warped_dost.py` — to mirror style

If any conflict arises between this file and `DESIGN-H9.md`, **DESIGN-H9.md wins**.
Update DESIGN-H9.md if you discover a needed deviation; do NOT silently diverge.

## Build / Test / Lint

```bash
# Install (CPU dev)
pip install -e ".[cpu]"

# Run h9-specific tests
pytest tests/test_h9_components.py -v
pytest tests/test_h9_resolution_invariance.py -v
pytest tests/test_h9_dual_attribution.py -v

# Run all tests (h9 must not break existing s9 tests)
pytest

# Type check (if mypy is set up in the repo)
mypy src/h9 src/h9/examples
```

## Project layout (h9 additions)

```
src/
├── h9/                    # NEW
│   ├── __init__.py                # Public API: HSSBlock, PhaseAwareSPN, etc.
│   ├── spn.py                     # Phase-Aware SpectralPulseNet
│   ├── spectral_kernel.py         # Learnable complex kernel
│   ├── sagu.py                    # DOST-domain SAGU
│   ├── sass.py                    # SASS = SPN ∘ Kernel ∘ SAGU
│   ├── hss_block.py               # Full block: dual proj + SASS + out gate + FFN
│   ├── components.py              # ComplexLayerNorm, ComplexFFN
│   └── attribution.py             # Dual attribution hooks and views
│   └── examples/                   # NEW
│       ├── __init__.py
│       └── h9_classifier.py           # H9ClassifierModelExample
└── ...                             # (existing s9 modules — DO NOT MODIFY)

tests/
├── test_h9_components.py          # NEW: unit tests for SPN, kernel, SAGU
├── test_h9_resolution_invariance.py  # NEW: cross-resolution test
└── test_h9_dual_attribution.py    # NEW: attribution capture
```

## Hard rules

- Do not modify any file outside `src/h9/`, `src/h9/examples/`, `tests/test_h9_*.py`, and the design docs in repo root. The h9 work must be **fully additive** and must not break the existing s9 API.
- The h9 work should be activated only when `h9` extra is specified, for which the version suffix is `+h9`.

## s9 conventions to mirror

When reading existing s9 code (e.g. `S9Layer`, `RS9Layer`, `ARS9Layer`),
notice these patterns and replicate:

### Constructor signature pattern
```python
def __init__(
    self,
    d_model: int,
    spatial_dims: int,
    gen_activation: type[nn.Module] = StableModReLU,
    init_mode: Literal["legacy", "hippo_n", "s4d_real", "gaussian"] = "gaussian",
    dtype_idx: Literal[32, 64] = 64,
    eps: float = 1e-8,
    ...
):
```

For h9 specifically:
- `init_mode` defaults to `"gaussian"` (HAMSA-style)
- We add `n_per_axis: int` (DOST band count)

### `dtype_idx` interpretation
- `dtype_idx=32`: real `float16` / complex `complex32`
- `dtype_idx=64`: real `float32` / complex `complex64`
- `dtype_idx=128`: real `float64` / complex `complex128`

### Activation factory pattern

s9 takes `gen_activation` as a factory, not an instance:
```python
def make_activation(d_model, eps, dtype_idx):
    return ThASh()
```

This lets the layer construct activations with knowledge of d_model.
Mirror this pattern for h9.

### Type hints

- Python 3.12+ syntax: `list[int]`, `dict[str, Tensor]`, `int | None`
- Use `Literal` for enumerated string args
- Use `torch.Tensor` directly (not `Tensor` from `torch`) in annotations to be unambiguous

## Common pitfalls

### Pitfall 1: Treating `Z` as `(B, L, D)`

This is the original HAMSA pattern. **DO NOT** do this in h9.
All h9 tensors keep shape `(B, D', H, W)` throughout the HSS blocks.

If you find yourself writing `.flatten(-2, -1)` or `rearrange(z, 'b c h w -> b (h w) c')`
inside an HSS block, **stop and re-read DESIGN-H9.md §5**.

### Pitfall 2: Spatial mixing inside HSS

HSS blocks do **NOT** mix spatial positions. All weights are channel-only,
broadcast over `(H, W)`. Use `einsum('bchw,cd->bdhw', z, w)` or the equivalent.

The only spatial mixing in the entire architecture is:
- Inside Warped DOST (Φ) — non-learnable
- Inside Inverse Warped DOST (Φ⁻¹) — non-learnable
- Inside Global Average Pool (head)

### Pitfall 3: Gradient through magnitude at zero

`|U| = sqrt(Re(U)² + Im(U)²)` has gradient singularity at `U = 0`.
Always use `m = (|U|² + eps).sqrt()` or equivalently
`m = torch.clamp(|U|, min=eps)` for any subsequent division.

The `cos_θ = Re(U) / m`, `sin_θ = Im(U) / m` computation is the most likely
place for NaN gradients. Test with `torch.autograd.gradcheck` on small inputs.

### Pitfall 4: Complex `nn.Linear`

PyTorch's `nn.Linear` does support complex dtypes in recent versions, but the
behavior may surprise. Prefer **explicit `nn.Parameter` + `einsum`** for clarity
in the SASS core. Use `nn.Linear` only for unambiguous cases (e.g., real linear
on real tensor).

### Pitfall 5: Forgetting to register parameters

`torch.tensor(...)` creates a regular tensor that won't be saved/loaded with
the module. Always use `nn.Parameter(...)` for learnable values, and the kernel
real/imag parts must each be a separate `nn.Parameter`.

## Calibration workflow (for examples)

The classifier MUST follow the WarpedDOST calibration pattern:

```python
class H9ClassifierModelExample(nn.Module):
    def calibrate(self, x: Tensor) -> None:
        """One-shot calibration."""
        u = self.stem(x)
        self.dost.fit(u)

    @property
    def fitter(self):
        """Streaming fitter."""
        return _ClassifierFitter(self)

class _ClassifierFitter:
    def __init__(self, model):
        self._model = model
        self._dost_fitter = model.dost.fitter

    def accumulate(self, x):
        u = self._model.stem(x)
        self._dost_fitter.accumulate(u)

    def finalize(self):
        self._dost_fitter.finalize()
```

Reference: see `s9.examples_warped.WarpedS9ClassifierModelExample` for the
exact pattern.

## Test-driven discipline

For each new module, write the test first (TDD). Tests live in
`tests/test_h9_*.py`. The test scaffolding is provided — fill in implementation
to make tests pass.

Each module's PR-ready criteria:
1. All tests in `tests/test_h9_*.py` pass
2. No existing `tests/` are broken
3. Module has a docstring referencing the relevant `DESIGN-H9.md` section
4. Type-checked clean (if mypy configured)

## When to ask for human review

Stop and ask the human (병익) before:
- Diverging from any spec in DESIGN-H9.md
- Adding learnable spatial mixing (would violate the no-positional-embedding principle)
- Choosing a different default for `n_per_axis` than 2
- Modifying any file outside the h9 scope listed above
- Encountering numerical instabilities not addressed by the eps strategies in DESIGN-H9.md

## Implementation order (suggested)

1. `components.py` (ComplexLayerNorm, ComplexFFN) — utilities
2. `spectral_kernel.py` (SpectralKernel) — simplest learnable module
3. `spn.py` (PhaseAwareSPN) — core innovation, write tests first
4. `sagu.py` (DOSTDomainSAGU)
5. `sass.py` (SASS) — composition of 2, 3, 4
6. `hss_block.py` (HSSBlock) — wraps SASS + dual proj + out gate + FFN
7. `attribution.py` (DualAttribution) — non-intrusive hooks
8. `h9_classifier.py` (H9ClassifierModelExample) — full integration
9. Final integration tests in `test_h9_resolution_invariance.py`

Each step should leave the test suite green before moving on.

## Notes on what is intentionally NOT here

- **No batch sampling strategy specified** — use whatever the existing s9 examples use
- **No optimizer/LR schedule** — out of scope for h9 module implementation; that's training-script territory
- **No data loaders** — same; expect users to use standard torchvision loaders
- **No CUDA-specific optimizations** — first get correctness, optimize later

## End-of-session checklist

Before considering a session complete:
- [ ] All new code matches `DESIGN-H9.md` (re-read it!)
- [ ] All h9 tests pass
- [ ] No existing s9 tests broken
- [ ] Type hints present on all public APIs
- [ ] Docstrings reference DESIGN-H9.md section numbers where relevant
- [ ] No print/debug statements left in code
- [ ] Commit message references the DESIGN-H9.md section(s) implemented
