# h9 Branch: H9 — HAMSA improvements inspired by s9 package.

Re-formulation of HAMSA (Patro & Agneeswaran, 2026, arXiv:2604.14724) using
S9's Warped DOST preprocessor and complex-domain multidimensional SSM
infrastructure. **No patch embedding, no positional embedding, no tokenization** —
images are processed end-to-end as multidimensional signals.

## Why h9

| | Original HAMSA | h9 (this branch) |
|---|---|---|
| Spectral preprocessor | Learnable FFT + complex kernel | Warped DOST (non-learnable) + complex kernel |
| Token representation | $(B, L, D)$ flattened tokens | $(B, D', H, W)$ multidim signal |
| Positional info | Implicit in token order | **Implicit in tensor shape (no PE)** |
| Phase utilization | Magnitude-only gating | **sin/cos phase decomposition** |
| Cross-resolution | Retrain or adapt | **Refit DOST boundaries only** |
| Time-frequency localization | None (FFT) | Yes (Stockwell-style) |

## Architecture at a glance

```
X ──► Stem(1×1) ──► Warped DOST ──► [ HSS Block ] × L ──► Inverse DOST ──► GAP ──► Linear ──► ŷ
                    (calibrated)     ↑                      │
                                     ╰── (B, D', H, W)     ╰── (B, D, H, W)
                                         complex throughout      backbone output
```

Each HSS block: `Dual-Project ► Phase-Aware SPN ► Spectral Kernel ► SAGU ► Out-Gate ► FFN`,
all in complex DOST coefficient domain, all channel-mixing only (zero spatial mixing
inside the block). See `DESIGN-H9.md` for full math.

## Quickstart

### Installation

```bash
git checkout h9
pip install -e ".[cu130,h9]"   # or [cpu], [cu128], [cu126]
```

### Minimal usage

```python
import torch
from s9.h9.examples import H9ClassifierModelExample

model = H9ClassifierModelExample(
    in_channels=3,
    d_model=64,
    n_layers=8,
    num_classes=10,
    n_per_axis=2,        # DOST band count (D' = d_model * n_per_axis^2)
    spatial_dims=2,
)

# Calibrate Warped DOST on a sample of training data
calib_batch = next(iter(train_loader))[0]
model.calibrate(calib_batch)

# Train normally (no positional embedding setup needed)
for x, y in train_loader:
    logits = model(x)             # x.shape = (B, 3, H, W) — H, W can be anything calibrated
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
```

### Cross-resolution inference

The same trained model handles different resolutions after refitting Warped DOST:

```python
# Trained at 32×32. Now evaluate at 64×64.
f = model.fitter
for x_chunk, _ in eval_loader_64:
    f.accumulate(x_chunk)
f.finalize()

logits = model(x_64)              # works without any retraining
```

### Backbone-only usage (without classifier head)

`H9ClassifierModelExample` is a convenience wrapper. For tasks other than
classification (detection, segmentation, feature extraction, etc.), compose
the building blocks directly:

```python
import torch
from torch import nn
from s9.transforms.warped_dost import WarpedDOST
from s9.h9 import HSSBlock

class H9Backbone(nn.Module):
    def __init__(self, in_channels=3, d_model=64, n_layers=8, n_per_axis=3):
        super().__init__()
        self.d_model = d_model
        self.d_prime = d_model * n_per_axis ** 2

        self.stem = nn.Conv2d(in_channels, d_model, 1)
        self.dost = WarpedDOST(D=2, n_per_axis=n_per_axis)
        self.blocks = nn.ModuleList([
            HSSBlock(d_model=d_model, n_per_axis=n_per_axis)
            for _ in range(n_layers)
        ])

    def calibrate(self, x):
        with torch.no_grad():
            self.dost.fit(self.stem(x))

    def forward(self, x):
        u = self.stem(x)                           # (B, D, H, W) real
        z = self.dost(u)                           # (B, D', H, W) complex
        for block in self.blocks:
            z = block(z)                           # shape preserved
        inv = self.dost.get_inverse_transform()
        return inv(z)                              # (B, D, H, W) real
```

The output is a spatial feature map of shape `(B, D, H, W)` — attach any
downstream head without going through GAP. The key point: **Inverse DOST
collapses `D'` back to `D`**, so the head sees the original channel dim,
not the expanded one.

For tasks that benefit from multi-scale spectral features, you can also
skip the Inverse DOST and work directly in the DOST coefficient domain
`(B, D', H, W)` complex — but note that downstream modules must then
handle complex tensors.

### Streaming calibration (memory-constrained)

```python
f = model.fitter
for x_chunk, _ in big_loader:
    f.accumulate(x_chunk)
f.finalize()
```

Buffer size is `O(H + W)` floats — independent of dataset size.

## Module map

| Module | Purpose | Spec section |
|---|---|---|
| `s9.h9.HSSBlock` | One full block (Dual-Proj + SASS + Out-Gate + FFN) | DESIGN-H9 §5 |
| `s9.h9.PhaseAwareSPN` | Phase-aware SpectralPulseNet (sin/cos gating) | DESIGN-H9 §6.1 |
| `s9.h9.SpectralKernel` | Learnable complex per-band kernel | DESIGN-H9 §6.2 |
| `s9.h9.DOSTDomainSAGU` | SAGU adapted for complex coefficients | DESIGN-H9 §6.4 |
| `s9.h9.SASS` | Composition: SPN ∘ Kernel ∘ SAGU | DESIGN-H9 §6 |
| `s9.h9.ComplexLayerNorm` | Magnitude-based LN preserving phase | DESIGN-H9 §5.3 |
| `s9.h9.ComplexFFN` | Two-layer MLP, channel-only | DESIGN-H9 §5.4 |
| `s9.h9.DualAttribution` | Non-intrusive attribution capture | DESIGN-H9 §8 |
| `s9.h9.examples.H9ClassifierModelExample` | End-to-end classifier (GAP + Linear head) | DESIGN-H9 §2 |

For backbone-only usage (detection, segmentation, etc.), compose `WarpedDOST` +
`HSSBlock` directly — see [Backbone-only usage](#backbone-only-usage-without-classifier-head) above.

## Resolution invariance — what it means and what it doesn't

**It means**: model weights are reusable across resolutions. After training at one
resolution, refitting Warped DOST boundaries on a new-resolution calibration batch
is the only step needed before inference at the new resolution.

**It does NOT mean**: training-time data augmentation handles arbitrary resolution.
You still need an appropriate calibration sample for each resolution at inference.

**Compute scaling**: parameters are constant in $(H, W)$, but activations grow
linearly with $H \cdot W$. So the **memory** cost grows; only the **parameter count**
is invariant.

## Dual attribution

Per-block, per-(channel, position) activation can be captured for analysis:

```python
model.attribution_enabled = True
_ = model(x)
attr = model.get_attribution()    # dict[block_idx -> tensor of shape (B, D', H, W)]

from s9.h9.attribution import spatial_view, spectral_view, joint_view
heatmap = spatial_view(attr[5])              # (B, H, W) — last block, where it looks
profile = spectral_view(attr[5])             # (B, D')   — which bands matter
joint = joint_view(attr[5], n=2, d=64)       # (B, n, n, H, W) — full dual attribution
```

Attribution adds **zero overhead during training** (controlled by a boolean flag).

## Out of scope (this branch)

The following are deliberately deferred — see `DESIGN-H9.md` §12:
- Spectral MoE (separate paper)
- Wavelet-coefficient initialization
- Hierarchical (CNN-style) variant for ImageNet-scale efficiency
- Video extension with temporal Warped DOST
- Quantization variants (parallel to QS9 family)

## References

1. Patro, B. N., & Agneeswaran, V. S. (2026). HAMSA: Scanning-Free Vision State Space Models via SpectralPulseNet. arXiv:2604.14724.
2. S9 main README — for shared infrastructure (DOST, complex activations, etc.)
3. `README-WARPED-DOST.md` — for the calibration workflow underlying h9.
4. `DESIGN-H9.md` — full technical spec.

## License

Same as parent repository: GNU LGPLv2.1+. See `LICENSE.txt`.
