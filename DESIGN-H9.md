# H9: Design Specification

> **Status**: Design document for h9 branch (H9 MVP, Phase 1+2).
> This is the **single source of truth** for the math and architecture.
> All implementation must be consistent with this document. If implementation
> reveals a needed deviation, **update this document first**, then code.

## 0. Branch Identity

**h9** = HAMSA (Patro & Agneeswaran, 2026, arXiv:2604.14724) re-formulated on
top of S9's complex-domain, multidimensional, non-learnable-preprocessor design.

Three departures from original HAMSA:
1. FFT replaced by Warped DOST (time-frequency localization + resolution invariance)
2. **No patch embedding, no positional embedding, no tokenization** —
   image is processed as a multidimensional signal end-to-end
3. Phase information actively used in gating via sin/cos decomposition

Three departures from S9 standard layers:
1. Adds learnable spectral kernel (HAMSA's "simplified kernel parameterization")
2. Adds input-dependent SpectralPulseNet gating
3. Adds Spectral Adaptive Gating Unit (SAGU) operating in DOST coefficient domain

## 1. Notation

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | Batch size | `int` |
| $C_{\text{in}}$ | Input channels (e.g. 3 for RGB) | `int` |
| $H, W$ | **Raw image** spatial dimensions (NOT patch grid) | `int` |
| $D$ | Model channel dim after stem | `int` |
| $n$ | `n_per_axis` for Warped DOST | `int` |
| $D' = D \cdot n^2$ | Channel dim after DOST (2D case) | `int` |
| $L$ | Number of HSS blocks | `int` |
| $\Phi$ | 2D Warped DOST operator | non-learnable |
| $\Phi^{-1}$ | Inverse Warped DOST | non-learnable |
| $\odot$ | Element-wise product | — |
| $|\cdot|, \angle\cdot$ | Complex magnitude / phase | — |

**Critical invariant**: All learnable weight tensors have shapes
that depend ONLY on $D'$ (and $D$), never on $H, W$. This is what
guarantees resolution invariance.

## 2. Pipeline Overview

```
X ∈ ℝ^{B×C_in×H×W}
  │
  │  Stage 0: Stem (1×1 Conv2D, channel mapping only, no spatial change)
  ▼
U₀ ∈ ℝ^{B×D×H×W}
  │
  │  Stage 1: Warped DOST (Φ), calibrated once per resolution
  ▼
Z₀ ∈ ℂ^{B×D'×H×W}    (D' = D·n²)
  │
  │  Stage 2: L × HSS blocks, shape-preserving
  ▼
Z_L ∈ ℂ^{B×D'×H×W}
  │
  │  Stage 3: Inverse Warped DOST (Φ⁻¹)
  ▼
Y ∈ ℝ^{B×D×H×W}
  │
  │  Stage 4: Global Average Pool + Linear classifier
  ▼
ŷ ∈ ℝ^{B×num_classes}
```

The model accepts **any** $H, W$ for which Warped DOST has been calibrated.
There is no hardcoded spatial shape anywhere in the learnable parameters.

## 3. Stage 0: Stem

### 3.1 Specification

A 1×1 Conv2D mapping `C_in → D` channels.

$$
U_0 = \text{Conv2D}_{1 \times 1, \text{stride}=1, \text{padding}=0}(X) \in \mathbb{R}^{B \times D \times H \times W}
$$

### 3.2 Why 1×1 and stride 1

- 1×1 kernel: no implicit spatial mixing (preserves per-pixel locality of DOST coefficients downstream)
- stride 1: no spatial downsampling (resolution invariance)
- This is **NOT** patch embedding because (a) no spatial reduction, (b) no tokenization, (c) no positional encoding follows

### 3.3 Optional skip

If `D == C_in`, stem can be skipped entirely (set as `nn.Identity()`).

## 4. Stage 1: Warped DOST Preprocessing

### 4.1 Specification

Use `s9.transforms.warped_dost.WarpedDOST(D=2, n_per_axis=n)` directly.
Output:

$$
Z_0 = \Phi(U_0) \in \mathbb{C}^{B \times D' \times H \times W}, \quad D' = D \cdot n^2
$$

### 4.2 Calibration contract

The model MUST expose:
- `model.calibrate(x_batch: Tensor) -> None` — one-shot calibration
- `model.fitter` — property returning fresh streaming fitter

Both must call through to the underlying `WarpedDOST` instance. Without
calibration, forward pass MUST raise `RuntimeError`.

### 4.3 Cross-resolution workflow

For inference at resolution different from training:
```python
f = model.fitter
for x_chunk in eval_calib_loader:
    f.accumulate(x_chunk)
f.finalize()
# Now model.forward(eval_input) works at the new resolution
```

The model weights themselves are **never** modified — only the DOST boundaries.

## 5. Stage 2: HSS Block

Each block is shape-preserving:
$\mathbb{C}^{B \times D' \times H \times W} \to \mathbb{C}^{B \times D' \times H \times W}$.

### 5.1 Block-level dataflow

```
Z_ℓ
 ├──────────────────────────────────────┐
 │                                       │
 ▼                                       │
ComplexLN ──► HSS_core(·)                │
                  │                      │
                  ▼                      │
                  ⊕ ◄────────────────────┘  (residual 1)
                  │
                  ├──────────────────────┐
                  │                       │
                  ▼                       │
              ComplexLN ──► ComplexFFN    │
                              │           │
                              ▼           │
                              ⊕ ◄─────────┘  (residual 2)
                              │
                              ▼
                          Z_{ℓ+1}
```

### 5.2 HSS Core (`hss_core` function)

Input $Z \in \mathbb{C}^{B \times D' \times H \times W}$, output same shape.

**Step 1 — Dual projection (channel-only mixing)**:

$$
U = \text{einsum}(\text{"bchw,cd→bdhw"}, Z, W_u) + b_u
$$
$$
V = \text{einsum}(\text{"bchw,cd→bdhw"}, Z, W_v) + b_v
$$

with $W_u, W_v \in \mathbb{C}^{D' \times D'}$, $b_u, b_v \in \mathbb{C}^{D'}$.

**Step 2 — SASS (Spectral Adaptive State Space)**:
See §6.

**Step 3 — Output gating + projection**:

$$
O = (\tilde{U}_{\text{SAGU}} W_y + b_y) \odot V
$$
$$
Z_{\text{out}} = O W_o
$$

with $W_y, W_o \in \mathbb{C}^{D' \times D'}$. Element-wise $\odot$ between
$O$ candidate and $V$ is the GLU-style output gate.

### 5.3 ComplexLN

Operates on the channel dimension. Computes mean and variance from the
**magnitude** $|Z|$, normalizes the complex-valued $Z$ by the resulting real
scalar per (b, h, w), then applies learnable complex affine.

```
mean_mag = |Z|.mean(dim=channel)             # (B, H, W) real
var_mag  = |Z|.var(dim=channel, unbiased=False)
Z_norm   = Z / sqrt(var_mag + eps)            # complex
Z_out    = Z_norm * γ + β                     # γ, β ∈ ℂ^{D'}
```

(If `s9.modules` already provides ComplexLayerNorm, USE THAT and remove this
module. Document the choice in implementation comments.)

### 5.4 ComplexFFN

Two-layer MLP in channel dim:

$$
\text{FFN}(Z) = (\text{Activation}(Z W_1 + b_1)) W_2 + b_2
$$

with $W_1 \in \mathbb{C}^{D' \times d_{\text{ff}}}$,
$W_2 \in \mathbb{C}^{d_{\text{ff}} \times D'}$,
$d_{\text{ff}} = 4 D'$ (default; configurable).

Activation: complex-domain. Default: `s9.activations.complex.StableModReLU`.
(See `gen_activation` parameter following s9 convention.)

## 6. SASS Core

Operates on $U \in \mathbb{C}^{B \times D' \times H \times W}$ (output of dual
projection).

### 6.1 Phase-Aware SpectralPulseNet (SPN)

This is the **core innovation** of h9.

**Step 1 — Decompose**:

$$
m = |U| \in \mathbb{R}_{\geq 0}^{B \times D' \times H \times W}
$$
$$
\cos\theta = \text{Re}(U) / (m + \epsilon), \quad \sin\theta = \text{Im}(U) / (m + \epsilon)
$$

with $\epsilon = 10^{-8}$ for numerical stability at $m \to 0$.

**Step 2 — Linear combination + sigmoid**:

$$
g = \sigma(W_m m + W_p \cos\theta + W_p' \sin\theta + b_g)
$$

with $W_m, W_p, W_p' \in \mathbb{R}^{D' \times D'}$, $b_g \in \mathbb{R}^{D'}$.
All matmul are channel-only (einsum `"bchw,cd→bdhw"`); broadcast over $(h, w)$.

Output: $g \in [0, 1]^{B \times D' \times H \times W}$, real-valued.

### 6.2 Spectral Kernel

Single learnable complex vector:

$$
\hat{K} = \psi_{\text{re}} + j \cdot \psi_{\text{im}}, \quad \psi_{\text{re}}, \psi_{\text{im}} \in \mathbb{R}^{D'}
$$

Initialization (MVP): $\psi_{\text{re}}, \psi_{\text{im}} \sim \mathcal{N}(0, \sigma_0^2)$ with $\sigma_0 = 1.0$ (learnable scalar via `nn.Parameter`).

Future: alternative inits per `init_mode` parameter (`"gaussian"` default,
others added in later phases).

### 6.3 Combined application

$$
\tilde{U}_{b,c,h,w} = g_{b,c,h,w} \cdot U_{b,c,h,w} \cdot \hat{K}_c
$$

- $g$ is real, broadcast over its own shape (no broadcast needed)
- $\hat{K}$ is complex shape $(D',)$, broadcast over $(B, H, W)$
- Result: $\tilde{U} \in \mathbb{C}^{B \times D' \times H \times W}$

### 6.4 SAGU (DOST-Domain Variant)

$$
\text{SAGU}(\tilde{U}) = (\tilde{U} W_1) \odot \sigma(|\tilde{U}| W_2 + b_2)
$$

with $W_1 \in \mathbb{C}^{D' \times D'}$ (complex linear branch),
$W_2 \in \mathbb{R}^{D' \times D'}$ (gate from magnitudes), $b_2 \in \mathbb{R}^{D'}$.

Both matmuls are channel-only, applied per spatial location.

**Why magnitude-based gate (not direct sigmoid on complex)**: applying sigmoid
to complex values produces phase discontinuities and unstable gradients.
Magnitude-based gating (real sigmoid output) preserves both magnitude AND phase
of $\tilde{U} W_1$ when multiplied by the gate.

## 7. Stage 3: IDOST and Head

### 7.1 IDOST

$$
Y = \Phi^{-1}(Z_L) \in \mathbb{R}^{B \times D \times H \times W}
$$

Use `WarpedDOST.get_inverse_transform()` from s9. The boundaries used must be
the same as forward.

### 7.2 Head

Standard:
1. Global Average Pool over (H, W): $(B, D, H, W) \to (B, D)$
2. (Optional) `LayerNorm`
3. Linear: $(B, D) \to (B, \text{num\_classes})$

GAP works for any $(H, W) \geq (1, 1)$, preserving resolution invariance through
to the final logits.

## 8. Dual Attribution (Eval Only)

Implementation must be **non-intrusive**: zero overhead during training,
opt-in during inference.

### 8.1 What to capture

Per HSS block $\ell$, capture the tensor:

$$
A^{(\ell)}_{c, h, w} = |g^{(\ell)}_{c, h, w} \cdot U^{(\ell)}_{c, h, w}|^2 \in \mathbb{R}_{\geq 0}^{B \times D' \times H \times W}
$$

Trigger: forward hook on the SPN output, only when `model.attribution_enabled`.

### 8.2 Aggregation views

Provided as utility functions:

| View | Shape | Computation |
|---|---|---|
| `spatial(A)` | $(B, H, W)$ | $\sum_c A_{c,h,w}$ |
| `spectral(A)` | $(B, D')$ | $\sum_{h,w} A_{c,h,w}$ |
| `joint(A, n)` | $(B, n, n, H, W)$ | reshape $D' = D \cdot n^2$ → $(D, n, n)$, then sum over $D$ |

The `joint` view is the dual attribution proper — for each pair of spatial
frequencies $(f_x, f_y) \in \{0, ..., n-1\}^2$ it gives a spatial heatmap.

## 9. Resolution Invariance Theorem

**Claim**: For any two resolutions $(H_1, W_1)$ and $(H_2, W_2)$, the same
trained model weights can be used at both, provided Warped DOST has been
calibrated for each.

**Proof sketch**:
1. Stem (1×1 Conv2D): weights $\in \mathbb{R}^{D \times C_{\text{in}}}$, depends only on $C_{\text{in}}, D$.
2. Warped DOST: 0 learnable parameters; `D'` is invariant by Warped DOST design (s9 README).
3. HSS block: every weight matrix has shape $D' \times D'$ or $D' \times d_{\text{ff}}$, etc. None depend on $H, W$.
4. ComplexLN: γ, β $\in \mathbb{C}^{D'}$.
5. IDOST: 0 learnable parameters.
6. GAP + Linear: GAP independent of $(H, W)$; Linear weights $\in \mathbb{R}^{\text{num\_classes} \times D}$.

Therefore the parameter set is closed under change of $(H, W)$. □

This must be **tested explicitly** via `test_h9_resolution_invariance.py`.

## 10. Parameter Count and Sizing Guidelines

For a single HSS block at $D' = D \cdot n^2$ with $n=2$ (so $D' = 4D$):

| Module | Parameter count |
|---|---|
| Dual projections ($W_u, W_v$, complex) | $2 \cdot 2 \cdot (D')^2 = 4 \cdot 16 D^2 = 64 D^2$ |
| Phase-Aware SPN ($W_m, W_p, W_p'$, real) | $3 \cdot (D')^2 = 48 D^2$ |
| Spectral kernel | $2 D' = 8 D$ (negligible) |
| SAGU ($W_1$ complex + $W_2$ real) | $2(D')^2 + (D')^2 = 48 D^2$ |
| Output gate + proj ($W_y, W_o$ complex) | $2 \cdot 2 (D')^2 = 64 D^2$ |
| ComplexFFN (×2 LN, $d_{\text{ff}} = 4D'$) | $2 \cdot 2 \cdot D' \cdot 4D' = 64 D^2$ |
| **Total per block** | $\sim 290 D^2$ |

For $D = 64$: ~1.2M params per block. With $L = 12$ blocks: ~14M params. Comparable to small ViT.

**Sizing recommendations**:
- Tiny: $D=64, n=2, L=8$
- Small: $D=96, n=2, L=12$
- Base: $D=128, n=2, L=12$ (or $D=96, n=3, L=12$)

`n=2` is a strong default. `n=3` or higher only if compute budget allows
($D'$ scales as $n^2$, so per-block cost as $(D')^2 = n^4 D^2$).

## 11. Implementation Notes

### 11.1 Numerical considerations

- All complex tensors: `torch.complex64` (matches s9 `dtype_idx=64` convention; their `dtype_idx=64` actually means **complex64** = float32 real + float32 imag; verify in s9 code before relying)
- Magnitude epsilon: `1e-8` for `cos_θ`, `sin_θ` computation
- LayerNorm epsilon: `1e-5`

### 11.2 s9 conventions to mirror

Reference: `s9.modules.S9Layer.__init__` signature.

```python
class HSSBlock(nn.Module):
    def __init__(
        self,
        d_model: int,                        # = D
        n_per_axis: int,                     # = n (so D' = d_model * n_per_axis ** spatial_dims)
        spatial_dims: int = 2,
        gen_activation: type[nn.Module] = StableModReLU,
        d_ff_mult: int = 4,
        init_mode: Literal["gaussian"] = "gaussian",
        dtype_idx: int = 64,
        ...
    ):
```

The `init_mode` parameter is forward-looking — Phase 4 will add other options.

### 11.3 What to import from s9

- `s9.transforms.warped_dost.WarpedDOST` — preprocessor
- `s9.activations.complex.stable_modrelu.StableModReLU` — default complex activation
- `s9.activations.complex.StableComplexCardioid` — alternative
- `s9.activations.real.thash.ThASh` — for any real activation needs

If `s9.modules` provides `ComplexLayerNorm` or similar, use it. Otherwise
implement locally per §5.3.

### 11.4 Testing checkpoints

A successful implementation must pass:
1. **Shape tests**: forward pass at 32×32, 64×64, 128×128 produces correct shapes
2. **Resolution invariance test**: train at 32×32, evaluate at 64×64 after refit
3. **Calibration contract test**: forward without calibration raises `RuntimeError`
4. **Attribution non-intrusion test**: training pass identical with/without attribution flag set
5. **Round-trip test**: `Φ⁻¹(Φ(x))` close to x within DOST's known reconstruction tolerance

## 12. Out of Scope (Phase 1+2)

These are deliberately deferred:
- Spectral MoE (Phase 3, separate paper)
- Wavelet-coefficient initialization (Phase 4)
- Hierarchical/strided variants for ImageNet-scale (Phase 1.1, future)
- Video extension with temporal Warped DOST (separate task)
- Quantization (q variants, follow s9's QS9 patterns when needed)

## 13. References

1. Patro, B. N., & Agneeswaran, V. S. (2026). HAMSA: Scanning-Free Vision State Space Models via SpectralPulseNet. arXiv:2604.14724.
2. S9 repository: https://github.com/Honey-Be/s9
3. Warped DOST documentation: `README-WARPED-DOST.md` in s9 repo.
4. S4ND: Nguyen et al. (2022), arXiv:2210.06583.
5. DOST: Wang & Orchard (2009), IEEE Trans. Signal Processing.
