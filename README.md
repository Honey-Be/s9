# S9/RS9/ARS9: Multidimensional SSM based on S4ND + S7 Fusion

S9은 최신 상태 공간 모델(SSM) 연구인 S4ND와 S7의 장점을 융합하여 설계된 새로운 다차원 상태 공간 모델입니다.
이 모델은 실수 도메인의 데이터를 복소수 도메인으로 확장하여 **위상(Phase)과 진폭(Amplitude) 정보를 동시에 활용**합니다. 이를 위해 **Multidimensional Discrete Orthogonal Stockwell Transform (MD-DOST)** 기반의 학습되지 않는(Non-learnable) 전처리기를 도입했습니다.

v0.4.0에서는 S5(Simplified State Space Layers)의 핵심 기법들을 backport하여 **HiPPO-N 초기화**, **exact ZOH 이산화**, **ARS9(Advanced RS9) 계열**, **양자화(Q) 계열**을 도입했습니다.

## 주요 특징 (Key Features)
* **Multidimensional S9 Layer**: N차원 데이터(1D 시계열, 2D 이미지, 3D 비디오 등)들을 처리할 수 있는 일반화된 SSM 백본입니다.
* **S4ND + S7 Fusion**:
    - **S4ND 구조**: 각 차원을 독립적으로 처리한 후 Outer Product를 통해 다차원 커널을 생성하여 N차원 컨볼루션을 수행합니다.
    - **S7 상태 공유**: 효율적인 파라미터 공유 및 초기화 기법을 적용하여 모델의 경량화와 안정성을 확보했습니다.
* **Complex Domain Processing**:
    - **MD-DOST Preprocessor**: 입력 신호를 주파수 대역별로 분해하여 복소수 텐서로 변환합니다. 학습 가능한 파라미터 없이 고정된 변환을 수행합니다.
* **Stable Activation**: 복소수 연산의 특이점($z=0$) 문제를 해결한 `StableModReLU` 및 `StableComplexCardioid` 활성화 함수들을 제공합니다.
* **Type Safety**: Python 3.12+의 최신 타입 힌팅 기능을 적극 활용하여 코드의 안정성을 높였습니다.

### v0.4.0 신규 특징
* **ARS9 (Advanced RS9)**: 복소 conjugate-pair 내부 상태를 사용하면서 I/O는 실수. DOST 없이도 진동 모드를 표현 가능.
* **HiPPO-N / S4D-Real 초기화**: S5의 HiPPO-N 대각화 기법을 도입하여 장거리 의존성 학습 성능을 개선.
* **Exact ZOH 이산화**: 기존 1차 근사(`B·dt`) 대신 정확한 Zero-Order Hold 이산화를 기본으로 채택.
* **QS9/QRS9/QARS9 (양자화 계열)**: Q-S5의 per-component bit-budget 분석에 기반한 QAT/PTQ 지원 레이어.
* **체크포인트 마이그레이션**: v0.3.x → v0.4.0 전환을 위한 자동 파라미터 재매핑 스크립트 제공.

---

## v0.4.0 Migration Guide

### Breaking Change: Exact ZOH Discretization (default 변경)
v0.4.0에서는 SSM 커널 이산화 기본값이 `B_bar = B·dt` (1차 근사)에서 **`B_bar = (exp(A·dt) - 1) / A · B` (exact ZOH)**로 변경되었습니다.

* **이전 동작 복원**: 모든 Layer 생성자에 `discretization="approx"`를 명시합니다.
* **체크포인트 마이그레이션**: v0.3.x에서 학습한 모델을 v0.4.0 위에서 사용하려면 파라미터를 재매핑해야 합니다.
  ```python
  from s9.migration import migrate_state_dict_zoh
  new_sd = migrate_state_dict_zoh(old_state_dict)
  model.load_state_dict(new_sd)
  ```
  또는 CLI:
  ```bash
  python scripts/migrate_checkpoint.py --in checkpoint_v0.3.pt --out checkpoint_v0.4.pt
  ```
* **역방향 마이그레이션** (v0.4.0 → v0.3.x):
  ```python
  from s9.migration import migrate_state_dict_from_zoh
  old_sd = migrate_state_dict_from_zoh(new_state_dict)
  ```

마이그레이션은 수학적으로 정확(lossless)하며, forward 결과가 변환 전후로 동일합니다.

---

## 설치 (Installation)
이 프로젝트는 Poetry를 사용하여 패키지를 관리합니다.

**요구 사항:**
* Python >= 3.12, < 3.15
    - note: PyTorch doesn't support Python 3.15 yet.
* PyTorch(`torch`) >= 2.10.0
    - note 1: tests needed for PyTorch 2.8.x ~ 2.9.x
    - note 2: PyTorch <= 2.7.x won't be supported.

```bash
# CPU 백엔드
pip install "s9[cpu] @ git+https://github.com/Honey-Be/s9.git@v0.4.0"

# CUDA 12.6 백엔드
pip install "s9[cu126] @ git+https://github.com/Honey-Be/s9.git@v0.4.0"

# CUDA 12.8 백엔드
pip install "s9[cu128] @ git+https://github.com/Honey-Be/s9.git@v0.4.0"

# CUDA 13.0 백엔드
pip install "s9[cu130] @ git+https://github.com/Honey-Be/s9.git@v0.4.0"
```

---

## S9 사용법 (Usage)

### 1. 기본 모델 생성 (Classification Example)
`S9ClassifierModelExample`은 S9 레이어를 활용한 분류 모델의 예시입니다.

```python
import torch
from s9.examples import S9ClassifierModelExample

# 예: 32x32 컬러 이미지(2D)를 10개 클래스로 분류
model = S9ClassifierModelExample(
    in_channels=3,
    d_model=64,
    n_layers=4,
    num_classes=10,
    spatial_shape=(32, 32)  # (H, W)
)

# 더미 입력
x = torch.randn(2, 3, 32, 32)
logits = model(x)
print(logits.shape) # torch.Size([2, 10])
```

### 2. S9 레이어 직접 활용 (Backbone)
`S9Layer`를 여러분의 모델의 백본으로 사용할 수 있습니다. 단, 입력은 복소수 텐서여야 하므로 실수 -> 복소수 가역변환(예: DOST)으로 구성된 non-learnable 전처리기와 함께 사용하는 것을 권장합니다.

```python
import torch
from s9.transforms.dost import DOST
from s9.modules import S9Layer
from s9.activations.complex.stable_modrelu import StableModReLU

# 설정
d_model = 64
spatial_shape = (32, 32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모듈 초기화
dost = DOST(D=2) # 2D DOST 전처리기
layer = S9Layer(
    d_model=d_model,
    spatial_dims=len(spatial_shape),
    gen_activation=StableModReLU, # 활성화 함수 선택
    dtype_idx=64
).to(device)

# Forward Pass
x = torch.randn(2, 3, 32, 32).to(device) # Real Input
z = dost(x) # Real -> Complex (Channel Expansion)
out = layer(z) # Complex -> Complex Output
```

### 3. 초기화 모드 및 이산화 설정 (v0.4.0+)
모든 S9/RS9/ARS9 계열 레이어는 `init_mode`와 `discretization` 인자를 지원합니다.

```python
from s9.modules import S9Layer
from s9.activations.complex.stable_modrelu import StableModReLU

# HiPPO-N 초기화 + exact ZOH (기본값)
layer = S9Layer(
    d_model=64,
    spatial_dims=2,
    gen_activation=StableModReLU,
    init_mode="hippo_n",       # "legacy" | "hippo_n" (S9/ARS9) | "s4d_real" (RS9)
    discretization="zoh",      # "zoh" (기본) | "approx" (v0.3.x 호환)
)
```

| `init_mode` | 대상 커널 | 설명 |
|---|---|---|
| `"legacy"` | S9, RS9, ARS9 | v0.3.x 호환 초기화 (기본값) |
| `"hippo_n"` | S9, ARS9 | HiPPO-N 대각화: $\text{Im}(\lambda_n) = \pi(n + \tfrac{1}{2})$, $B_n = \sqrt{2n+1}$ |
| `"s4d_real"` | RS9 | S4D-Real: $A_n = -(n+1)/2$ log-spaced decay |

---

## 부록 1: RS9

복소수 도메인을 굳이 필요로 하지 않는 task들을 위해 v0.2.0에서 추가된 **RS9(real-valued S9)** 레이어(`s9.rs9_modules.RS9Layer`)는 domain이 $\mathbb{C}$ 에서 $\mathbb{R}$ 로 변경되면서 DOST를 통한 전처리가 불필요하게 되었지만, 이를 제외한 S9 레이어의 나머지 주요 특징들은 대부분 동일하게 갖습니다.

---

## 부록 2: ARS9 (Advanced RS9)

v0.4.0에서 추가된 **ARS9(Advanced RS9)** 레이어(`s9.ars9_modules.ARS9Layer`)는 RS9의 한계인 **진동 모드 부재**를 해결합니다.

### 핵심 아이디어

RS9의 상태 행렬 $A$는 순실수 음수이므로 지수 감쇠만 표현 가능합니다. ARS9는 S5(Smith et al., ICLR 2023)의 **complex conjugate-pair** 기법을 도입하여:

* 내부 상태는 $\mathbb{C}^{N/2}$ (N/2개의 켤레 복소 쌍)
* 커널 출력은 $2 \cdot \text{Re}(\sum_n C_n \bar{B}_n \bar{A}_n^t)$ → **실수 보장**
* 입력/출력은 실수 → DOST 불필요, RS9과 동일한 파이프라인

### 사용 예시

```python
import torch
from s9.ars9_modules import ARS9Layer
from s9.activations.real.thash import ThASh

def make_activation(d_model, eps, dtype_idx):
    return ThASh()

layer = ARS9Layer(
    d_model=64,
    spatial_dims=2,
    gen_activation=make_activation,
    init_mode="hippo_n",  # 장거리 모델링에 권장
    dtype_idx=64
)

x = torch.randn(2, 64, 32, 32)  # Real input
y = layer(x)  # (2, 64, 32, 32), real output
```

### ARS9 계열 전체 목록

| 레이어 | 모듈 경로 |
|---|---|
| `ARS9Layer` | `s9.ars9_modules` |
| `MultiheadARS9Layer` | `s9.multihead_ars9_modules` |
| `BiaffineARS9Layer` | `s9.biaffine_ars9_modules` |
| `GatedDeltaARS9Layer` | `s9.contrib.gated_delta_ars9_modules` |
| `BiaffineGatedDeltaARS9Layer` | `s9.contrib.gated_delta_ars9_modules` |

### 예시 분류 모델

* `s9.examples.ARS9ClassifierModelExample`
* `s9.examples.MultiheadARS9ClassifierModelExample`
* `s9.examples.BiaffineARS9ClassifierModelExample`
* `s9.contrib.examples.GatedDeltaARS9ClassifierExample`
* `s9.contrib.examples.BiaffineGatedDeltaARS9ClassifierExample`

---

## 부록 3: 실수 -> 복소수 가역변환 기반 non-learnable 전처리기

v0.2.5에서는 DOST/IDOST의 대체재로 사용할 수 있는 전처리기들과 이들에 결합할 수 있는 Synchrosqueezing Transform 기반 non-learnable wrapper가 추가되었습니다.
* **[2D-only, standalone]** Fast Curvelet Transform
* **[3D-only, standalone]** Fast Surfacelet Transform
* **[`N`-dimensional, standalone]** Riesz Transform
* **[`N`-dimensional, non-standalone]** Synchrosqueezing Transform

---

## 부록 4: 활성 함수 리팩터링 및 추가

v0.2.6에서는 이전까지 `s9.modules` 네임스페이스에서 직접 제공하였던 `StableComplexCardioid` 및 `StableModReLU` 활성화함수들을 `s9.activations.complex.*` 네임스페이스로 이관하였으며, `s9.activations.real.*` 네임스페이스에 `ThASh`(TanhArSinh) 및 `HGLU`(Hyperbolic Gain Linear Unit) 활성화함수들을 추가하였습니다.

$$ \text{ThASh}(x) = \text{tanh}(\text{arsinh}(x)) = \frac{x}{\sqrt{1 + x^2}} $$
$$ \forall k > 0,\; \text{HGLU}_k(x) = \frac{x + \sqrt{k + x^2}}{2} $$

v0.4.0에서는 모든 실수 활성함수가 `s9.activations.real.base.RealActivationBase`를 상속합니다.

---

## 부록 5: Multi-head S9/RS9/ARS9 및 Biaffine S9/RS9/ARS9

v0.2.8에서는 기존 S9/RS9 레이어를 다음과 같은 계층으로 일반화한 파생 레이어들이 추가되었습니다. v0.4.0에서는 ARS9 계열도 동일 계층을 갖습니다.

$$
\text{기존 S9/RS9/ARS9}
\;\to\;
\text{Multi-head S9/RS9/ARS9}
\;\to\;
\text{Biaffine S9/RS9/ARS9}
$$

### 5-1. Multi-head

각 head는 다음 절차로 동작합니다:

1. 입력 채널을 head별 잠재 채널(latent channels)로 선형 사상
2. 각 공간 차원별로 독립적인 1D SSM kernel 생성
3. 이 1D kernel들의 outer product를 통해 다차원 global kernel 구성
4. FFT 기반의 다차원 convolution 수행
5. head 출력을 다시 모델 차원으로 사상

모든 head의 출력은 합산되며, 그 뒤에 activation, output linear layer, dropout이 적용됩니다.

### 5-2. Biaffine

multi-head 구조를 바탕으로, 각 head 내부의 단순 선형 입력/출력 사상을 **latent kernel bank + biaffine channel coupling**으로 일반화합니다.

### 5-3. 전체 레이어 목록

| 도메인 | Base | Multi-head | Biaffine |
|---|---|---|---|
| Complex (S9) | `S9Layer` | `MultiheadS9Layer` | `BiaffineS9Layer` |
| Real (RS9) | `RS9Layer` | `MultiheadRS9Layer` | `BiaffineRS9Layer` |
| Conjugate-pair (ARS9) | `ARS9Layer` | `MultiheadARS9Layer` | `BiaffineARS9Layer` |

---

## 부록 6: Gated Delta S9/RS9/ARS9 (`s9.contrib`)

Gated DeltaNet(ICLR 2025, Songlin Yang et al.)의 내부 SSM을 S9/RS9/ARS9로 교체한 실험적 레이어들입니다. 기존 Gated DeltaNet의 세 가지 근본적 한계를 해소합니다:

| 한계 | 기존 Gated DeltaNet | Gated Delta S9/RS9/ARS9 |
|------|---------------------|---------------------|
| 고정 크기 상태 | $S \in \mathbb{C}^{d_k \times d_v}$ | 상태 없음; SSM 커널이 입력 길이에 맞게 생성 |
| Rank-2 전이 | $(I - \beta kk^\top) + \beta vk^\top$ | Full-rank SSM 커널 (N=64 지수 기저함수의 합) |
| 스칼라 게이팅 | $\alpha \in \mathbb{R}$ | 위치별·채널별 full-tensor gate + 복소 동역학 |

### 6-1. 핵심 수식

입력 $u$에 대해:

1. **Gate 생성**: $[\alpha, \beta] = \sigma(W_g \cdot g_{\text{in}})$, $z = W_z \cdot g_{\text{in}}$ (데이터 의존적)
2. **Multi-head S9/RS9/ARS9 컨볼루션**: $y = \sum_h \text{head}_h(u)$ (차원별 SSM 커널의 outer product → FFT 컨볼루션)
3. **Gated Delta 결합**: $\text{combined} = \alpha \odot u + \beta \odot y$
4. **출력 게이팅**: $\text{output} = \text{Norm}(\text{Activation}(W_{\text{out}} \cdot \text{combined})) \odot \text{SiLU}(z)$

### 6-2. 제공되는 레이어

| 도메인 | Multi-head | Biaffine |
|---|---|---|
| Complex (S9) | `GatedDeltaS9Layer` | `BiaffineGatedDeltaS9Layer` |
| Real (RS9) | `GatedDeltaRS9Layer` | `BiaffineGatedDeltaRS9Layer` |
| Conjugate-pair (ARS9) | `GatedDeltaARS9Layer` | `BiaffineGatedDeltaARS9Layer` |

### 6-3. 사용 예시

```python
import torch
from s9.transforms.dost import DOST
from s9.contrib.gated_delta_s9_modules import GatedDeltaS9Layer
from s9.activations.complex.stable_modrelu import StableModReLU

# 설정
d_model = 64
spatial_shape = (32, 32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모듈 초기화
dost = DOST(D=2)
layer = GatedDeltaS9Layer(
    d_model=d_model,
    spatial_dims=2,
    gen_activation=StableModReLU,
    n_heads=4,
    head_channels=(16,),  # 각 head의 latent channels
    dtype_idx=64
).to(device)

# Forward Pass
x = torch.randn(2, 3, 32, 32).to(device)
z = dost(x)  # Real -> Complex (Channel Expansion)
# z를 d_model 채널로 projection 후 사용
```

RS9/ARS9(real-valued) 레이어는 DOST 전처리 없이 직접 사용할 수 있습니다:

```python
import torch
from s9.contrib.gated_delta_ars9_modules import GatedDeltaARS9Layer
from s9.activations.real.thash import ThASh

def make_activation(d_model, eps, dtype_idx):
    return ThASh()

layer = GatedDeltaARS9Layer(
    d_model=64,
    spatial_dims=2,
    gen_activation=make_activation,
    n_heads=4,
    head_channels=(16,),
    init_mode="hippo_n",  # ARS9는 HiPPO-N 초기화 지원
    dtype_idx=64
)

x = torch.randn(2, 64, 32, 32)
y = layer(x)  # (2, 64, 32, 32)
```

### 6-4. 예시 분류 모델

`s9.contrib.examples` 모듈에서 6개의 분류 모델 예시를 제공합니다:

* `GatedDeltaS9ClassifierExample` — DOST + GatedDeltaS9Layer
* `BiaffineGatedDeltaS9ClassifierExample` — DOST + BiaffineGatedDeltaS9Layer
* `GatedDeltaRS9ClassifierExample` — GatedDeltaRS9Layer (DOST 없음)
* `BiaffineGatedDeltaRS9ClassifierExample` — BiaffineGatedDeltaRS9Layer (DOST 없음)
* `GatedDeltaARS9ClassifierExample` — GatedDeltaARS9Layer (DOST 없음)
* `BiaffineGatedDeltaARS9ClassifierExample` — BiaffineGatedDeltaARS9Layer (DOST 없음)

---

## 부록 7: 양자화 계열 QS9/QRS9/QARS9 (`s9.quantization`)

v0.4.0에서 추가된 양자화 계열은 Q-S5(Abreu et al., 2024)의 per-component sensitivity 분석에 기반합니다.

### 7-1. 설계 원칙

Q-S5의 핵심 발견을 S9의 FFT convolution 구조에 맞게 재해석합니다:

| 구성 요소 | 정밀도 | 근거 |
|---|---|---|
| $A, \Delta$ (재귀 행렬, 스텝 크기) | fp32 유지 | 커널 생성에만 사용; FFT conv 구조에서 비용 부담 미미 |
| $B, C$ (입/출력 행렬) | int8 | Q-S5: 비재귀 가중치는 4-bit까지 가능 |
| `output_linear` | int4 | 비재귀 선형층 |
| 입력 $u$ | int8 | 활성 양자화 |

### 7-2. 사용 예시

```python
import torch
from s9.qrs9_modules import QRS9Layer
from s9.quantization import QuantConfig
from s9.activations.real.thash import ThASh

def make_act(d, eps, idx): return ThASh()

# 커스텀 bit-budget 설정
config = QuantConfig(
    w_bits_B=8,
    w_bits_C=8,
    w_bits_output=4,
    a_bits_input=8,
    enforce_stability=True,
    stability_epsilon=1e-3,
)

layer = QRS9Layer(
    d_model=64,
    spatial_dims=2,
    gen_activation=make_act,
    quant_config=config,
)

x = torch.randn(2, 64, 32, 32)
y = layer(x)  # Quantized forward pass
```

### 7-3. 제공되는 Q 계열 레이어

| 도메인 | Base Q Layer |
|---|---|
| Complex (S9) | `s9.qs9_modules.QS9Layer` |
| Real (RS9) | `s9.qrs9_modules.QRS9Layer` |
| Conjugate-pair (ARS9) | `s9.qars9_modules.QARS9Layer` |

### 7-4. 양자화 유틸리티 (`s9.quantization`)

* `QuantConfig` — per-component bit-width 설정 dataclass
* `fake_quant(x, bits)` — STE 기반 fake quantization
* `symmetric_per_tensor_quantize(x, bits)` — 복소 텐서 지원 per-tensor symmetric 양자화
* `QuantizedKernelCache` — eval 모드에서 커널 캐시 + 양자화
* `PolarStableModReLU` — 복소 활성의 polar 분해 후 독립 양자화
* `QThASh`, `QHGLU` — 실수 활성의 양자화 변종
* `assert_discrete_stability(A_bar, epsilon)` — 이산 극점 안정성 검증

---

## 부록 8: Warped DOST - Inverse Warped DOST 변환쌍
[README-WARPED-DOST.md](./README-WARPED-DOST.md) 문서 참조.

---

## 출처 및 참고 문헌 (References)

이 프로젝트는 다음의 연구 논문들에 기반하여 구현되었습니다.

1. **S4ND** (Multidimensional SSM)
    * Nguyen, E., Goel, K., Gu, A., Downs, G., Shah, P., Dao, T., Baccus, S., & Ré, C. (2022). **S4ND: Modeling Images and Videos as Multidimensional Signals Using State Spaces**. *arXiv preprint arXiv:2210.06583*.
    * DOI: [10.48550/arXiv.2210.06583](https://doi.org/10.48550/arXiv.2210.06583)
2. **S7** (Simplified SSM)
    * Wang, J., Zhu, W., Wang, P., Yu, X., Liu, L., & Saligrama, V. (2024). **S7: Simplified State Space Layers for Sequence Modeling**. *arXiv preprint arXiv:2410.03464*.
    * DOI: [10.48550/arXiv.2410.03464](https://doi.org/10.48550/arXiv.2410.03464)
3. **S5** (Simplified State Space Layers)
    * Smith, J. T. H., Warrington, A., & Linderman, S. W. (2023). **Simplified State Space Layers for Sequence Modeling**. *ICLR 2023*. *arXiv preprint arXiv:2208.04933*.
    * DOI: [10.48550/arXiv.2208.04933](https://doi.org/10.48550/arXiv.2208.04933)
4. **Q-S5** (Quantized State Space Models)
    * Abreu, S., Pedersen, J. E., Heckel, K. M., & Pierro, A. (2024). **Q-S5: Towards Quantized State Space Models**. *ICML 2024 Workshop (NGSM)*. *arXiv preprint arXiv:2406.09477*.
    * DOI: [10.48550/arXiv.2406.09477](https://doi.org/10.48550/arXiv.2406.09477)
5. **Gated Delta Networks**
    * Yang, S., Kautz, J., & Hatamizadeh, A. (2025). **Gated Delta Networks: Improving Mamba2 with Delta Rule**. *ICLR 2025*. *arXiv preprint arXiv:2412.06464*.
    * DOI: [10.48550/arXiv.2412.06464](https://doi.org/10.48550/arXiv.2412.06464)
6. **DOST** (Discrete Orthogonal Stockwell Transform)
    * Wang, Y., & Orchard, J. (2009). Fast Discrete Orthogonal Stockwell Transform. IEEE Transactions on Signal Processing, 57(9), 3615-3625.
    * (Note: 본 프로젝트에서는 이를 다차원 딥러닝 파이프라인에 맞게 근사 및 최적화하여 구현한 버전을 사용합니다.)

---

## 라이선스 (License)

이 프로젝트는 **GNU Lesser General Public License v2.1 or later** 하에 배포됩니다.
This project is licensed under the **GNU LGPLv2.1+**. See the `LICENSE.txt` file for details.
