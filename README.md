# S9/RS9: Multidimensional SSM based on S4ND + S7 Fusion

S9은 최신 상태 공간 모델(SSM) 연구인 S4ND와 S7의 장점을 융합하여 설계된 새로운 다차원 상태 공간 모델입니다.
이 모델은 실수 도메인의 데이터를 복소수 도메인으로 확장하여 **위상(Phase)과 진폭(Amplitude) 정보를 동시에 활용**합니다. 이를 위해 **Multidimensional Discrete Orthogonal Stockwell Transform (MD-DOST)** 기반의 학습되지 않는(Non-learnable) 전처리기를 도입했습니다.

## 🌟 주요 특징 (Key Features)
* **Multidimensional S9 Layer**: N차원 데이터(1D 시계열, 2D 이미지, 3D 비디오 등)들을 처리할 수 있는 일반화된 SSM 백본입니다.
* **S4ND + S7 Fusion**:
    - **S4ND 구조**: 각 차원을 독립적으로 처리한 후 Outer Product를 통해 다차원 커널을 생성하여 N차원 컨볼루션을 수행합니다.
    - **S7 상태 공유**: 효율적인 파라미터 공유 및 초기화 기법을 적용하여 모델의 경량화와 안정성을 확보했습니다.
* **Complex Domain Processing**:
    - **MD-DOST Preprocessor**: 입력 신호를 주파수 대역별로 분해하여 복소수 텐서로 변환합니다. 학습 가능한 파라미터 없이 고정된 변환을 수행합니다.
* **Stable Activation**: 복소수 연산의 특이점($z=0$) 문제를 해결한 `StableModReLU` 및 `StableComplexCardioid` 활성화 함수들을 제공합니다.
* **Type Safety**: Python 3.12+의 최신 타입 힌팅 기능을 적극 활용하여 코드의 안정성을 높였습니다.

## 📦 설치 (Installation)
이 프로젝트는 Poetry를 사용하여 패키지를 관리합니다.

**요구 사항:**
    * Python >= 3.12, < 3.15
        - note: PyTorch doesn't support Python 3.15 yet.
    * PyTorch(`torch`) >= 2.10.0
        - note 1: tests needed for PyTorch 2.8.x ~ 2.9.x
        - note 2: PyTorch <= 2.7.x won't be supported.

``` bash
# CPU 백엔드
pip install "s9[cpu] @ git+https://github.com/Honey-Be/s9.git@v0.3.0"

# CUDA 12.6 백엔드
pip install "s9[cu126] @ git+https://github.com/Honey-Be/s9.git@v0.3.0"

# CUDA 12.8 백엔드
pip install "s9[cu128] @ git+https://github.com/Honey-Be/s9.git@v0.3.0"

# CUDA 13.0 백엔드
pip install "s9[cu130] @ git+https://github.com/Honey-Be/s9.git@v0.3.0"
```
## 🚀 S9 사용법 (Usage)
1. **기본 모델 생성 (Classification Example)**
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

2. **S9 레이어 직접 활용 (Backbone)**
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

## 부록 1: RS9
복소수 도메인을 굳이 필요로 하지 않는 task들을 위해 v0.2.0에서 추가된 **RS9(real-valued S9)** 레이어(`s9.rs9_modules.RS9Layer`)는 domain이 $\mathbb{C}$ 에서 $\mathbb{R}$ 로 변경되면서 DOST를 통한 전처리가 불필요하게 되었지만, 이를 제외한 S9 레이어의 나머지 주요 특징들은 대부분 동일하게 갖습니다.

## 부록 2: 실수 -> 복소수 가역변환 기반 non-learnable 전처리기
v0.2.5에서는 DOST/IDOST의 대체재로 사용할 수 있는 전처리기들과 이들에 결합할 수 있는 Synchrosqueezing Transform 기반 non-learnable wrapper가 추가되었습니다.
* **[2D-only, standalone]** Fast Curvelet Transform
* **[3D-only, standalone]** Fast Surfacelet Transform
* **[`N`-dimensional, standalone]** Riesz Transform
* **[`N`-dimensional, non-standalone]** Synchrosqueezing Transform

## 부록 3: 활성 함수 리팩터링 및 추가
v0.2.6에서는 이전까지 `s9.modules` 네임스페이스에서 직접 제공하였던 `StableComplexCardioid` 및 `StableModReLU` 활성화함수들을 `s9.activations.complex.*` 네임스페이스로 이관하였으며, `s9.activations.real.*` 네임스페이스에 `ThASh`(TanhArSinh) 및 `HGLU`(Hyperbolic Gain Linear Unit) 활성화함수들을 추가하였습니다.
$$ \text{ThASh}(x) = \text{tanh}(\text{arsinh}(x)) = \frac{x}{\sqrt{1 + x^2}} $$
$$ \forall k > 0 \text{HGLU}_k(x) = \frac{x + \sqrt{k + x^2}}{2} $$

## 부록 4: Multi-head S9/RS9 및 Biaffine S9/RS9

v0.2.8에서는 기존 S9/RS9 레이어를 다음과 같은 계층으로 일반화한 파생 레이어들이 추가되었습니다.

$$
\text{기존 S9/RS9}
\;\to\;
\text{Multi-head S9/RS9}
\;\to\;
\text{Biaffine S9/RS9}
$$

추가된 레이어들은 다음과 같습니다.

* `s9.multihead_s9_modules.MultiheadS9Layer`
* `s9.multihead_rs9_modules.MultiheadRS9Layer`
* `s9.biaffine_s9_modules.BiaffineS9Layer`
* `s9.biaffine_rs9_modules.BiaffineRS9Layer`

### 4-1. Multi-head S9/RS9

`MultiheadS9Layer` 및 `MultiheadRS9Layer`는 기존 S9/RS9의 다차원 SSM 구조를 유지하면서, 여러 개의 head를 병렬로 두도록 일반화한 레이어들입니다. 각 head는 다음과 같은 절차로 동작합니다.

1. 입력 채널을 head별 잠재 채널(latent channels)로 선형 사상
2. 각 공간 차원별로 독립적인 1D SSM kernel 생성
3. 이 1D kernel들의 outer product를 통해 다차원 global kernel 구성
4. FFT 기반의 다차원 convolution 수행
5. head 출력을 다시 모델 차원으로 사상

모든 head의 출력은 합산되며, 그 뒤에 기존 S9/RS9와 마찬가지로 activation, output linear layer, dropout이 적용됩니다. 즉, **기존 S4ND 계열의 outer-product kernel 구성 및 FFT convolution 경로는 유지하면서, head 단위의 병렬 분해를 추가한 구조**라고 볼 수 있습니다.

`MultiheadS9Layer`는 complex-valued 입력/출력을 대상으로 하며, `MultiheadRS9Layer`는 real-valued 입력/출력을 대상으로 합니다. 이를 제외한 큰 구조적 아이디어는 양쪽이 거의 동일합니다.

### 4-2. Biaffine S9/RS9

`BiaffineS9Layer` 및 `BiaffineRS9Layer`는 위의 multi-head 구조를 바탕으로, 각 head 내부의 단순 선형 입력/출력 사상을 **latent kernel bank + biaffine channel coupling**으로 일반화한 레이어들입니다.

직관적으로는, multi-head 계열이 “head별 잠재 채널로 투영한 뒤 convolution을 수행하는 구조”라면, biaffine 계열은 여기에 더해 **입력 채널과 출력 채널 사이의 쌍별 상호작용(pairwise interaction)** 을 더 풍부하게 모델링합니다. 이를 위해 각 head는 다음과 같은 두 종류의 계수를 학습합니다.

* 입력 채널 쪽 mixing 계수
* 출력 채널 쪽 mixing 계수

그리고 이 둘을 latent kernel bank와 결합하여, 단순한 채널별 convolution이 아니라 **입력 채널–출력 채널 쌍에 대해 보다 풍부한 coupling** 이 일어나도록 구성합니다. 이 구조는 기존 S9/RS9의 다차원 SSM 및 FFT 기반 convolution 경로를 보존하면서도, 채널 간 상호작용 표현력을 확장하는 데 목적이 있습니다.

### 4-3. 공통점과 차이점

이 네 가지 파생 레이어들은 모두 다음 공통점을 가집니다.

* 각 공간 차원별 1D SSM kernel 생성
* outer product 기반 다차원 global kernel 구성
* FFT 기반 다차원 convolution
* pointwise activation + output linear + dropout

즉, **기존 S9/RS9의 다차원 SSM backbone은 유지**됩니다. 차이점은 채널 결합 방식에 있습니다.

* `Multihead*Layer`: head 단위의 병렬 분해를 도입
* `Biaffine*Layer`: multi-head 구조 위에 biaffine channel coupling을 추가

또한 field 관점에서 보면,

* `S9` 계열: complex-valued
* `RS9` 계열: real-valued

이라는 차이만 있으며, 구조적 아이디어 자체는 최대한 공통적으로 유지됩니다.

### 4-4. 관련 예시 코드

v0.2.8에서는 S9 계열에 대해 아래 예시 모델도 함께 제공됩니다.

* `s9.examples.MultiheadS9ClassifierModelExample`
* `s9.examples.BiaffineS9ClassifierModelExample`

이 예시들은 DOST 기반 복소수 전처리 뒤에 Multi-head S9 또는 Biaffine S9 백본을 쌓아 분류를 수행하는 최소 예제입니다.

## 부록 5: Gated Delta S9/RS9 (`s9.contrib`)

Gated DeltaNet(ICLR 2025, Songlin Yang et al.)의 내부 SSM을 S9/RS9로 교체한 실험적 레이어들입니다. 기존 Gated DeltaNet의 세 가지 근본적 한계를 해소합니다:

| 한계 | 기존 Gated DeltaNet | Gated Delta S9/RS9 |
|------|---------------------|---------------------|
| 고정 크기 상태 | $S \in \mathbb{C}^{d_k \times d_v}$ | 상태 없음; SSM 커널이 입력 길이에 맞게 생성 |
| Rank-2 전이 | $(I - \beta kk^\top) + \beta vk^\top$ | Full-rank SSM 커널 (N=64 지수 기저함수의 합) |
| 스칼라 게이팅 | $\alpha \in \mathbb{R}$ | 위치별·채널별 full-tensor gate + 복소 동역학 |

### 5-1. 핵심 수식

입력 $u$에 대해:

1. **Gate 생성**: $[\alpha, \beta] = \sigma(W_g \cdot g_{\text{in}})$, $z = W_z \cdot g_{\text{in}}$ (데이터 의존적)
2. **Multi-head S9/RS9 컨볼루션**: $y = \sum_h \text{head}_h(u)$ (차원별 SSM 커널의 outer product → FFT 컨볼루션)
3. **Gated Delta 결합**: $\text{combined} = \alpha \odot u + \beta \odot y$
4. **출력 게이팅**: $\text{output} = \text{Norm}(\text{Activation}(W_{\text{out}} \cdot \text{combined})) \odot \text{SiLU}(z)$

### 5-2. 제공되는 레이어

* `s9.contrib.gated_delta_s9_modules.GatedDeltaS9Layer` — complex-valued, multi-head
* `s9.contrib.gated_delta_s9_modules.BiaffineGatedDeltaS9Layer` — complex-valued, biaffine channel coupling
* `s9.contrib.gated_delta_rs9_modules.GatedDeltaRS9Layer` — real-valued, multi-head
* `s9.contrib.gated_delta_rs9_modules.BiaffineGatedDeltaRS9Layer` — real-valued, biaffine channel coupling

### 5-3. 사용 예시

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

RS9(real-valued) 레이어는 DOST 전처리 없이 직접 사용할 수 있습니다:

```python
import torch
from s9.contrib.gated_delta_rs9_modules import GatedDeltaRS9Layer
from s9.activations.real.thash import ThASh

def make_activation(d_model, eps, dtype_idx):
    return ThASh()

layer = GatedDeltaRS9Layer(
    d_model=64,
    spatial_dims=2,
    gen_activation=make_activation,
    n_heads=4,
    head_channels=(16,),
    dtype_idx=64
)

x = torch.randn(2, 64, 32, 32)
y = layer(x)  # (2, 64, 32, 32)
```

### 5-4. 예시 분류 모델

`s9.contrib.examples` 모듈에서 4개의 분류 모델 예시를 제공합니다:

* `GatedDeltaS9ClassifierExample` — DOST + GatedDeltaS9Layer
* `BiaffineGatedDeltaS9ClassifierExample` — DOST + BiaffineGatedDeltaS9Layer
* `GatedDeltaRS9ClassifierExample` — GatedDeltaRS9Layer (DOST 없음)
* `BiaffineGatedDeltaRS9ClassifierExample` — BiaffineGatedDeltaRS9Layer (DOST 없음)

## 📚 출처 및 참고 문헌 (References)
이 프로젝트는 다음의 연구 논문들에 기반하여 구현되었습니다.
1. S4ND (Multidimensional SSM)
    * Nguyen, E., Goel, K., Gu, A., Downs, G., Shah, P., Dao, T., Baccus, S., & Ré, C. (2022). **S4ND: Modeling Images and Videos as Multidimensional Signals Using State Spaces**. *arXiv preprint arXiv:2210.06583*.
    * DOI: [10.48550/arXiv.2210.06583](https://doi.org/10.48550/arXiv.2210.06583)
2. S7 (Simplified SSM)
    * Wang, J., Zhu, W., Wang, P., Yu, X., Liu, L., & Saligrama, V. (2024). **S7: Simplified State Space Layers for Sequence Modeling**. *arXiv preprint arXiv:2410.03464*.
    * DOI: [10.48550/arXiv.2410.03464](https://doi.org/10.48550/arXiv.2410.03464)
3. Gated Delta Networks
    * Yang, S., Kautz, J., & Hatamizadeh, A. (2025). **Gated Delta Networks: Improving Mamba2 with Delta Rule**. *ICLR 2025*. *arXiv preprint arXiv:2412.06464*.
    * DOI: [10.48550/arXiv.2412.06464](https://doi.org/10.48550/arXiv.2412.06464)
4. DOST (Discrete Orthogonal Stockwell Transform)
    * Wang, Y., & Orchard, J. (2009). Fast Discrete Orthogonal Stockwell Transform. IEEE Transactions on Signal Processing, 57(9), 3615-3625.
    * (Note: 본 프로젝트에서는 이를 다차원 딥러닝 파이프라인에 맞게 근사 및 최적화하여 구현한 버전을 사용합니다.)

## 📝 라이선스 (License)
이 프로젝트는 **GNU Lesser General Public License v2.1 or later** 하에 배포됩니다.
This project is licensed under the **GNU LGPLv2.1+**. See the `LICENSE.txt` file for details.
