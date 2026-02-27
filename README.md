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
    * Python >= 3.12, < 3.14
        - note: tests needed for Python v3.14.x or later
    * PyTorch(`torch`) >= 2.10.0
        - note 1: tests needed for PyTorch v2.8.x ~ v2.9.x
        - note 2: PyTorch <= 2.7.x won't be supported.

``` bash
# CPU 백엔드
pip install "s9[cpu] @ git+https://github.com/Honey-Be/s9.git@v0.2.4"

# CUDA 12.6 백엔드
pip install "s9[cu126] @ git+https://github.com/Honey-Be/s9.git@v0.2.4"

# CUDA 12.8 백엔드
pip install "s9[cu128] @ git+https://github.com/Honey-Be/s9.git@v0.2.4"

# CUDA 13.0 백엔드
pip install "s9[cu130] @ git+https://github.com/Honey-Be/s9.git@v0.2.4"
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
from s9.dost import DOST
from s9.modules import S9Layer, StableModReLU

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

# 부록 2: 실수 -> 복소수 가역변환 기반 non-learnable 전처리기
v0.2.5에서는 DOST/IDOST의 대체재로 사용할 수 있는 전처리기들과 이들에 결합할 수 있는 Synchrosqueezing Transform 기반 non-learnable wrapper가 추가되었습니다.
* **[2D-only, standalone]** Fast Curvelet Transform
* **[3D-only, standalone]** Fast Surfacelet Transform
* **[`N`-dimensional, standalone]** Riesz Transform
* **[`N`-dimensional, non-standalone]** Synchrosqueezing Transform

## 📚 출처 및 참고 문헌 (References)
이 프로젝트는 다음의 연구 논문들에 기반하여 구현되었습니다.
1. S4ND (Multidimensional SSM)
    * Nguyen, E., Goel, K., Gu, A., Downs, G., Shah, P., Dao, T., Baccus, S., & Ré, C. (2022). **S4ND: Modeling Images and Videos as Multidimensional Signals Using State Spaces**. *arXiv preprint arXiv:2210.06583*.
    * DOI: [10.48550/arXiv.2210.06583](https://doi.org/10.48550/arXiv.2210.06583)
2. S7 (Simplified SSM)
    * Wang, J., Zhu, W., Wang, P., Yu, X., Liu, L., & Saligrama, V. (2024). **S7: Simplified State Space Layers for Sequence Modeling**. *arXiv preprint arXiv:2410.03464*.
    * DOI: [10.48550/arXiv.2410.03464](https://doi.org/10.48550/arXiv.2410.03464)
3. DOST (Discrete Orthogonal Stockwell Transform)
    * Wang, Y., & Orchard, J. (2009). Fast Discrete Orthogonal Stockwell Transform. IEEE Transactions on Signal Processing, 57(9), 3615-3625.
    * (Note: 본 프로젝트에서는 이를 다차원 딥러닝 파이프라인에 맞게 근사 및 최적화하여 구현한 버전을 사용합니다.)

## 📝 라이선스 (License)
이 프로젝트는 **GNU Lesser General Public License v2.1 or later** 하에 배포됩니다.
This project is licensed under the **GNU LGPLv2.1+**. See the `LICENSE.txt` file for details.
