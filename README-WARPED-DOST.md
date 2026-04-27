# S9 Integration: WarpedDOST (refit-per-N + fitter pattern)

## 설계 핵심

* **출력 채널 수가 입력 해상도와 무관하게 `C × n_per_axis^D`로 고정**된다.
  기존 `DOST`가 `(log₂(N)+1)^D × C`로 N에 의존했던 것과 대비된다.
* **학습 파라미터 0개.** `nn.Parameter`도 `register_buffer`도 사용하지 않으며,
  boundary는 plain `dict`에 저장된다 — optimizer가 보지 못한다.
* **Boundary는 입력 데이터의 spectrum에서 결정**된다 (equal-power quantile).
* **Calibration이 강제**된다. Calibrate 없이 `transform`을 부르면
  명확한 `RuntimeError`. silently fallback 없음.

## Calibration 인터페이스

`WarpedDOST`는 두 가지 calibration 경로를 제공한다 — 의미는 동일하고
사용 편의성만 다르다.

### One-shot

```python
t.fit(calibration_x)
```

calibration 데이터가 한 번에 메모리에 올라갈 때.

### Streaming (fitter property)

```python
f = t.fitter            # property; 매 access마다 새 fitter
for chunk in chunks:
    f.accumulate(chunk)
f.finalize()
```

데이터가 chunk로 분산되어 있거나 메모리 제약이 있을 때. 누적 buffer는
spatial shape당 `sum_d N_d` floats만 차지하므로 chunk 크기/개수와 무관하게
constant memory.

`t.fitter`는 매번 새 fitter 객체를 반환하므로 **동시 calibration 세션이
자연스럽게 분리**된다 — 분산 학습이나 멀티소스 calibration에서도 안전.

## 사용 예시

### 변환쌍 단독 (one-shot)

```python
import torch
from s9.transforms.warped_dost import WarpedDOST

t = WarpedDOST(D=2, n_per_axis=6)

# 학습 시작 전 한 번 calibrate
x_calib = next(iter(train_loader))[0]   # (B, C, 32, 32)
t.fit(x_calib)

# 사용
z = t(some_input)                        # (B, C * 36, 32, 32) complex

# 다른 해상도로 추론하려면 그 해상도에서 다시 calibrate
x_calib_eval = sample_unlabeled_batch_at(64, 64)
t.fit(x_calib_eval)
z_eval = t(eval_input_64x64)             # 같은 채널 수, 다른 해상도

# 역변환
inv = t.get_inverse_transform()
x_rec = inv(z)                           # (B, C, 32, 32) real
```

### 변환쌍 단독 (streaming, 메모리 절약)

```python
t = WarpedDOST(D=2, n_per_axis=6)

f = t.fitter
for x_chunk, _ in train_loader:
    f.accumulate(x_chunk)
f.finalize()

# 사용
z = t(some_input)
```

수치적으로 ``t.fit(torch.cat(chunks))``와 ``f = t.fitter; for c in chunks:
f.accumulate(c); f.finalize()``는 동일한 boundary를 산출한다 (검증됨).

### 동시 fitter 세션 (분산 / 멀티소스)

```python
t = WarpedDOST(D=2, n_per_axis=6)

# 두 calibration 소스가 동시에 진행될 때
fA = t.fitter
fB = t.fitter

for cA, cB in zip(loader_A, loader_B):
    fA.accumulate(cA)
    fB.accumulate(cB)

# 어느 한 쪽을 commit
fA.finalize()                            # bA가 t에 commit됨
# fB는 별도 사용 가능 (또는 commit하면 bB가 bA를 덮어씀)
```

### 예제: Classifier로 통합

```python
from s9.examples_warped import WarpedS9ClassifierModelExample

model = WarpedS9ClassifierModelExample(
    in_channels=3,
    d_model=64,
    n_layers=4,
    num_classes=10,
    spatial_shape=(32, 32),  # nominal — 어떤 spatial shape도 통과 가능
    n_per_axis=6,
)

# (a) one-shot
x_calib = next(iter(train_loader))[0]
model.calibrate(x_calib)

# (b) streaming
f = model.fitter
for x, _ in calib_loader:
    f.accumulate(x)
f.finalize()

# 학습
for x, y in train_loader:
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    ...

# 다른 해상도로 추론하려면 그 해상도에서 다시 calibrate
f = model.fitter
for x_chunk, _ in eval_loader_64:
    f.accumulate(x_chunk)
f.finalize()
logits = model(x_eval)
```

## 제약 조건

* **`n_per_axis ≤ N // 2 + 2`가 모든 spatial 축에 대해 성립해야 한다.**
  모델이 다룰 **최소 해상도**를 기준으로 `n_per_axis`를 정하는 것을 권장.
  예: 최소 N=16이면 `n_per_axis ≤ 9`. 일반적으로 `n_per_axis = 6~8`.
* **첫 사용 전 calibration 필수.** `fit()` 또는 `fitter.{accumulate,finalize}`
  를 부르지 않은 spatial shape에 대해 transform을 호출하면 `RuntimeError`.
* **하나의 fitter는 하나의 spatial shape에 lock된다.** 첫 `accumulate` 호출
  이 shape를 결정하고, 이후 다른 shape의 chunk를 accumulate하려 하면 reject.
  여러 shape을 calibrate하려면 각각 새 fitter를 생성.
* **finalize 후 fitter는 unusable.** 재사용하려면 `t.fitter`로 새 인스턴스 획득.
* `WarpedS9ClassifierModelExample`은 `input_proj`을 **eager init**한다.
  채널 수가 construction 시점에 결정되므로 lazy init이 불필요하며, 동일
  모델로 다양한 해상도를 forward할 수 있다.

## Streaming calibration의 메모리 footprint

Fitter의 누적 buffer 크기는 `sum_d N_d × 4 bytes` (float32). chunk 크기/개수와 무관.

| 설정 | One-shot (`fit`)에 필요한 메모리 | Fitter 누적 buffer |
|---|---|---|
| D=2, N=256, B=64, C=8 | ~32 MiB (calibration tensor 자체) | **2 KiB** |
| D=3, N=64, B=16, C=4 | ~16 MiB | **0.75 KiB** |
| D=2, N=1024, B=128, C=3 | ~1.5 GiB | **8 KiB** |

Peak 메모리는 fitter `accumulate` 한 번이 처리하는 chunk 한 개의 forward
pass에 필요한 만큼이다.

## Toy task ablation 결과 (참고)

별도 환경에서 실행한 4-class spectral classification (train at N=64,
test at N ∈ {64, 128, 256}, K=6):

| 방식 | acc@N=64 | acc@N=128 | acc@N=256 |
|---|---|---|---|
| Vanilla DOST | retraining 필요 | retraining 필요 | retraining 필요 |
| Log-uniform boundaries (no fit) | 0.62 | 0.58 | 0.38 |
| Fit once at N=64 | 0.95 | 0.43 | 0.31 |
| **Refit per N (이 패치)** | **0.95** | **0.92** | **0.91** |

추론 해상도마다 unlabeled batch로 재calibration하는 워크플로우가
cross-resolution generalization을 가장 잘 지킨다. 학습 파라미터는
모든 경우 0개.
