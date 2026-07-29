# Chapter 3 과제 — Q러닝

OpenAI Gym 내장 환경 `CliffWalking-v0` 에서 Q러닝 에이전트를 구현합니다.

| 파일 | 설명 |
|---|---|
| `play.py` | 키보드로 직접 해보기 (터미널에 텍스트로 표시) |
| `play_human.py` | 키보드로 직접 해보기 (pygame 창으로 표시) |
| `q_learning.py` | 과제 파일. TODO 두 곳 |
| `watch_training.py` | 학습 과정을 창으로 지켜보기 (구현을 마친 뒤 실행) |

## 설치

```
pip install "numpy<2" gym
```

**`numpy<2` 의 따옴표를 빼지 마세요.** `gym` 은 2022년 이후 유지보수가 중단되어
NumPy 2.0 을 지원하지 않습니다. 그냥 `pip install numpy gym` 하면 NumPy 2.x 가 깔리고
실행 시 아래 오류가 납니다.

```
AttributeError: module 'numpy' has no attribute 'bool8'
```

이미 이 상태라면 `pip install "numpy<2"` 로 되돌리면 됩니다.

선택 사항 — 없어도 과제 수행에는 지장 없습니다.

| 패키지 | 필요한 경우 |
|---|---|
| `matplotlib` | `--plot` 으로 학습 곡선을 저장할 때 |
| `pygame` | `--watch`, `play_human.py`, `watch_training.py` — 창을 띄우는 것 전부 |

```
pip install matplotlib pygame
```

**Python 3.8 과 3.11 에서 동작을 확인했습니다.** 더 최신 버전에서는 `gym` 설치나 실행이
실패할 수 있습니다(2022년 이후 유지보수 중단). 그런 경우 파이썬을 3.11 로 맞추거나,
아래처럼 버전을 직접 지정해 설치하세요.

```
pip install "numpy<2" "gym==0.26.2"
```

## 진행 순서

```
python play.py            # 1. 먼저 손으로 해볼 것
python q_learning.py      # 2. 구현 후 실행
python watch_training.py  # 3. 학습 과정을 창으로 보기 (약 5분)
```

conda 를 쓴다면 터미널을 열 때마다 환경을 먼저 활성화하세요. 한 번만 하면
그 터미널에서는 계속 유효합니다.

```
conda activate <환경이름>
```

`play.py` 대신 `play_human.py` 를 쓰면 터미널 텍스트 대신 pygame 창이 뜹니다.
조작 방법은 같지만 **키는 창이 아니라 터미널에서 입력**합니다. 창을 클릭하면
키가 먹지 않습니다.

## 환경

```
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
S  C  C  C  C  C  C  C  C  C  C  T
```

`S` 시작(36) · `T` 목표(47) · `C` 절벽

| 항목 | 내용 |
|---|---|
| 상태 | `0` ~ `47` (`state = row * 12 + col`) |
| 행동 | `0` 위 · `1` 오른쪽 · `2` 아래 · `3` 왼쪽 |
| 보상 | 이동 `-1` · 절벽 `-100`(시작점 복귀, 종료 아님) |
| 종료 | `T` 도달 (또는 200스텝 초과 시 `truncated`) |
| **최적** | **13 스텝, return `-13.0`** |

```python
state, info = env.reset(seed=0)
next_state, reward, terminated, truncated, info = env.step(a)
```

`terminated` 와 `truncated` 는 구분해야 합니다. 시간초과는 진짜 종료가 아니므로
부트스트랩(`γ·max Q(s')`)을 그대로 유지합니다.

## 과제

`q_learning.py` 의 TODO 두 곳을 구현하세요.

**1. `select_action(Q, s, eps, rng)`** — ε-greedy. `eps` 확률로 무작위, 아니면 `argmax`.

**2. `update_q(Q, s, a, r, s2, terminated)`**

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$$

`terminated=True` 면 목표값이 `r` 입니다.

## 성공 기준

```
탐욕 정책: return -13.0, 13 스텝   (최적은 -13.0, 13 스텝)
```

정책 화살표가 절벽 바로 윗줄을 따라 오른쪽으로 이어져야 합니다.

```
python q_learning.py --plot     # 학습 곡선 저장
python q_learning.py --watch    # 완성된 정책을 창으로 재생 (pygame 필요)
```

## 학습 과정 지켜보기

`--watch` 는 **다 학습된 뒤의 정책**만 재생합니다. 학습이 진행되는 과정을 보려면
`watch_training.py` 를 실행하세요. 옵션 없이 그냥 실행하면 됩니다.

```
python watch_training.py
```

| 에피소드 | ε | 무엇이 보이나 |
|---|---|---|
| 1 | 1.00 | 사방으로 무작위. 절벽에 계속 빠진다 |
| 100 | 0.37 | 목표에 도달은 하는데 위쪽으로 크게 돌아간다 |
| 300 | 0.05 | 절벽 윗줄에 붙어서 간다 |
| 500 | 0.01 | 13스텝 최단 경로 |

500 에피소드는 총 2만 스텝이 넘어서 전부 그리면 90분이 걸리고, 그중 3분의 2가
초반의 무작위 구간입니다. 그래서 **10 에피소드마다 하나씩, 에피소드당 최대
50스텝까지만** 보여 줍니다. 약 5분 걸립니다. 이 값들은 파일 상단에서 바꿀 수 있습니다.

```python
WATCH_EVERY = 10        # 몇 에피소드마다 창으로 볼지
RENDER_FPS = 6          # 초당 몇 스텝을 그릴지 (gym 기본값은 4)
MAX_RENDER_STEPS = 50   # 한 에피소드에서 최대 몇 스텝까지 그릴지
PAUSE = 1.2             # 에피소드 사이 정지 시간(초)
```

## 실험

- `EPS_DECAY` 를 `0.999` / `0.9` 로 바꾸면?
- `GAMMA` 를 `0.9`, `0.5` 로 낮추면 절벽에 얼마나 가까이 붙는가?
- Q러닝 대신 SARSA(`max` 대신 실제 다음 행동의 Q)로 바꾸면 경로가 어떻게 달라지는가?

## 참고

`Gym has been unmaintained since 2022...` 경고는 무시해도 됩니다.
