# Chapter 3 과제 — Q러닝

OpenAI Gym 내장 환경 `CliffWalking-v0` 에서 Q러닝 에이전트를 구현합니다.

| 파일 | 설명 |
|---|---|
| `play.py` | 키보드로 직접 해보기 |
| `q_learning.py` | 과제 파일. TODO 두 곳 |


## 진행 순서

```
python play.py          # 1. 먼저 손으로 해볼 것
python q_learning.py    # 2. 구현 후 실행
```

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

혹은

$$Q(s,a) \leftarrow Q(s,a) + \big[r +\max_{a'} Q(s',a')\big]$$

## 성공 기준

```
탐욕 정책: return -13.0, 13 스텝   (최적은 -13.0, 13 스텝)
```

정책 화살표가 절벽 바로 윗줄을 따라 오른쪽으로 이어져야 합니다.

```
python q_learning.py --plot     # 학습 곡선 저장
python q_learning.py --watch    # 정책을 창으로 재생 (pygame 필요)
```

## 실험

- `EPS_DECAY` 를 `0.999` / `0.9` 로 바꾸면?
- `GAMMA` 를 `0.9`, `0.5` 로 낮추면 절벽에 얼마나 가까이 붙는가?
- Q러닝 대신 SARSA(`max` 대신 실제 다음 행동의 Q)로 바꾸면 경로가 어떻게 달라지는가?

