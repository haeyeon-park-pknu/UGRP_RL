"""
Q러닝 구현 과제 — CliffWalking-v0

TODO 두 곳을 채우세요. 나머지는 건드릴 필요 없습니다.

    python q_learning.py            학습하고 결과 출력
    python q_learning.py --plot     학습 곡선을 learning_curve.png 로 저장
    python q_learning.py --watch    학습된 정책을 창으로 재생

환경: 4x12 격자, 상태 0~47, 행동 0=위 1=오른쪽 2=아래 3=왼쪽
보상: 이동 -1, 절벽 -100(시작점 복귀), 목표 도달 시 종료
최적: 13 스텝, return -13.0
"""

import argparse
import os
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import gym

EPISODES = 500
ALPHA, GAMMA = 0.1, 0.99
EPS_START, EPS_MIN, EPS_DECAY = 1.0, 0.01, 0.99
MAX_STEPS = 200


# ----------------------------------------------------------------------
# TODO 1. ε-greedy 행동 선택
#   eps 확률로 무작위, 아니면 Q값이 가장 큰 행동을 반환한다.
#   힌트: rng.random(), rng.randint(Q.shape[1]), np.argmax(Q[s])
# ----------------------------------------------------------------------
def select_action(Q, s, eps, rng):
    raise NotImplementedError("select_action 을 구현하세요")


# ----------------------------------------------------------------------
# TODO 2. Q값 갱신
#   Q(s,a) <- r + max Q(s',a')
#   terminated=True 면 다음 상태의 가치가 없으므로 목표값은 r 이다.
# ----------------------------------------------------------------------
def update_q(Q, s, a, r, s2, terminated):
    raise NotImplementedError("update_q 를 구현하세요")


# ----------------------------------------------------------------------
# 아래는 수정하지 않아도 됩니다
# ----------------------------------------------------------------------
def train():
    env = gym.make("CliffWalking-v0", max_episode_steps=MAX_STEPS)
    rng = np.random.RandomState(0)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    eps, returns = EPS_START, []

    for ep in range(EPISODES):
        s, _ = env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            a = select_action(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            update_q(Q, s, a, r, s2, term)
            s, total, done = s2, total + r, term or trunc

        returns.append(total)
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if (ep + 1) % 100 == 0:
            print("에피소드 %4d | 최근 100회 평균 return %8.2f | eps %.3f"
                  % (ep + 1, np.mean(returns[-100:]), eps))

    env.close()
    return Q, returns


def show_policy(Q):
    arrows = "^>v<"
    print("\n학습된 정책 (C 절벽, T 목표):")
    for r in range(4):
        row = []
        for c in range(12):
            if r == 3:
                row.append("T" if c == 11 else ("S" if c == 0 else "C"))
            else:
                row.append(arrows[int(np.argmax(Q[r * 12 + c]))])
        print("  " + " ".join(row))


def run_greedy(Q, render_mode=None):
    env = gym.make("CliffWalking-v0", max_episode_steps=MAX_STEPS,
                   render_mode=render_mode)
    s, _ = env.reset(seed=12345)
    total, steps, done = 0.0, 0, False
    while not done:
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
        total, steps, done = total + r, steps + 1, term or trunc
    env.close()
    return total, steps


def plot(returns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = 20
    smooth = np.convolve(returns, np.ones(w) / w, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(returns, alpha=.3, label="episode return")
    plt.plot(range(w - 1, len(returns)), smooth, label="moving avg (%d)" % w)
    plt.axhline(-13, ls="--", c="gray", label="optimal (-13)")
    plt.xlabel("episode"); plt.ylabel("return"); plt.legend()
    plt.title("Q-learning on CliffWalking-v0")
    plt.tight_layout(); plt.savefig("learning_curve.png", dpi=120)
    print("학습 곡선 -> learning_curve.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="학습 곡선 저장")
    ap.add_argument("--watch", action="store_true", help="정책을 창으로 재생")
    args = ap.parse_args()

    Q, returns = train()
    show_policy(Q)

    total, steps = run_greedy(Q)
    print("\n탐욕 정책: return %+.1f, %d 스텝   (최적은 -13.0, 13 스텝)"
          % (total, steps))

    if args.plot:
        plot(returns)
    if args.watch:
        run_greedy(Q, render_mode="human")
