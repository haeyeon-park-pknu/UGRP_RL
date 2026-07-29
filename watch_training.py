"""
학습 과정을 창으로 지켜보기 — CliffWalking-v0

q_learning.py 의 TODO 두 곳을 채운 뒤에 실행하세요.
이 파일은 여러분이 구현한 select_action / update_q 를 그대로 가져다 씁니다.

    python watch_training.py

옵션 없습니다. 그냥 실행하면 약 5분간 창이 뜨고, 10 에피소드마다 하나씩
에이전트가 학습해 가는 과정을 보여 줍니다. 끝나면 완성된 정책을 한 번
재생합니다.

무엇을 보면 되나:
  에피소드   1  ε=1.00  사방으로 무작위. 절벽에 계속 빠진다.
  에피소드 100  ε=0.37  목표에 도달은 하는데 위쪽으로 크게 돌아간다.
  에피소드 300  ε=0.05  절벽 윗줄에 붙어서 간다.
  에피소드 500  ε=0.01  13스텝 최단 경로.

왜 전부 보여 주지 않나:
  500 에피소드는 총 2만 스텝이 넘습니다. 전부 그리면 90분이 걸리고,
  그중 3분의 2가 초반의 무작위 구간입니다. 그래서 10 에피소드마다,
  에피소드당 최대 50스텝까지만 그립니다.
"""

import os
import time
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import gym

try:
    import pygame
except ImportError:
    raise SystemExit("이 스크립트는 pygame 이 필요합니다:  pip install pygame")

from q_learning import (select_action, update_q, show_policy, run_greedy,
                        EPISODES, EPS_START, EPS_MIN, EPS_DECAY, MAX_STEPS)

# ----------------------------------------------------------------------
# 관찰 설정 — 이 네 값이 전체 소요 시간을 정합니다 (현재 약 4.6분)
# ----------------------------------------------------------------------
WATCH_EVERY = 10        # 몇 에피소드마다 창으로 볼지
RENDER_FPS = 6          # 초당 몇 스텝을 그릴지 (gym 기본값은 4)
MAX_RENDER_STEPS = 50   # 한 에피소드에서 최대 몇 스텝까지 그릴지
PAUSE = 1.2             # 에피소드 사이 정지 시간(초)


def check_implemented():
    """TODO 를 채우지 않고 실행했을 때 안내 메시지를 보여 준다."""
    try:
        select_action(np.zeros((1, 4)), 0, 0.0, np.random.RandomState(0))
        update_q(np.zeros((1, 4)), 0, 0, -1.0, 0, False)
    except NotImplementedError:
        raise SystemExit(
            "\nq_learning.py 의 TODO 두 곳(select_action, update_q)을 먼저 "
            "구현하세요.\n이 스크립트는 그 두 함수를 가져다 씁니다.\n")


def wait(seconds):
    """창이 '응답 없음'이 되지 않도록 이벤트를 돌리면서 기다린다."""
    end = time.time() + seconds
    while time.time() < end:
        try:
            pygame.event.pump()
        except pygame.error:
            pass
        time.sleep(0.02)


def train_watching(episodes=EPISODES):
    # render_mode="human" 으로 만들어 두고, 관찰하지 않는 구간에서는
    # unwrapped.render_mode 를 None 으로 꺼 둔다. 이것만으로 step() 안의
    # 자동 렌더링이 켜졌다 꺼진다.
    env = gym.make("CliffWalking-v0", render_mode="human",
                   max_episode_steps=MAX_STEPS)
    # metadata 는 클래스 공용 딕셔너리라 그대로 고치면 다른 환경까지 영향을
    # 받는다. 이 인스턴스용으로 복사한 뒤 바꾼다.
    env.unwrapped.metadata = dict(env.unwrapped.metadata)
    env.unwrapped.metadata["render_fps"] = RENDER_FPS

    rng = np.random.RandomState(0)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    eps, returns = EPS_START, []

    for ep in range(episodes):
        watching = (ep + 1) % WATCH_EVERY == 0 or ep == 0
        env.unwrapped.render_mode = "human" if watching else None
        if watching:
            print("\n[에피소드 %3d / %d]  eps %.2f" % (ep + 1, episodes, eps))

        s, _ = env.reset(seed=ep)
        total, steps, done, cut = 0.0, 0, False, False
        while not done:
            a = select_action(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            update_q(Q, s, a, r, s2, term)
            s, total, steps, done = s2, total + r, steps + 1, term or trunc

            # 너무 길어지면 화면만 끄고 학습은 계속한다
            if watching and not done and steps >= MAX_RENDER_STEPS:
                env.unwrapped.render_mode = None
                watching, cut = False, True

        returns.append(total)
        eps = max(EPS_MIN, eps * EPS_DECAY)

        if cut or (ep + 1) % WATCH_EVERY == 0 or ep == 0:
            print("   %d 스텝, return %+.1f%s"
                  % (steps, total,
                     "   (%d스텝까지만 표시)" % MAX_RENDER_STEPS if cut else ""))
            wait(PAUSE)

        if (ep + 1) % 100 == 0:
            print("   -- 최근 100회 평균 return %.1f --" % np.mean(returns[-100:]))

    env.close()
    return Q, returns


if __name__ == "__main__":
    check_implemented()

    print(__doc__)
    print("시작합니다. 창을 닫지 마세요. 약 5분 걸립니다.\n")

    Q, returns = train_watching()
    show_policy(Q)

    total, steps = run_greedy(Q)
    print("\n탐욕 정책: return %+.1f, %d 스텝   (최적은 -13.0, 13 스텝)"
          % (total, steps))

    print("\n학습된 정책을 마지막으로 한 번 재생합니다...")
    wait(1.5)
    run_greedy(Q, render_mode="human")
    print("\n끝났습니다.\n")
