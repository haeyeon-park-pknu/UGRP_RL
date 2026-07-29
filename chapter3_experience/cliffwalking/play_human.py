"""
CliffWalking 직접 조작해 보기 — pygame 창 버전

원본 play.py 는 render_mode="ansi" 라 터미널에 텍스트 격자를 그립니다.
이 파일은 render_mode="human" 으로 pygame 창을 띄웁니다.

    python play_human.py

키 입력은 여전히 "터미널"이 받습니다. 창이 아니라 터미널을 포커스한 채로
누르세요. (창을 클릭하면 키가 터미널로 가지 않습니다.)

W 위 / A 왼쪽 / S 아래 / D 오른쪽 | R 다시 / Q 종료

human 모드에서 달라지는 점 두 가지:
  1. env.render() 가 None 을 반환한다. 그리기는 step()/reset() 안에서
     자동으로 일어나므로 print(env.render()) 를 하면 안 된다.
  2. 키를 기다리며 블로킹하는 동안 pygame 이벤트를 처리하지 않으면
     Windows 가 창을 "응답 없음"으로 만든다. 그래서 아래 read_key() 는
     블로킹 대신 kbhit() 로 폴링하면서 event.pump() 를 돌린다.
"""

import os
import time
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore")

import gym

try:
    import pygame
except ImportError:
    raise SystemExit("이 스크립트는 pygame 이 필요합니다:  pip install pygame\n"
                     "(설치하지 않으려면 play.py 를 쓰세요. 조작은 같습니다.)")

KEYS = {"w": 0, "d": 1, "s": 2, "a": 3}
NAMES = ["위", "오른쪽", "아래", "왼쪽"]


def read_key():
    """창을 살려 둔 채로 터미널 키 입력을 기다린다."""
    try:
        import msvcrt
    except ImportError:  # 윈도우가 아니면 그냥 Enter 방식으로
        return (input("키 입력 후 Enter > ").strip().lower() or " ")[0]

    while True:
        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
        try:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return "q"
        except pygame.error:
            pass
        time.sleep(0.02)


def show(s, msg, total, steps, best):
    os.system("cls" if os.name == "nt" else "clear")
    print("격자는 pygame 창에 표시됩니다. (키는 이 터미널에서 입력)")
    print("W 위 / A 왼쪽 / S 아래 / D 오른쪽 | R 다시 / Q 종료")
    print("-" * 46)
    print("상태 %2d | 스텝 %3d | 누적 보상 %+8.1f%s"
          % (s, steps, total, "" if best is None else " | 최고 %+.1f" % best))
    print(msg)


def main():
    env = gym.make("CliffWalking-v0", render_mode="human", max_episode_steps=200)
    s, _ = env.reset(seed=None)
    total, steps, best = 0.0, 0, None
    msg = "움직여 보세요. 목표(T)까지 최소 몇 스텝일까요?"

    while True:
        show(s, msg, total, steps, best)
        key = read_key()

        if key == "q":
            break
        if key == "r":
            s, _ = env.reset(seed=None)
            total, steps, msg = 0.0, 0, "다시 시작합니다."
            continue
        if key not in KEYS:
            msg = "W, A, S, D 중 하나를 누르세요."
            continue

        a = KEYS[key]
        s, r, term, trunc, _ = env.step(a)
        total, steps = total + r, steps + 1
        msg = "행동: %s, 보상: %+.0f%s" % (
            NAMES[a], r, "  → 절벽! 시작점으로 돌아갑니다." if r == -100 else "")

        if term or trunc:
            show(s, msg, total, steps, best)
            print("\n%s  %d 스텝, 최종 return %+.1f"
                  % ("목표 도달!" if term else "시간 초과.", steps, total))
            if term and total < -13:
                print("최적은 13 스텝, return -13.0 입니다. 더 줄일 수 있을까요?")
            elif term:
                print("최적 경로입니다. Q러닝도 결국 이 경로를 찾아내야 합니다.")
            best = total if best is None else max(best, total)

            print("\n아무 키나 누르면 다시 시작합니다. (Q 는 종료)")
            if read_key() == "q":
                break
            s, _ = env.reset(seed=None)
            total, steps, msg = 0.0, 0, "다시 시작합니다."

    env.close()
    print("\n이제 q_learning.py 를 열어 에이전트를 구현하세요.\n")


if __name__ == "__main__":
    main()
