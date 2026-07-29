"""
CliffWalking 직접 조작해 보기

Q러닝을 구현하기 전에 먼저 실행해서 환경을 손으로 움직여 보세요.
목표까지 최소 몇 스텝인지, 절벽에 빠지면 어떻게 되는지 직접 확인하는 것이 목적입니다.

    python play.py

W 위 / A 왼쪽 / S 아래 / D 오른쪽 | R 다시 / Q 종료
"""

import os
import warnings

warnings.filterwarnings("ignore")

import gym

KEYS = {"w": 0, "d": 1, "s": 2, "a": 3}
NAMES = ["위", "오른쪽", "아래", "왼쪽"]


def read_key():
    try:
        import msvcrt
    except ImportError:
        return (input("키 입력 후 Enter > ").strip().lower() or " ")[0]
    return msvcrt.getwch().lower()


def show(env, s, msg, total, steps, best):
    os.system("cls" if os.name == "nt" else "clear")
    print(env.render())
    print("(o 빈칸 / C 절벽 / T 목표 / x 현재위치)")
    print("W 위 / A 왼쪽 / S 아래 / D 오른쪽 | R 다시 / Q 종료")
    print("-" * 46)
    print("상태 %2d | 스텝 %3d | 누적 보상 %+8.1f%s"
          % (s, steps, total, "" if best is None else " | 최고 %+.1f" % best))
    print(msg)


def main():
    env = gym.make("CliffWalking-v0", render_mode="ansi", max_episode_steps=200)
    s, _ = env.reset(seed=None)
    total, steps, best = 0.0, 0, None
    msg = "움직여 보세요. 목표(T)까지 최소 몇 스텝일까요?"

    while True:
        show(env, s, msg, total, steps, best)
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
            show(env, s, msg, total, steps, best)
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
