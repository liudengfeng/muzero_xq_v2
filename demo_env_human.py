"""human模式"""

import gymnasium as gym
import gymxq
import time

# from gymnasium.wrappers import StepAPICompatibility

init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 100 0 180"

env = gym.make(
    "xqv1",
    init_fen=init_fen,
    # render_mode="ansi",
    render_mode="human",
    gen_qp=True,
)

# env = StepAPICompatibility(env, False)
for _ in range(10):
    obs, info = env.reset()
    while True:
        action = env.sample_action()
        obs, r, t, tr, info = env.step(action)
        # obs, r, t, info = env.step(action)
        # render_mode="ansi",
        # print(env.render())
        env.render()
        if t or tr:
            break
        if tr:
            print(info["truncated"])
    time.sleep(3)
# env.close()
