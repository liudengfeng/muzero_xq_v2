"""演示录制视频"""

import gymnasium as gym
import gymxq
import time
from gymnasium.wrappers.record_video import RecordVideo

init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 100 0 180"

env = gym.make(
    "xqv1",
    init_fen=init_fen,
    render_mode="rgb_array",
    gen_qp=True,
)

env = RecordVideo(
    env,
    "demo_videos",
    episode_trigger=lambda x: x % 2 == 0,
    disable_logger=True,
    name_prefix="test_play",
)

# env = StepAPICompatibility(env, False)
for _ in range(10):
    obs, info = env.reset()
    while True:
        action = env.sample_action()
        obs, r, t, tr, info = env.step(action)
        env.render()
        if t or tr:
            break
    # time.sleep(3)
# env.close()
