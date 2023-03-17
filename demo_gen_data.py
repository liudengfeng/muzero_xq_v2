import time
from gymnasium.wrappers.record_video import RecordVideo
import gymnasium as gym
import gymxq
import xqcpp


def print_actions(info):
    print(info["legal_actions"])
    print([xqcpp.m2a(a) for a in info["legal_actions"]])
    print("=" * 30)


def gen_data(init_fen):
    environment = gym.make(
        "xqv1",
        init_fen=init_fen,
        gen_qp=True,
        render_mode="rgb_array",
    )
    environment = RecordVideo(
        environment,
        "demo_videos",
        episode_trigger=lambda x: True,
        disable_logger=True,
    )

    o, info = environment.reset()
    print_actions(info)
    while True:
        action = environment.sample_action()
        print("action = {} move = {}".format(action, xqcpp.m2a(action)))
        o, r, t, tr, info = environment.step(action)
        print_actions(info)
        if t or tr:
            break


gen_data("5a3/5k3/5a3/9/9/9/9/3C5/9/4K4 r - 110 0 180")
