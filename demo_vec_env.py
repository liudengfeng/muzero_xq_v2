import gymxq
import gymnasium as gym
import time
import numpy as np
from gymxq.constants import NUM_ACTIONS
import random
from muzero.feature_utils import show_board_by_pieces
import xqcpp


def view_envs_step(steps, actions, obs, rewards, terminations, truncation, infos):
    """演示矢量环境

    Args:
        steps (int): 步数
        actions (array): 1d 移动
        obs (dict): 观察信息字典
        rewards (array): 1d 即时奖励
        terminations (array): 1d 终止标记
        truncation (array): 1d 截断标记
        infos (dict): 补充信息
    """
    print(f"{steps=}")
    n = len(terminations)
    key = "final_observation"
    for i in range(n):
        print("环境 {}".format(i + 1))
        pieces = obs["s"][i].reshape(10, 9)
        to_play = obs["to_play"][i]
        show_board_by_pieces(pieces, to_play)
        if key in infos.keys():
            f = infos["final_observation"][i]
            if f is None:
                print("未重置")
            else:
                print("重置状态" if terminations[i] else "截断状态")
                pieces = f["s"].reshape(10, 9)
                show_board_by_pieces(pieces, to_play)
        else:
            print("未重置")
        print(
            "action = {} move = {} reward = {:.2f} terminal = {} truncation = {}".format(
                actions[i],
                xqcpp.m2a(actions[i]),
                rewards[i],
                terminations[i],
                truncation[i],
            )
        )
        print("-" * 30)
    print("=" * 60)


# 步数较少
init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190"
workers = 1
envs = gym.vector.make(
    "xqv1", init_fen=init_fen, num_envs=workers, render_mode="rgb_array"
)

obs, infos = envs.reset()

print(obs)

actions = [643]
observations, rewards, terminations, truncation, infos = envs.step(actions)

# for action in actions:
#     obs, rewards, terminations, truncation, infos = envs.step(actions)
#     print(infos["l"])
#     steps += 1
#     view_envs_step(steps, actions, obs, rewards, terminations, truncation, infos)

# steps = 0
# while steps < 12:
#     legal_actions = infos["legal_actions"]
#     actions = [random.choice(legal_actions[i]) for i in range(workers)]
#     obs, rewards, terminations, truncation, infos = envs.step(actions)
#     print(infos["l"])
#     steps += 1
#     view_envs_step(steps, actions, obs, rewards, terminations, truncation, infos)
import numpy as np

a = [1, 2, 4]
np.asarray(a)
