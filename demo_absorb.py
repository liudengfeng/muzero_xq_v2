"""验证吸收状态设计"""
from muzero.config import PLANE_NUM, MuZeroConfig
from muzero.self_play import SelfPlay
from muzero.replay_buffer_utils import make_target
import xqcpp
import numpy as np
import pickle


def view_policy(cvs, title):
    print(f"{title} {len(cvs)=}")
    for i, cv in enumerate(cvs):
        pi = {}
        for a, prob in enumerate(cv):
            if prob > 0.0:
                pi[xqcpp.m2a(a)] = round(prob, 2)
        print(pi)
        print("合计", sum(pi.values()))
        print("=" * 30)


# def rollout():
config = MuZeroConfig()
init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190"
config.num_simulations = 60
player = SelfPlay(config, init_fen=init_fen)
gh = player.rollout(1)
view_policy(gh.child_visits, "rollout")

# 修改政策及根值
gh.root_values = [0.999]
moves_policy = {
    "2818": 0.005,
    "2829": 0.49,
    "2838": 0.49,
    "4838": 0.005,
    "4849": 0.005,
    "4858": 0.005,
}
sum(moves_policy.values())
actions_policy = [0] * 2086
for k, v in moves_policy.items():
    actions_policy[xqcpp.m2a(k)] = v
gh.child_visits[0] = actions_policy

target_values, target_rewards, target_policies, actions = make_target(gh, 0, config)
view_policy(target_policies, "target")
print()
