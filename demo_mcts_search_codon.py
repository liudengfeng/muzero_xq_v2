import time

import gymnasium as gym
import gymxq
import torch
import xqcpp

from muzero.apply import get_pretrained_model
from muzero.codon_tree_search import codon_mcts_search
from muzero.config import MuZeroConfig
from muzero.feature_utils import obs2feature
from muzero.mcts import MCTS

config = MuZeroConfig()
config.runs = 9
config.num_simulations = 60
config.init_fen = "2b6/3ka4/2Pa5/3N5/9/3R5/9/9/5pr2/3AK4 r - 110 0 190"
config.restore_from_latest_checkpoint = True

environment = gym.make("xqv1", init_fen=config.init_fen, render_mode="ansi")
obs, info = environment.reset()
observation = obs2feature(obs, flatten=False)
legal_actions = info["legal_actions"]
to_play = info["to_play"]

model = get_pretrained_model(config)


N = 3


def use_codon():
    with torch.no_grad():
        for _ in range(N):
            root, mcts_info = codon_mcts_search(
                config, model, observation, legal_actions, to_play, True
            )
    print("Root value = {:.2f} mcts_info = {}".format(root.value(), mcts_info))


def no_codon():
    with torch.no_grad():
        for _ in range(N):
            root, mcts_info = MCTS(config).run(
                model, observation, legal_actions, to_play, True
            )
    print("Root value = {:.2f} mcts_info = {}".format(root.value(), mcts_info))


start = time.time()
no_codon()
print("no_codon duration={:.2f}".format(time.time() - start))

start = time.time()
use_codon()
print("use_codon duration={:.2f}".format(time.time() - start))
