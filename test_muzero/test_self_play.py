import time
import xqcpp
import numpy as np
import pytest
from gymxq.constants import BLACK_PLAYER, RED_PLAYER

from muzero.config import MuZeroConfig
from muzero.feature_utils import obs2feature
from muzero.mcts import GameHistory
from muzero.self_play import select_action


@pytest.mark.parametrize(
    "init_fen,min_root,max_root",
    [
        ("3k5/2P6/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 192", -1, -0.5),
        ("3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 192", 0.5, 1),
    ],
)
def test_selfplay(environment_factory, mcts_factory, init_fen, min_root, max_root):
    config = MuZeroConfig()
    config.num_simulations = 120

    config.init_fen = init_fen

    game_history = GameHistory()

    environment = environment_factory(config.init_fen, True)

    obs, info = environment.reset()
    observation = obs2feature(obs, flatten=False)
    game_history.observation_history.append(observation)
    to_play = info["to_play"]
    game_history.to_play_history.append(to_play)

    reset = False
    episode_steps = 0

    while not reset:
        legal_actions = info["legal_actions"]
        root = mcts_factory(config, legal_actions)
        action = select_action(root, 0)

        # action = select_action(root, 0)
        obs, reward, termination, truncation, info = environment.step(action)

        # 修改fen状态
        config.init_fen = info["fen"]

        episode_steps += 1

        observation = obs2feature(obs, flatten=False)

        game_history.store_search_statistics(root, config.action_space)

        # Next batch
        game_history.action_history.append(action)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(reward)
        to_play = info["to_play"]
        game_history.to_play_history.append(to_play)
        game_history.terminated_history.append(termination)
        game_history.truncated_history.append(truncation)

        reset = termination or truncation

    v = game_history.root_values[0]
    assert v > min_root and v < max_root
