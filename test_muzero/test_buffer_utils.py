import numpy as np
import pytest
import xqcpp
from gymxq.constants import BLACK_PLAYER, NUM_ACTIONS, RED_PLAYER, NUM_COL, NUM_ROW

from muzero.replay_buffer_utils import (
    Buffer,
    make_target,
    sample_position,
    update_gamehistory_priorities,
)
from muzero.config import MuZeroConfig, PLANE_NUM
from muzero.mcts import GameHistory, Node, backpropagate, MinMaxStats, render_root


def fake_observation_history(n):
    res = []
    for i in range(n):
        res.append(np.full((1, PLANE_NUM, NUM_ROW, NUM_COL), i, dtype=np.float32))
    return res


def simulate_game_history(render_mcts=False):
    # "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190"
    # 2818 -> 6151
    moves = ["2818", "6151"]
    config = MuZeroConfig()
    # 便于观察
    config.discount_factor = 0.95

    gh = GameHistory()
    gh.to_play_history = [1, 2, 1]
    n = len(gh.to_play_history)
    gh.observation_history = fake_observation_history(n)
    gh.reward_history = [0, 0, 1]
    gh.action_history = [NUM_ACTIONS] + [xqcpp.m2a(m) for m in moves]

    root = Node(0)
    min_max_stats = MinMaxStats()

    # 一次迭代
    action = xqcpp.m2a(moves[0])
    root.expand(
        [action],
        1,
        0,
        {action: 1.0},
        None,
        True,
        True,
    )
    node1 = root.children[action]

    reward = 0
    next_action = xqcpp.m2a(moves[1])
    node1.expand(
        [next_action],
        2,
        reward,
        {next_action: 1.0},
        None,
        True,
        True,
    )
    search_path = [root, node1]
    # V(s,a)
    value = 1
    backpropagate(search_path, value, 2, min_max_stats, config)
    gh.store_search_statistics(root, config.action_space)

    # 二次迭代
    node2 = node1.children[next_action]
    reward = 1
    node2.expand(
        config.action_space,
        1,
        reward,
        {},
        None,
        True,
    )
    search_path = [root, node1, node2]
    value = 0
    backpropagate(search_path, value, 1, min_max_stats, config)
    gh.store_search_statistics(root, config.action_space)
    if render_mcts:
        # 演示根
        render_root(root, "test", "svg", "test_mcts_tree")

    gh.terminated_history = [False] * (n - 1) + [True]
    gh.truncated_history = [False] * n
    return gh


def test_update_gamehistory_priorities():
    config = MuZeroConfig()
    gh = simulate_game_history()
    assert gh.priorities is None
    assert gh.game_priority is None

    update_gamehistory_priorities(gh, config)
    np.testing.assert_array_almost_equal(gh.priorities, [0.22, 1.40], 2)
    np.testing.assert_approx_equal(gh.game_priority, 1.40, 2)


def test_buffer():
    config = MuZeroConfig()
    config.batch_size = 1
    config.PER = True
    gh = simulate_game_history()

    assert gh.priorities is None
    assert gh.game_priority is None

    buffer = Buffer(config)
    buffer.save_game(gh)

    gh_buf = buffer.buffer[0]

    # 更新优先度
    assert gh_buf.priorities is not None
    assert gh_buf.game_priority is not None

    index_batch, (
        observation_batch,
        action_batch,
        value_batch,
        reward_batch,
        policy_batch,
        weight_batch,
        gradient_scale_batch,
    ) = buffer.get_batch()
    assert weight_batch.shape == (config.batch_size,)
