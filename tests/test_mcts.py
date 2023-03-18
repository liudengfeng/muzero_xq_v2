import gymnasium as gym
import pytest
import xqcpp
from gymxq.constants import BLACK_PLAYER, MAX_EPISODE_STEPS

from muzero.checkpoint import get_current_checkpoint
from muzero.config import MuZeroConfig
from muzero.feature_utils import obs2feature
from muzero.mcts import GameHistory, backpropagate, Node, MinMaxStats


@pytest.mark.parametrize(
    "moves,expected_len",
    [
        (["2838"], 2),
    ],
)
def test_game_histroy(moves, expected_len):
    environment = gym.make(
        "xqv1", init_fen="3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 180"
    )
    actions = [xqcpp.m2a(m) for m in moves]
    game_history = GameHistory()
    obs, info = environment.reset()
    observation = obs2feature(obs, flatten=False)
    game_history.observation_history.append(observation)
    to_play = info["to_play"]
    game_history.to_play_history.append(to_play)

    reset = False
    i = 0
    while not reset and len(game_history.action_history) < MAX_EPISODE_STEPS:
        action = actions[i]
        obs, reward, termination, truncation, info = environment.step(action)

        observation = obs2feature(obs, flatten=False)

        game_history.child_visits.append(0)
        game_history.root_values.append(0)

        # Next batch
        game_history.action_history.append(action)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(reward)
        to_play = info["to_play"]
        game_history.to_play_history.append(to_play)
        game_history.terminated_history.append(termination)
        game_history.truncated_history.append(truncation)

        reset = termination or truncation

    assert len(game_history.reward_history) == expected_len


@pytest.mark.parametrize(
    "to_plays,rewards,muzero_reward,expected",
    [
        ([1, 2], [1], 1, {"muzero_reward": 1, "opponent_reward": 0}),
        ([1, 2], [1], 2, {"muzero_reward": 0, "opponent_reward": 1}),
        ([1, 2, 1], [0, 1], 1, {"muzero_reward": 0, "opponent_reward": 1}),
        ([1, 2, 1, 2], [0, 0, -1], 1, {"muzero_reward": -1, "opponent_reward": 0}),
    ],
)
def test_reward_pair(to_plays, rewards, muzero_reward, expected):
    # to_play  [1, 2, 1]
    #           ↘️ ↘️
    # reward   [0, 0, 1]
    gh = GameHistory()
    gh.to_play_history = to_plays
    gh.reward_history.extend(rewards)
    actual = gh.get_reward_pair(muzero_reward)
    assert actual == expected


@pytest.mark.parametrize(
    "init_fen,init_legal_actions,expected",
    [
        (
            "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 110 0 180",
            [632, 643, 641, 1129, 1142, 1139],
            [641, 643],
        ),
        (
            "5a3/5k3/5a3/9/9/9/9/3C5/9/4K4 r - 110 0 180",
            [
                920,
                931,
                922,
                726,
                727,
                719,
                721,
                724,
                737,
                740,
                742,
                743,
                744,
                728,
                729,
                730,
                731,
                732,
                733,
                734,
            ],
            [733],
        ),
    ],
)
def test_mcts_simulate(mcts_factory, init_fen, init_legal_actions, expected):
    config = MuZeroConfig()
    config.num_simulations = 60
    config.init_fen = init_fen

    p = 1 / len(init_legal_actions)
    init_policy = {a: p for a in init_legal_actions}
    old = sum([init_policy[a] for a in expected])

    root = mcts_factory(config, init_legal_actions)
    sum_visits = sum(child.visit_count for child in root.children.values())
    new_policy = {
        a: child.visit_count / sum_visits for a, child in root.children.items()
    }
    new = sum([new_policy[a] for a in expected])
    assert new > 0.75
    assert new > 1.5 * old
    assert root.value() > 0.75


def test_backpropagate_1():
    # 红先
    config = MuZeroConfig()
    min_max_stats = MinMaxStats()
    root = Node(0)
    moves = ["2838", "2829"]
    legal_actions = [xqcpp.m2a(m) for m in moves]
    policy = {xqcpp.m2a(m): 0.5 for m in moves}
    root.expand(legal_actions, 1, 0, policy, None, True, True)

    for i in range(len(moves)):
        search_path = [root]
        action = xqcpp.m2a(moves[i])
        node = root.children[action]
        to_play = 2
        search_path.append(node)
        # 终止状态 value 设为 0
        value, reward, policy = 0, 1, {}
        node.expand(config.action_space, to_play, reward, policy, None, True, True)
        backpropagate(search_path, value, to_play, min_max_stats, config)
    assert root.value() == 1.0


def test_backpropagate_2():
    # 黑先
    config = MuZeroConfig()
    min_max_stats = MinMaxStats()
    root = Node(0)
    moves = ["6151", "6160"]
    legal_actions = [xqcpp.m2a(m) for m in moves]
    policy = {xqcpp.m2a(m): 0.5 for m in moves}
    root.expand(legal_actions, 2, 0, policy, None, True, True)

    for i in range(len(moves)):
        search_path = [root]
        action = xqcpp.m2a(moves[i])
        node = root.children[action]
        to_play = 1
        search_path.append(node)
        # 终止状态 value 设为 0
        value, reward, policy = 0, 1, {}
        node.expand(config.action_space, to_play, reward, policy, None, True, True)
        backpropagate(search_path, value, to_play, min_max_stats, config)
    assert root.value() == 1.0
