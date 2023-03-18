import pytest
import gymxq
import gymnasium as gym
import time
import xqcpp
from gymxq.envs.game import Game
from gymxq.constants import (
    NUM_COL,
    NUM_ROW,
    BLACK_PLAYER,
    MAX_NUM_NO_EAT,
    MAX_EPISODE_STEPS,
)


@pytest.mark.parametrize(
    "moves,expected",
    [
        (["2838"], 1),
        (["2818", "6151"], 1),
    ],
)
def test_game_reward(moves, expected):
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190"
    g = Game(init_fen, False)
    g.reset()
    r = 0
    t = False
    for m in moves:
        a = xqcpp.movestr2action(m)
        _, r, t = g.step(a)
    assert expected == r
    assert t


# def test_env_v0():
#     init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 100 0 180"
#     env = gym.make(
#         "xqv0",
#         init_fen=init_fen,
#         render_mode="human",
#         gen_qp=True,
#     )
#     env.reset()
#     action = env.sample_action()
#     env.step(action)
#     time.sleep(2)


def test_env_v1_ansi_1():
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 119 0 180"
    env = gym.make(
        "xqv1",
        init_fen=init_fen,
        render_mode="ansi",
        # gen_qp=True,
    )
    env.reset()
    # print(env.render())
    # action = env.sample_action()
    # 连续未吃子判和
    action = 2079
    observation, reward, terminated, truncated, info = env.step(action)
    assert isinstance(observation, dict)
    assert observation["s"].shape == (NUM_ROW * NUM_COL,)
    assert observation["steps"] == 180
    assert observation["continuous_uneaten"] == MAX_NUM_NO_EAT
    assert observation["to_play"] == BLACK_PLAYER
    assert reward == 0
    assert terminated == True
    assert truncated == False
    # print(env.render())


def test_env_v1_ansi_2():
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 119 0 190"
    env = gym.make(
        "xqv1",
        init_fen=init_fen,
        render_mode="ansi",
        # gen_qp=True,
    )
    env.reset()
    # print(env.render())
    # 一步杀
    move = "6957"
    action = xqcpp.movestr2action(move)
    observation, reward, terminated, truncated, info = env.step(action)
    assert isinstance(observation, dict)
    assert observation["s"].shape == (NUM_ROW * NUM_COL,)
    assert observation["steps"] == MAX_EPISODE_STEPS
    # 首先检查游戏是否正常结束，然后检查连续未吃子及步数超限
    assert observation["continuous_uneaten"] == MAX_NUM_NO_EAT
    assert observation["to_play"] == BLACK_PLAYER
    assert reward == 1
    assert terminated == True
    assert truncated == False
    # print(env.render())


def test_env_v1_ansi_3():
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 119 0 190"
    env = gym.make(
        "xqv1",
        init_fen=init_fen,
        render_mode="ansi",
        # gen_qp=True,
    )
    env.reset()
    # print(env.render())
    # 确保首先检查连续未吃子，然后检查步数超限
    move = "8988"
    action = xqcpp.movestr2action(move)
    observation, reward, terminated, truncated, info = env.step(action)
    assert isinstance(observation, dict)
    assert observation["s"].shape == (NUM_ROW * NUM_COL,)
    assert observation["steps"] == MAX_EPISODE_STEPS
    assert observation["continuous_uneaten"] == MAX_NUM_NO_EAT
    assert observation["to_play"] == BLACK_PLAYER
    assert reward == 0
    assert terminated == True
    assert truncated == False
    # print(env.render())


def test_env_v1_ansi_4():
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 119 0 190"
    env = gym.make(
        "xqv1",
        init_fen=init_fen,
        render_mode="ansi",
        # gen_qp=True,
    )
    env.reset()
    # print(env.render())
    # 如吃子，超限 truncated = True
    move = "6948"
    action = xqcpp.movestr2action(move)
    observation, reward, terminated, truncated, info = env.step(action)
    assert isinstance(observation, dict)
    assert observation["s"].shape == (NUM_ROW * NUM_COL,)
    assert observation["steps"] == MAX_EPISODE_STEPS
    # 吃子后 清0
    assert observation["continuous_uneaten"] == 0
    assert observation["to_play"] == BLACK_PLAYER
    assert reward == 0
    assert terminated == False
    assert truncated == True
    # print(env.render())


@pytest.mark.parametrize(
    "init_fen,moves_list,expected",
    [
        (
            "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190",
            [["2838"], ["2818", "6151"]],
            {
                "total": 2,
                "win": 1,
                "draw": 0,
                "loss": 1,
                # 红方胜率
                "win_rate": 0.5,
                "draw": 0,
                # 黑方胜率
                "loss_rate": 0.5,
            },
        ),
        (
            "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 b - 100 0 190",
            [["6151"], ["6171", "2829"]],
            {
                "total": 2,
                "win": 1,
                "draw": 0,
                "loss": 1,
                # 红方胜率
                "win_rate": 0.5,
                "draw": 0,
                # 黑方胜率
                "loss_rate": 0.5,
            },
        ),
    ],
)
def test_env_satistics_info(init_fen, moves_list, expected):
    env = gym.make(
        "xqv1",
        init_fen=init_fen,
        render_mode="ansi",
        # gen_qp=True,
    )
    info = {}
    for moves in moves_list:
        env.reset()
        for m in moves:
            a = xqcpp.movestr2action(m)
            (
                _,
                _,
                _,
                _,
                info,
            ) = env.step(a)
    for k in expected.keys():
        assert info[k] == expected[k]
