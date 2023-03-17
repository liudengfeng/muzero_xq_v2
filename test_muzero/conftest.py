import gymnasium as gym
import gymxq
import numpy as np
import pytest
import xqcpp
from gymnasium.wrappers.record_video import RecordVideo
from gymxq.constants import BLACK_PLAYER, RED_PLAYER
from muzero.mcts import MinMaxStats, Node, backpropagate, select_child


@pytest.fixture(scope="module")
def environment_factory():
    # Factories as fixtures
    def _make_fn(init_fen, video: bool = True, render_mode=""):
        if render_mode == "human":
            environment = gym.make(
                "xqv1",
                init_fen=init_fen,
                gen_qp=True,
                render_mode="human",
            )
        else:
            if video:
                environment = gym.make(
                    "xqv1",
                    init_fen=init_fen,
                    gen_qp=True,
                    render_mode="rgb_array",
                )
                environment = RecordVideo(
                    environment,
                    "test_videos",
                    episode_trigger=lambda x: True,
                    disable_logger=True,
                )
            else:
                # 便于调试
                environment = gym.make(
                    "xqv1",
                    init_fen=init_fen,
                    render_mode="ansi",
                )
        return environment

    return _make_fn


@pytest.fixture(scope="module")
def selfplay_factory(environment_factory):
    # Factories as fixtures
    def _simulate_fn(config, node, action):
        # 取消 记录棋谱、录制视频 加快速度
        environment = environment_factory(config.init_fen, False)
        obs, info = environment.reset()
        to_play = info["to_play"]
        moved = node.path().split(" -> ")[1:]
        for m in moved:
            a = xqcpp.movestr2action(m)
            o, r, t, tr, info = environment.step(a)
            to_play = info["to_play"]
            if t or tr:
                # print(
                #     "selfplay_factory 初始fen = {} moved = {}".format(
                #         config.init_fen, moved
                #     )
                # )
                # print(environment.render())
                return 0, r, {}
        _, r, t, tr, info = environment.step(action)
        to_play = info["to_play"]
        if t or tr:
            # print(
            #     "selfplay_factory 初始fen = {} moved = {} action = {}".format(
            #         config.init_fen, moved, xqcpp.action2movestr(action)
            #     )
            # )
            # print(environment.render())
            return 0, r, {}
        legal_actions = info["legal_actions"]
        policy = {}
        p = 1 / len(legal_actions)
        policy = {a: p for a in legal_actions}
        return 0, r, policy

    return _simulate_fn


@pytest.fixture(scope="module")
def mcts_factory(selfplay_factory):
    # 模拟mcts搜索
    def mcts_search_fn(config, init_legal_actions):
        np.random.seed(config.seed)
        to_play = RED_PLAYER if config.init_fen.split(" ")[1] == "r" else BLACK_PLAYER
        root = Node(0)
        p = 1 / len(init_legal_actions)
        init_policy = {a: p for a in init_legal_actions}
        root.expand(init_legal_actions, to_play, 0, init_policy, None, True)
        # 政策加噪
        root.add_exploration_noise(
            dirichlet_alpha=config.root_dirichlet_alpha,
            exploration_fraction=config.root_exploration_fraction,
        )
        min_max_stats = MinMaxStats()
        for n in range(config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            while node.expanded():
                action, node = select_child(node, min_max_stats, config)
                search_path.append(node)
                # Players play turn by turn
                virtual_to_play = (
                    RED_PLAYER if virtual_to_play == BLACK_PLAYER else BLACK_PLAYER
                )
            parent = search_path[-2]
            value, reward, policy = selfplay_factory(config, parent, action)
            node.expand(
                config.action_space, virtual_to_play, reward, policy, None, True
            )
            backpropagate(search_path, value, BLACK_PLAYER, min_max_stats, config)
        return root

    return mcts_search_fn
