import math
import os
import time
from typing import List

import graphviz
import numpy as np
import torch
import xqcpp
from gymxq.constants import BLACK_PLAYER, NUM_ACTIONS, RED_PLAYER
import pytest


ACTION_SPACE = list(range(1, 7))
# 每一步大约损失1%，便于观察
DISCOUT_FACTOR = 0.99
PB_C_BASE = 19652
PB_C_INT = 1.25
ROOT_DIRICHLET_ALPHA = 0.25
ROOT_EXPLOTATION_FRACTION = 0.25


def fake_initial_inference(observation):
    """虚拟初始推理"""
    # discount factor = 0.99
    match observation:
        # value, reward, policy, hidden_state
        case 0:
            return DISCOUT_FACTOR**2, 0, {1: 1.0, 2: 0.0}, 0
    raise ValueError(f"observation = {observation}")


def fake_recurrent_inference(hidden_state, a):
    # value, reward, policy, hidden_state
    match hidden_state:
        # value, reward, policy, hidden_state
        case 0:
            match a:
                case 1:
                    return -DISCOUT_FACTOR, 0, {3: 1.0}, 1
                case 2:
                    return 1, 0, {4: 0.5, 5: 0.5}, 2
        case 1:
            match a:
                case 3:
                    return 1, 0, {6: 1.0}, 3
        case 2:
            match a:
                case 4:
                    return 0, 1, {}, 4
                case 5:
                    return 0, 1, {}, 5
        case 3:
            match a:
                case 6:
                    return 0, 1, {}, 6
    raise ValueError(f"observation = {hidden_state}")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        # 子节点关键字为整数
        self.children = {}
        self.hidden_state = None
        self.reward = 0

        self.policy = {}
        # for graphviz
        self.parent = None
        # self.depth_ = 0
        self.path_ = "root"
        # 注意，此处为字符串，代表移动字符串
        self.action = "root"
        self.ucb_score = 0

    def __hash__(self) -> int:
        return hash(self.path_)

    def __eq__(self, other):
        return self.path() == other.path()

    def path(self):
        return self.path_

    def depth(self):
        return len(self.path_.split(" -> ")) - 1

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(
        self,
        actions,
        to_play,
        reward,
        policy_logits,
        hidden_state,
        use_policy=False,
        debug=False,
    ):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        if not use_policy:
            policy_values = torch.softmax(
                torch.tensor([policy_logits[0][a] for a in actions]), dim=0
            ).tolist()
            policy = {a: policy_values[i] for i, a in enumerate(actions)}
        else:
            # 以下测试专用
            assert isinstance(policy_logits, dict), "policy_logits应为字典对象"
            assert all(
                [isinstance(k, int) for k in policy_logits.keys()]
            ), "测试时所提供的政策，其关键字应为代表移动的整数编码"
            policy = policy_logits

        self.policy = policy

        if debug:
            # 调试模式需要与父节点链接
            for action, p in policy.items():
                # 添加概率限制
                if p > 0.0:
                    c = Node(p)
                    c.parent = self
                    c.to_play = (
                        BLACK_PLAYER if self.to_play == RED_PLAYER else RED_PLAYER
                    )
                    # c.action = xqcpp.a2m(action)
                    c.action = "{:04d}".format(action)
                    c.path_ = "{} -> {}".format(c.parent.path(), c.action)
                    self.children[action] = c
        else:
            for action, p in policy.items():
                # 添加概率限制
                if p > 0.0:
                    c = Node(p)
                    c.to_play = (
                        BLACK_PLAYER if self.to_play == RED_PLAYER else RED_PLAYER
                    )
                    # c.action = xqcpp.a2m(action)
                    c.action = "{:04d}".format(action)
                    c.path_ = "{} -> {}".format(self.path(), c.action)
                    self.children[action] = c

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

    def get_updated_policy(self):
        # 更新后的政策(用于显示，精确到2位)
        sum_visits = sum(child.visit_count for child in self.children.values())
        if sum_visits > 0:
            return {
                xqcpp.a2m(a): round(child.visit_count / sum_visits, 2)
                for a, child in self.children.items()
            }
        else:
            return {}

    def get_root_value(self):
        if self.parent is None:
            return max([c.value() for c in self.children.values()])
        return self.value()


def get_root_node_table_like_label(state: Node):
    """根节点label
    Args:
        state (Node): 根节点
    Returns:
        str: 节点标签
    """
    return """<
<TABLE>
  <TR>
    <TD ALIGN='LEFT'>State</TD>
    <TD ALIGN='RIGHT'>Root</TD>
  </TR>
    <TR>
    <TD ALIGN='LEFT'>Vsum</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Visit</TD>
    <TD ALIGN='RIGHT'>{:4d}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Value</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
</TABLE>>""".format(
        state.value_sum,
        state.visit_count,
        state.value(),
    )


def get_node_table_like_label(state: Node):
    return """<
<TABLE>
  <TR>
    <TD ALIGN='LEFT'>Reward</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Vsum</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Value</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>UCB</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Prior</TD>
    <TD ALIGN='RIGHT'>{:.2f}</TD>
  </TR>
  <TR>
    <TD ALIGN='LEFT'>Visit</TD>
    <TD ALIGN='RIGHT'>{:4d}</TD>
  </TR>
</TABLE>>""".format(
        state.reward,
        state.value_sum,
        state.value(),
        state.ucb_score,
        state.prior,
        state.visit_count,
    )


def default_node_decorator(state: Node):
    """Decorates a state-node of the game tree.
    This method can be called by a custom decorator to prepopulate the attributes
    dictionary. Then only relevant attributes need to be changed, or added.
    Args:
      state: The state.
    Returns:
      `dict` with graphviz node style attributes.
    """
    _PLAYER_COLORS = {-1: "black", 1: "red", 2: "blue"}
    player = state.parent.to_play if state.parent else -1
    attrs = {}
    attrs["label"] = get_node_table_like_label(state)
    attrs["color"] = _PLAYER_COLORS.get(player, "black")
    return attrs


def default_edge_decorator(child):
    """Decorates a state-node of the game tree.
    This method can be called by a custom decorator to prepopulate the attributes
    dictionary. Then only relevant attributes need to be changed, or added.
    Args:
      parent: The parent state.
    Returns:
      `dict` with graphviz node style attributes.
    """
    _PLAYER_COLORS = {-1: "black", 1: "red", 2: "blue"}
    player = child.parent.to_play if child.parent else -1
    attrs = {}
    attrs["label"] = child.action
    attrs["color"] = _PLAYER_COLORS.get(player, "black")
    return attrs


def build_mcts_tree(dot, state, depth):
    """Recursively builds the mcts tree."""
    if not state.expanded():
        return

    for child in state.children.values():
        if child.visit_count >= 1:
            dot.node(child.path(), **default_node_decorator(child))
            dot.edge(
                child.parent.path(),
                child.path(),
                **default_edge_decorator(child),
            )
            build_mcts_tree(dot, child, depth + 1)


def render_root(
    root: Node,
    filename: str,
    format: str = "png",
    saved_path=None,
):
    """演示根节点
    Args:
        root (Node): 根节点
        filename (str):  文件名称
        format (str, optional): 输出文件格式. Defaults to "png".
        saved_path (str, optional):  存储路径
    """
    assert format in ["png", "svg"], "仅支持png和svg格式"
    graph_attr = {"rankdir": "LR", "fontsize": "8"}
    node_attr = {"shape": "plaintext"}
    # 不需要扩展名
    name = filename.split(".")[0]
    dot = graphviz.Digraph(
        name,
        comment="蒙特卡洛搜索树",
        format=format,
        graph_attr=graph_attr,
        node_attr=node_attr,
    )
    dot.node("root", label=get_root_node_table_like_label(root), shape="oval")
    build_mcts_tree(dot, root, 0)
    # 尚未展开，to_play = -1
    # 多进程操作
    if saved_path:
        fp = os.path.join(saved_path, "mcts_{}".format(name))
    else:
        fp = "pid_{:06d}".format(name, os.getpid())
    dot.render(fp, view=False, cleanup=True)


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def ucb_score(parent, child, min_max_stats):
    """
    The score for a node is based on its value, plus an exploration bonus based on the prior.
    """
    pb_c = math.log((parent.visit_count + PB_C_BASE + 1) / PB_C_BASE) + PB_C_INT
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        # Mean value Q
        value_score = min_max_stats.normalize(
            child.reward + DISCOUT_FACTOR * (-child.value())
        )
    else:
        value_score = 0

    return prior_score + value_score


def select_child(node, min_max_stats):
    # 简化计算量
    kvs = {
        action: ucb_score(node, child, min_max_stats)
        for action, child in node.children.items()
    }
    # 更新ucb得分
    for action, _ in kvs.items():
        node.children[action].ucb_score = kvs[action]

    max_ucb = sorted(kvs.values())[-1]
    # 可能有多个相同的最大值
    action = np.random.choice(
        [action for action, value in kvs.items() if value == max_ucb]
    )
    return action, node.children[action]


def backpropagate(search_path, value, to_play, min_max_stats):
    """
    At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root.
    """
    # 调整为玩家角度奖励
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.reward + DISCOUT_FACTOR * -node.value())

        value = (
            -node.reward if node.to_play == to_play else node.reward
        ) + DISCOUT_FACTOR * value


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def run(
        self,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        num_simulations,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        root = Node(0)
        assert to_play in (RED_PLAYER, BLACK_PLAYER), "玩家编码必须为{}".format(
            (RED_PLAYER, BLACK_PLAYER)
        )

        assert isinstance(observation, int), "Observation shape be {}".format("int")
        (root_predicted_value, reward, policy, hidden_state) = fake_initial_inference(
            observation
        )

        assert reward == 0, "初始推理reward应等于0"

        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(ACTION_SPACE)
        ), "Legal actions should be a subset of the action space."
        root.expand(
            legal_actions,
            to_play,
            reward,
            policy,
            hidden_state,
            True,
            True,
        )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=ROOT_DIRICHLET_ALPHA,
                exploration_fraction=ROOT_EXPLOTATION_FRACTION,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0

        for n in range(num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            # 限制深度
            while node.expanded() and current_tree_depth < 200:
                current_tree_depth += 1
                action, node = select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                virtual_to_play = (
                    RED_PLAYER if virtual_to_play == BLACK_PLAYER else BLACK_PLAYER
                )

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]

            value, reward, policy, hidden_state = fake_recurrent_inference(
                parent.hidden_state,
                action,
            )

            node.expand(
                ACTION_SPACE,
                virtual_to_play,
                reward,
                policy,
                hidden_state,
                True,
                True,
            )

            backpropagate(
                search_path,
                value,
                virtual_to_play,
                min_max_stats,
            )

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info


def test_mcts_value():
    # 验证符号、反向传播
    root, info = MCTS().run(0, [1, 2], 1, True, 30)
    # render_root(root, "research", "svg", "mcts_tree")
    np.testing.assert_approx_equal(root.value(), 0.98, 2)
