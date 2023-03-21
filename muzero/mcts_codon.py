import math
import os
import time
from typing import List

# import codon
import graphviz
import numpy as np
import torch
import xqcpp
from gymxq.constants import BLACK_PLAYER, NUM_ACTIONS, RED_PLAYER

from .feature_utils import encoded_action


# @codon.convert
class Node:

    # __slots__ = (
    #     "visit_count",
    #     "to_play",
    #     "prior",
    #     "value_sum",
    #     "children",
    #     "hidden_state",
    #     "reward",
    #     "policy",
    #     "parent",
    #     "path_",
    #     "action",
    #     "ucb_score",
    # )

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

    # @codon.jit
    def path(self):
        return self.path_

    # @codon.jit
    def depth(self):
        return len(self.path_.split(" -> ")) - 1

    # @codon.jit
    def expanded(self):
        return len(self.children) > 0

    # @codon.jit
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    # @codon.jit(pyvars=["Node", "torch", "RED_PLAYER", "BLACK_PLAYER", "xqcpp"])
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
                    c.action = xqcpp.a2m(action)
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
                    c.action = xqcpp.a2m(action)
                    c.path_ = "{} -> {}".format(self.path(), c.action)
                    self.children[action] = c

    # @codon.jit(pyvars=["np"])
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

    # @codon.jit(pyvars=["xqcpp"])
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

    # @codon.jit
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


def default_edge_decorator(child: Node):
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


# @codon.convert
class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    # __slots__ = "maximum", "minimum"

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    # @codon.jit
    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    # @codon.jit
    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


# @codon.jit(pyvars=["math"])
def ucb_score_fn(parent, child, min_max_stats, config):
    """
    The score for a node is based on its value, plus an exploration bonus based on the prior.
    """
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        # Mean value Q
        value_score = min_max_stats.normalize(
            child.reward
            + config.discount_factor
            * (child.value() if len(config.players) == 1 else -child.value())
        )
    else:
        value_score = 0

    return prior_score + value_score


# @codon.jit(pyvars=["np", "ucb_score_fn", "Node"])
def select_child(node, min_max_stats, config):
    # 简化计算量
    kvs = {
        action: ucb_score_fn(node, child, min_max_stats, config)
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


# @codon.jit
def backpropagate(search_path, value, to_play, min_max_stats, config):
    """
    At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root.
    """
    if len(config.players) == 1:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + config.discount_factor * node.value())
            value = node.reward + config.discount_factor * value

    elif len(config.players) == 2:
        # 调整为玩家角度奖励
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.reward + config.discount_factor * -node.value())

            value = (
                -node.reward if node.to_play == to_play else node.reward
            ) + config.discount_factor * value

    else:
        raise NotImplementedError("More than two player mode not implemented.")


# @codon.convert
class CodonMCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    # __slots__ = "config"

    def __init__(self, config):
        self.config = config

    # @codon.jit(pyvars=["Node", "torch", "RED_PLAYER", "BLACK_PLAYER", "np", "encoded_action"])
    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
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

        assert (
            observation.shape == self.config.encoded_observation_shape
        ), "Observation shape should be {}".format(
            self.config.encoded_observation_shape
        )
        observation = torch.tensor(observation).to(next(model.parameters()).device)
        (
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)

        del observation

        assert reward == 0, "初始推理reward应等于0"

        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        root.expand(
            legal_actions,
            to_play,
            reward,
            policy_logits,
            hidden_state,
            debug=True if self.config.debug_mcts else False,
        )

        del policy_logits

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0

        for n in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            # 限制深度
            while node.expanded() and current_tree_depth < self.config.max_moves:
                current_tree_depth += 1
                action, node = select_child(node, min_max_stats, self.config)
                search_path.append(node)

                # Players play turn by turn
                virtual_to_play = (
                    RED_PLAYER if virtual_to_play == BLACK_PLAYER else BLACK_PLAYER
                )

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]

            encoded_a = np.array([encoded_action(action)], dtype=np.float32)

            encoded_a = torch.tensor(encoded_a).to(parent.hidden_state.device)

            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                encoded_a,
            )

            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward.detach().cpu().numpy().item(),
                policy_logits,
                hidden_state,
                debug=True if self.config.debug_mcts else False,
            )

            backpropagate(
                search_path,
                value.detach().cpu().numpy().item(),
                virtual_to_play,
                min_max_stats,
                self.config,
            )

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.terminated_history = []
        self.truncated_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

        # 🚨对齐
        self.reward_history.append(0)
        self.action_history.append(NUM_ACTIONS)
        self.terminated_history.append(False)
        self.truncated_history.append(False)

    def get_stacked_observations(self, index):
        """
        Generate a new observation with the observation at the index position
        """
        n = len(self.observation_history)
        valid_idx = [-1] + list(range(n))
        assert index in valid_idx, "有效索引为{},输入{}".format(valid_idx, index)
        if index == -1:
            start = n - 1
        else:
            start = index
        return self.observation_history[start].copy()

    def store_search_statistics(self, root: Node, action_space: List[int]):
        """为游戏历史对象存储统计信息
        Args:
            root (Node): 根节点
            action_space (List[int]): 整数移动空间列表
        """
        # 将访问次数转换为政策
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def get_reward_pair(self, muzero_player):
        # 象棋只有最终结果可能非0
        idx = len(self.reward_history) - 1
        if self.to_play_history[idx - 1] == muzero_player:
            return {
                "muzero_reward": self.reward_history[idx],
                "opponent_reward": 0,
            }
        else:
            return {
                "muzero_reward": 0,
                "opponent_reward": self.reward_history[idx],
            }
