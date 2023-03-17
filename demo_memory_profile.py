import numpy as np
import torch
from memory_profiler import profile

from muzero.config import PLANE_NUM, MuZeroConfig
from muzero.feature_utils import encoded_action
from muzero.models import MuZeroNetwork
from muzero.self_play import SelfPlay, SelfTestPlay
from muzero.checkpoint import get_current_checkpoint
from muzero.mcts import MCTS


# @profile
# def run_model():
#     config = MuZeroConfig()
#     # Fix random generator seed
#     np.random.seed(config.seed)
#     torch.manual_seed(config.seed)

#     model = MuZeroNetwork(config)
#     weights = get_current_checkpoint(config)["weights"]
#     model.set_weights(weights)
#     # 100 M 即时删除会减少内存峰值
#     del weights

#     observations = torch.rand(
#         (config.batch_size, PLANE_NUM, 10, 9), dtype=torch.float32
#     )
#     (
#         root_predicted_value,
#         reward,
#         policy_logits,
#         hidden_state,
#     ) = model.initial_inference(observations)

#     del observations, policy_logits

#     high = len(config.action_space)
#     action = np.concatenate(
#         [
#             [encoded_action(np.random.randint(0, high - 1))]
#             for _ in range(config.batch_size)
#         ]
#     )
#     value, reward, policy_logits, hidden_state = model.recurrent_inference(
#         hidden_state, torch.tensor(action).to(hidden_state.device)
#     )


# @profile
# def run_mcts():
#     config = MuZeroConfig()
#     config.num_simulations = 120
#     # Fix random generator seed
#     np.random.seed(config.seed)
#     torch.manual_seed(config.seed)

#     model = MuZeroNetwork(config)
#     observations = np.random.random((1, PLANE_NUM, 10, 9)).astype(np.float32)
#     high = len(config.action_space)
#     # legal_actions = [np.random.randint(0, high - 1) for _ in range(60)]
#     legal_actions = config.action_space
#     with torch.no_grad():
#         root, extra_info = MCTS(config).run(model, observations, legal_actions, 1, True)
#         print(extra_info)


# @profile
def rollout():
    config = MuZeroConfig()
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190"
    # config.init_fen = ""
    config.num_simulations = 60
    config.debug_duration = True
    config.debug_mcts = True
    # model = MuZeroNetwork(config)
    # player = SelfTestPlay(config,init_fen=init_fen)
    player = SelfPlay(config, init_fen=init_fen)
    gh = player.rollout(1)
    print()


if __name__ == "__main__":
    # run_model()
    # run_mcts()
    rollout()
