import gymnasium as gym
import gymxq
import torch
import xqcpp
from .feature_utils import obs2feature, encoded_action
from .models import MuZeroNetwork
from gymxq.constants import NUM_ACTIONS
import numpy as np


def predicate(config):
    model = MuZeroNetwork(config)
    environment = gym.make("xqv1", init_fen=config.init_fen)
    obs, infos = environment.reset()
    observation = obs2feature(obs, flatten=False)
    # last_a = NUM_ACTIONS
    # observation = np.concatenate(
    #     [encoded_action(last_a)[np.newaxis, :], observation], axis=1
    # )
    observation = torch.tensor(observation).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        (
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)
    policy_values = torch.softmax(
        torch.tensor([policy_logits[0][a] for a in infos["legal_actions"]]), dim=0
    ).tolist()
    policy = {
        xqcpp.m2a(a): policy_values[i]
        for i, a in enumerate(infos["legal_actions"])
    }
    return {
        "policy": policy,
        "value": root_predicted_value.detach().cpu().numpy().item(),
    }
