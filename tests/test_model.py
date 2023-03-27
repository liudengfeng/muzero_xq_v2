import numpy as np
import pytest
import torch

from muzero.config import MuZeroConfig, PLANE_NUM, NUM_ROW, NUM_COL
from muzero.feature_utils import encoded_action
from muzero.models import MuZeroNetwork
from muzero.trainer_utils import loss_function


@pytest.mark.parametrize(
    "expected",
    [
        ([-0.011, 0.063, -0.03, 0.052, 0.133, -0.156, 0.1, 0.002]),
    ],
)
def test_inference(expected):
    workers = 8
    config = MuZeroConfig()

    # Fix random generator seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    encoded_action_shape = (workers, 2, NUM_ROW, NUM_COL)
    action_size = len(config.action_space)
    model = MuZeroNetwork(config)
    model = torch.compile(model)
    observations = torch.rand(
        (workers, PLANE_NUM, NUM_ROW, NUM_COL), dtype=torch.float32
    )
    (
        root_predicted_value,
        reward,
        policy_logits,
        hidden_state,
    ) = model.initial_inference(observations)
    assert root_predicted_value.shape == (workers,)

    np.testing.assert_almost_equal(
        root_predicted_value.detach().cpu().numpy(), np.array(expected), 3
    )

    assert policy_logits.shape == (workers, action_size)
    assert hidden_state.shape == (workers, 256, NUM_ROW, NUM_COL)
    action = np.concatenate([[encoded_action(a)] for a in range(workers)])

    value, reward, policy_logits, hidden_state = model.recurrent_inference(
        hidden_state, torch.tensor(action).to(hidden_state.device)
    )
    assert value.shape == (workers,)
    assert reward.shape == (workers,)
    assert policy_logits.shape == (workers, action_size)
    assert hidden_state.shape == (workers, 256, NUM_ROW, NUM_COL)


def test_loss_function():
    workers = 16
    config = MuZeroConfig()
    encoded_action_shape = (workers, 2, NUM_ROW, NUM_COL)
    action_size = len(config.action_space)
    model = MuZeroNetwork(config)
    model = torch.compile(model)
    ob1 = torch.rand((workers, PLANE_NUM, NUM_ROW, NUM_COL), dtype=torch.float32)
    ob2 = torch.rand((workers, PLANE_NUM, NUM_ROW, NUM_COL), dtype=torch.float32)
    (
        value,
        reward,
        policy_logits,
        hidden_state,
    ) = model.initial_inference(ob1)
    (
        target_value,
        target_reward,
        target_policy,
        hidden_state,
    ) = model.initial_inference(ob2)
    value_loss, reward_loss, policy_loss = loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    )
    assert value_loss.shape == (workers,)
    assert reward_loss == 0
    assert policy_loss.shape == (workers,)
