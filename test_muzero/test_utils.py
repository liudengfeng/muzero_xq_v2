import pytest
import xqcpp
import numpy as np
from muzero.utils import get_lr_policy, get_lr_feature
from gymxq.constants import NUM_ACTIONS


@pytest.mark.parametrize(
    "policy,expected",
    [
        ({"2829": 0.55, "2838": 0.45}, {"6869": 0.55, "6858": 0.45}),
        (
            {"2507": 0.31, "6746": 0.25, "6788": 0.436},
            {"6587": 0.31, "2746": 0.25, "2708": 0.436},
        ),
    ],
)
def test_lr_policy(policy, expected):
    expected_list = [0] * NUM_ACTIONS
    for k, v in expected.items():
        expected_list[xqcpp.m2a(k)] = v

    policy_list = [0] * NUM_ACTIONS
    for k, v in policy.items():
        policy_list[xqcpp.m2a(k)] = v

    # 列表形式
    actual_list = get_lr_policy(policy_list)
    np.testing.assert_array_equal(expected_list, actual_list)

    # 字典形式

    # 移动字符串为键
    actual_policy = get_lr_policy(policy)
    for k in expected.keys():
        assert actual_policy[k] == expected[k]

    # 移动编码为键
    action_policy = {xqcpp.m2a(k): v for k, v in policy.items()}
    action_expected = {xqcpp.m2a(k): v for k, v in expected.items()}
    actual_action_policy = get_lr_policy(action_policy)
    for k in action_expected.keys():
        assert actual_action_policy[k] == action_expected[k]


def test_4d_fliplr():
    # 测试特征左右互换
    obs = np.arange(2700).reshape((2, 15, 10, 9))
    s = obs.shape
    actual = get_lr_feature(obs)
    expected_list = []
    for b in range(s[0]):
        bs = []
        for p in range(s[1]):
            bs.append(np.fliplr(obs[b, p]))
        expected_list.append(bs)
    expected = np.array(expected_list)
    assert actual.shape == s
    np.testing.assert_array_equal(expected, actual)
