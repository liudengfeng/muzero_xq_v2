import pytest
import mctslib


def test_config():
    config = mctslib.MuZeroConfig()
    assert config.observation_shape == (15, 10, 9)
