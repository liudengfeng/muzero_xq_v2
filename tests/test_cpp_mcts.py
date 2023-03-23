import pytest
import mctslib
from muzero.config import MuZeroConfig
from muzero.apply import get_pretrained_model
import torch


def test_config():
    config = mctslib.MuZeroConfig()
    assert config.observation_shape == (15, 10, 9)


def test_scipt():
    config = MuZeroConfig()
    config.runs = 9
    model = get_pretrained_model(config)
    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 15, 10, 9)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module

    pth = "traced_resnet_model.pt"
    traced_script_module.save(pth)
    print(mctslib.infer(pth, example))
