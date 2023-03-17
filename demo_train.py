import os

# # This forces OpenMP to use 1 single thread, which is needed to
# # prevent contention between multiple actors.
# # See https://docs.ray.io/en/latest/ray-core/configure.html for
# # more details.
os.environ["OMP_NUM_THREADS"] = "1"
# # Tell numpy to only use one core. If we don't do this, each actor may
# # try to use all of the cores and the resulting contention may result
# # in no speedup over the serial version. Note that if numpy is using
# # OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
# # probably need to do it from the command line (so it happens before
# # numpy is imported).
os.environ["MKL_NUM_THREADS"] = "1"


import time
import ray
import numpy as np
import torch
import xqcpp

# from functools import partial
from muzero.replay_buffer_utils import Buffer, make_target
from muzero.config import PLANE_NUM, MuZeroConfig
from muzero.models import MuZeroNetwork
from muzero.self_play import SelfPlay
from muzero.trainer_utils import update_lr, update_weights

N = 8
num_gpus = float(1 / N)


@ray.remote(num_gpus=num_gpus)
class Player(SelfPlay):
    def __init__(self, config, worker_id: int = 0, init_fen: str = ""):
        super().__init__(config, worker_id, init_fen)
        self.model = MuZeroNetwork(self.config)
        self.model.eval()
        self.model.to("cuda")

    def selfplay(self, weights, training_step):
        self.model.set_weights(weights)
        gh = self.rollout(self.model, training_step)
        return gh


def train():
    batch_time = 0
    # init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190"
    init_fen = "2r2k3/6R1C/b4N1rb/9/5n3/5C3/6n2/5p3/4p4/5K1R1 r - 110 0 180"
    # init_fen = "2r2k3/6R1C/b4N1rb/9/5n3/5C3/6n2/5p3/4p4/5K1R1 r - 110 0 180"
    config = MuZeroConfig()
    config.batch_size = 128
    config.training_steps = 200
    config.num_simulations = 60

    buffer = Buffer(config)

    model = MuZeroNetwork(config)
    model.to("cuda")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr_init,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # buffer_ref = ray.put(buffer)
    players = [Player.remote(config, i, init_fen) for i in range(N)]
    start = time.time()
    for training_step in range(config.training_steps):
        
        weights_ref = ray.put(model.get_weights())
        task_refs = [
            player.selfplay.remote(weights_ref, training_step) for player in players
        ]
        ghs = ray.get(task_refs)
        for gh in ghs:
            buffer.save_game(gh)
        # 训练
        index_batch, batch = buffer.get_batch()
        model.train()
        (
            total_loss,
            value_loss,
            reward_loss,
            policy_loss,
        ) = update_weights(batch, model, optimizer, config, False)
        update_lr(training_step, optimizer, config)
        if training_step % 10 == 0:
            batch_time += time.time() - start
            print(
                "{:>7d}nth loss [total:{:>6.2f} value:{:>6.2f} reward:{:>6.2f} policy:{:>6.2f}] lr:{:.5f} games:{:>7d} duration:{:>7.2f}".format(
                    training_step,
                    total_loss,
                    value_loss,
                    reward_loss,
                    policy_loss,
                    optimizer.param_groups[0]["lr"],
                    buffer.num_played_games,
                    batch_time,
                )
            )
            batch_time = 0

    # 存储模型
    torch.save(model.get_weights(), "model_weights.pth")


if __name__ == "__main__":
    ray.init(num_gpus=1)
    train()
    ray.shutdown()
