"""演示
1. 计时
2. 录制视频
3. AI走子提示
"""

# import os

# # This forces OpenMP to use 1 single thread, which is needed to
# # prevent contention between multiple actors.
# # See https://docs.ray.io/en/latest/ray-core/configure.html for
# # more details.
# os.environ["OMP_NUM_THREADS"] = "1"
# # Tell numpy to only use one core. If we don't do this, each actor may
# # try to use all of the cores and the resulting contention may result
# # in no speedup over the serial version. Note that if numpy is using
# # OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
# # probably need to do it from the command line (so it happens before
# # numpy is imported).
# os.environ["MKL_NUM_THREADS"] = "1"

from muzero.config import MuZeroConfig
from muzero.models import MuZeroNetwork
from muzero.self_play import SelfTestPlay, SelfPlay


def demo():
    config = MuZeroConfig()
    init_fen = "3k5/2P1P4/9/9/9/9/9/9/4p1p2/5K3 r - 100 0 190"
    # config.init_fen = ""
    config.num_simulations = 120
    config.debug_duration = True
    # model = MuZeroNetwork(config)
    player = SelfTestPlay(config)
    # player = SelfPlay(config)
    gh = player.rollout(1)


if __name__ == "__main__":
    import timeit

    print(timeit.timeit("demo()", setup="from __main__ import demo", number=1))
