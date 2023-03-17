import cProfile, pstats, io
from pstats import SortKey
import torch
import numpy as np
from muzero.config import MuZeroConfig
from muzero.models import MuZeroNetwork
from muzero.mcts import MCTS

# torch.backends.cudnn.benchmark = True

config = MuZeroConfig()
# Fix random generator seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)

config.num_simulations = 60
model = MuZeroNetwork(config)
# model.to("cuda")
model.eval()


def test():
    with torch.no_grad():
        observations = np.random.random((1, 17, 10, 9)).astype(np.float32)
        legal_actions = np.random.randint(2086, size=60).tolist()
        root, extra_info = MCTS(config).run(
            model, observations, legal_actions, 1, True
        )
        print(extra_info)


pr = cProfile.Profile()
pr.enable()

# 监测函数
test()

pr.disable()
s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# 重点关注耗时长的部分
sortby = SortKey.TIME
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
