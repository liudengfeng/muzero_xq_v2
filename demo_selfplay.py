"""估计自玩一局用时"""

import logging
import itertools

import ray
import time
from muzero.config import MuZeroConfig
from muzero.self_play import SelfTestPlayer, SelfPlayer
import os
import coloredlogs
import random
import torch


@ray.remote
def run_self(idx, steps, num_simulations):
    config = MuZeroConfig()
    # 标准开局，通过步数控制长度
    init_fen = (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 {}".format(
            config.max_moves - steps + 1
        )
    )
    config.init_fen = init_fen
    config.num_simulations = num_simulations
    # cpu速度快 可能批量问题导致GPU无效???
    # config.selfplay_on_gpu = True
    # torch.backends.cudnn.benchmark = True
    config.debug_duration = True
    start = time.time()
    player = SelfPlayer(config, idx)
    # 录制视频、生成棋谱耗时
    # player = SelfTestPlayer(config, idx)
    player._play_game(idx)
    return steps, num_simulations, round(time.time() - start, 2)


def main_ray(logger):
    result = []
    MAX_NUM_PENDING_TASKS = os.cpu_count() - 1
    # 局部
    step_remain = list(range(1, 11))
    num_simulations = list(range(30, 270, 30))
    args = list(itertools.product(step_remain, num_simulations))
    # 均匀任务
    random.shuffle(args)
    result_refs = []
    for idx, (step, sim) in enumerate(args, 1):
        if len(result_refs) >= MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs)
            result.extend(ray.get(ready_refs))
            print(result)
        logger.info("第{:3d}实验 步数 {:3d} MCTS模拟次数 {:3d}".format(idx, step, sim))
        result_refs.append(run_self.remote(idx, step, sim))
    result.extend(ray.get(result_refs))
    print(result)


if __name__ == "__main__":
    coloredlogs.install(fmt="%(asctime)s[%(levelname)s] > %(message)s", level="DEBUG")
    logger = logging.getLogger("muzero")
    try:
        # ray.init(num_gpus=1)
        main_ray(logger)
    except KeyboardInterrupt:
        logger.warning("提前终止\n")
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        ray.shutdown()
