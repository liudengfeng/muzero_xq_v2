from muzero.config import MuZeroConfig
from muzero.apply import initial_inference, get_muzero_action
import xqcpp
import time

config = MuZeroConfig()
config.runs = 5
config.init_fen = "2b6/3ka4/2Pa5/3N5/9/3R5/9/9/5pr2/3AK4 r - 110 0 190"
config.restore_from_latest_checkpoint = True
print(initial_inference(config))

start = time.time()
config.num_simulations = 30
# moves = ["2737", "3839", "3738", "3949"]
# action = get_muzero_action(config, moves)
action = get_muzero_action(config, [])
print(
    "Mcts tree search move = {} duration = {:.2f}s".format(
        xqcpp.a2m(action), time.time() - start
    )
)
