# import codon
from .mcts_codon import CodonMCTS


# @codon.jit(pyvars=["CodonMCTS"])
def codon_mcts_search(
    config, model, observation, legal_actions, to_play, add_exploration_noise
):
    root, mcts_info = CodonMCTS(config).run(
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
    )
    return root, mcts_info
