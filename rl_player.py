# rl_player.py
import os
import numpy as np
from typing import List, Tuple
from rl_model_numpy import PolicyMLP
from hex_env import HexEnv5

Move = Tuple[int, int]
_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = os.environ.get("RL_WEIGHTS", "best_model.npz")
    #path = os.environ.get("RL_WEIGHTS", "best_model5_first_good.npz")
    #path = os.environ.get("RL_WEIGHTS", "potential_optimal5_black.npz")
    if not os.path.exists(path):
        # If not trained yet, initialize random model and save so arena calls work.
        _MODEL = PolicyMLP(seed=0)
        _MODEL.save(path, step=0)
    else:
        _MODEL = PolicyMLP.load(path)
    return _MODEL

def choose_move(size: int, moves: List[Move], rng=None) -> Move:
    if size != 5:
        raise ValueError("This RL player currently supports size=5 only.")
    model = _load_model()
    env = HexEnv5()
    env.moves = list(moves)
    state = env._obs()                                    # [3,5,5]
    x = state.reshape(-1)                                 # [75]
    legal = np.ones(25, dtype=bool)
    for (r, c) in set(moves):
        legal[r*5 + c] = False
    a = model.greedy(x, legal)
    #a = model.sample(x, legal, temperature = 3.0)
    return (a // 5, a % 5)
