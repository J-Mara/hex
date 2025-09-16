# rl_player9.py
import os
import numpy as np
from typing import List, Tuple
from rl_model_numpy_9 import PolicyMLP9
from hex_env9 import HexEnv9

Move = Tuple[int, int]
_MODEL9 = None

def _load_model9():
    global _MODEL9
    if _MODEL9 is not None:
        return _MODEL9
    path = os.environ.get("RL9_WEIGHTS", "best9_model.npz")
    if not os.path.exists(path):
        _MODEL9 = PolicyMLP9(seed=0)
        _MODEL9.save(path, step=0)
    else:
        _MODEL9 = PolicyMLP9.load(path)
    return _MODEL9

def choose_move(size: int, moves: List[Move], rng=None) -> Move:
    if size != 9:
        raise ValueError("rl_player9 supports size=9 only.")
    model = _load_model9()
    env = HexEnv9()
    env.moves = list(moves)
    x = env._obs().reshape(-1)  # [243]
    legal = np.ones(81, dtype=bool)
    for (r, c) in set(moves): legal[r*9 + c] = False
    a = model.greedy(x, legal)
    return (a // 9, a % 9)
