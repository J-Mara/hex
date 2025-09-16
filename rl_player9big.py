# rl_player9big.py
import os
import numpy as np
from typing import List, Tuple
from rl_model_numpy_9big import PolicyMLP9Big
from hex_env9 import HexEnv9

Move = Tuple[int, int]
_MODEL = None

def _load():
    global _MODEL
    if _MODEL is not None: return _MODEL
    path = os.environ.get("RL9BIG_WEIGHTS", "best9big_model.npz")
    if not os.path.exists(path):
        _MODEL = PolicyMLP9Big(seed=0); _MODEL.save(path, step=0)
    else:
        _MODEL = PolicyMLP9Big.load(path)
    return _MODEL

def choose_move(size: int, moves: List[Move], rng=None) -> Move:
    if size != 9:
        raise ValueError("rl_player9big supports size=9 only.")
    model = _load()
    env = HexEnv9(); env.moves = list(moves)
    x = env._obs().reshape(-1)  # [243]
    legal = np.ones(81, dtype=bool)
    for (r,c) in set(moves): legal[r*9+c] = False
    a = model.greedy(x, legal)
    return (a // 9, a % 9)
