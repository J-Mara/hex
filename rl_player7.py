# rl_player7.py
# checked for 9 artifacts
import os
import numpy as np
from typing import List, Tuple
from rl_model_numpy_7 import PolicyMLP7
from hex_env7 import HexEnv7

Move = Tuple[int, int]
_MODEL7 = None

def _load_model7():
    global _MODEL7
    if _MODEL7 is not None:
        return _MODEL7
    path = os.environ.get("RL7_WEIGHTS", "best7_model.npz")
    if not os.path.exists(path):
        _MODEL7 = PolicyMLP7(seed=0)
        _MODEL7.save(path, step=0)
    else:
        _MODEL7 = PolicyMLP7.load(path)
    return _MODEL7

def choose_move(size: int, moves: List[Move], rng=None) -> Move:
    if size != 7:
        raise ValueError("rl_player7 supports size=7 only.")
    model = _load_model7()
    env = HexEnv7()
    env.moves = list(moves)
    x = env._obs().reshape(-1)  # [243]
    legal = np.ones(49, dtype=bool)
    for (r, c) in set(moves): legal[r*7 + c] = False
    a = model.greedy(x, legal)
    #a = model.sample(x, legal, temperature=5.0)
    return (a // 7, a % 7)
