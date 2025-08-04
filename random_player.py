# random_player.py
# A minimal Hex “AI”: choose_move(size, moves, rng) → (row, col)

import random
from typing import List, Tuple

Move = Tuple[int, int]

def choose_move(size: int, moves: List[Move], rng: random.Random | None = None) -> Move:
    """Return a random unoccupied cell.
    Parameters
    ----------
    size   : board dimension n (the board is n×n)
    moves  : list of moves already played, each (row, col) with 0-based indices
    rng    : optional random.Random instance for reproducibility
    """
    if rng is None:
        rng = random
    occupied = set(moves)
    available = [(r, c) for r in range(size) for c in range(size)
                 if (r, c) not in occupied]

    if not available:
        raise ValueError("Board is full – no legal moves remain.")
    return rng.choice(available)
