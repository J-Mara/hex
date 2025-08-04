# one_ahead_random_player.py
#
# Strategy:
#   1. Enumerate every legal move
#   2. For each move, simulate playing it and call evaluate_hex().
#      • If it gives the *current* player an immediate win, play it at once.
#   3. If no instant-win move exists, fall back to a uniformly-random choice.

import random
from typing import List, Tuple

from hex_evaluator import evaluate_hex   # black-box win checker

Move = Tuple[int, int]

def choose_move(size: int, moves: List[Move], rng: random.Random | None = None) -> Move:
    """
    Parameters
    ----------
    size   : board dimension n (n×n)
    moves  : moves already played (oldest → newest), 0-based coordinates
    rng    : optional random.Random instance supplied by the arena
    """
    if rng is None:
        rng = random

    occupied = set(moves)
    available: List[Move] = [
        (r, c) for r in range(size) for c in range(size)
        if (r, c) not in occupied
    ]
    if not available:
        raise ValueError("No legal moves left (board is full).")

    # Determine whose turn it is
    my_color = 'Black' if len(moves) % 2 == 0 else 'White'

    # Try every legal move in a random order looking for an immediate win
    #rng.shuffle(available)
    for mv in available:
        hypothetical_moves = moves + [mv]
        winner, idx = evaluate_hex(size, hypothetical_moves)
        if winner == my_color and idx == len(hypothetical_moves):
            return mv          # found a winning move – play it immediately

    # Otherwise, pick any legal move
    return rng.choice(available)
