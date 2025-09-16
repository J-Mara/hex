# monte_carlo_player.py
#
# Strategy:
#   1. Enumerate every legal move
#   2. For each move, simulate m possible random games that could follow it and record how often you win.
#   3. Play the move with the highest number of wins.

import random
from typing import List, Tuple

from hex_evaluator import evaluate_hex   # black-box win checker

Move = Tuple[int, int]

def choose_move(size: int, moves: List[Move], rng: random.Random | None = None, sims: int=10) -> Move:
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
    if sims <= 0:
        raise ValueErroe("Number of simulations is not positive.")

    # Determine whose turn it is
    my_color = 'Black' if len(moves) % 2 == 0 else 'White'

    # For every legal move, evaluate m random games which could follow it and keep track of the best move.
    rng.shuffle(available)
    bestMove = available[0]
    mostWin = 0
    for i in range(len(available)):
        wins = 0
        randgame = available[:i] + available[i+1:]
        for j in range(sims):
            rng.shuffle(randgame)
            winner, idx = evaluate_hex(size, moves + [available[i]] + randgame)
            if winner == my_color: wins += 1
        if wins > mostWin:
            mostWin = wins
            bestMove = available[i]

    return bestMove
