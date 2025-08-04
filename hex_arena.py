# hex_arena.py

import argparse
import importlib
import random
from types import ModuleType
from typing import List, Tuple, Optional

from hex_evaluator import evaluate_hex

Move = Tuple[int, int]

def load_player(module_name: str) -> ModuleType:
    """Dynamically import a player module by name."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(f"Cannot import player module '{module_name}': {e}")
    if not hasattr(mod, "choose_move"):
        raise SystemExit(f"Player module '{module_name}' lacks a choose_move() function.")
    return mod

# -----------------------------------------------------------
# One complete game of Hex between two black-box player mods
# -----------------------------------------------------------

def play_single_game(
    size: int,
    black_mod: ModuleType,
    white_mod: ModuleType,
    rng: random.Random
) -> str:                                     # returns 'Black', 'White', or 'Draw'
    moves: List[Move] = []

    while True:
        to_move_mod = black_mod if len(moves) % 2 == 0 else white_mod
        try:
            move = to_move_mod.choose_move(size, moves, rng)
        except Exception as err:                # crash = immediate loss
            loser = 'Black' if len(moves) % 2 == 0 else 'White'
            winner = 'White' if loser == 'Black' else 'Black'
            print(f"⚠️  {loser} program raised {err.__class__.__name__}: {err}")
            return winner

        # basic legality check
        r, c = move
        if (not (0 <= r < size and 0 <= c < size)) or (move in moves):
            loser = 'Black' if len(moves) % 2 == 0 else 'White'
            winner = 'White' if loser == 'Black' else 'Black'
            print(f"⚠️  {loser} played illegal move {move}. {winner} wins.")
            return winner

        moves.append(move)

        winner, _ = evaluate_hex(size, moves)
        if winner is not None:
            return winner
        if len(moves) == size * size:
            return 'Draw'                     # extremely rare in Hex (only possible via error)

# -----------------------------------------------------------
# Match runner
# -----------------------------------------------------------

def run_match(
    size: int,
    games: int,
    player1_mod: ModuleType,
    player2_mod: ModuleType,
    mode: str,
    seed: int
) -> None:
    rng = random.Random(seed)

    # tallies
    p1_total = p2_total = draws = 0
    p1_black = p1_white = p2_black = p2_white = 0

    for g in range(1, games + 1):
        # decide colours
        if mode == "p1_black":
            black_mod, white_mod = player1_mod, player2_mod
            p1_is_black = True
        elif mode == "p2_black":
            black_mod, white_mod = player2_mod, player1_mod
            p1_is_black = False
        elif mode == "alternate":
            if g % 2 == 1:
                black_mod, white_mod = player1_mod, player2_mod
                p1_is_black = True
            else:
                black_mod, white_mod = player2_mod, player1_mod
                p1_is_black = False
        else:
            raise ValueError("mode must be one of: p1_black, p2_black, alternate")

        winner = play_single_game(size, black_mod, white_mod, rng)

        # bookkeeping
        #p1_is_black = (black_mod is player1_mod)
        if winner == 'Black':
            if p1_is_black:
                p1_total += 1; p1_black += 1
            else:
                p2_total += 1; p2_black += 1
        elif winner == 'White':
            if p1_is_black:
                p2_total += 1; p2_white += 1
            else:
                p1_total += 1; p1_white += 1
        else:
            draws += 1

    # ------------------  report  ------------------
    def pct(x: int) -> str:
        return f"{(100.0 * x / games):.1f}%"

    print("\n=== Results ===")
    print(f"Total games      : {games}")
    print(f"Draws            : {draws} ({pct(draws)})")
    print(f"Player1 wins     : {p1_total} ({pct(p1_total)})")
    print(f"Player2 wins     : {p2_total} ({pct(p2_total)})")
    print("----- split by color -----")
    print(f"Player1 as Black : {p1_black} / {games//2 if mode=='alternate' else games} "
          f"({pct(p1_black)})")
    print(f"Player1 as White : {p1_white} / {games//2 if mode=='alternate' else games} "
          f"({pct(p1_white)})")
    print(f"Player2 as Black : {p2_black} / {games//2 if mode=='alternate' else games} "
          f"({pct(p2_black)})")
    print(f"Player2 as White : {p2_white} / {games//2 if mode=='alternate' else games} "
          f"({pct(p2_white)})")

# -----------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Hex arena: pit two black-box players against each other")
    ap.add_argument("--player1", required=True, help="module name for player 1 (importable)")
    ap.add_argument("--player2", required=True, help="module name for player 2 (importable)")
    ap.add_argument("--size", type=int, default=7, help="board size n (n×n)")
    ap.add_argument("--games", type=int, default=100, help="number of games to play")
    ap.add_argument("--mode", choices=["p1_black", "p2_black", "alternate"],
                    default="alternate", help="color assignment scheme")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = ap.parse_args()

    p1 = load_player(args.player1)
    p2 = load_player(args.player2)

    run_match(
        size=args.size,
        games=args.games,
        player1_mod=p1,
        player2_mod=p2,
        mode=args.mode,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()

# run with: python3 hex_arena.py --player1 random_player --player2 one_ahead_random_player --size 7 --games 100 --mode alternate --seed 42
