# # hex_multi_arena.py
#
# import argparse
# import csv
# import importlib
# import os
# import random
# import time
# import inspect
# from dataclasses import asdict, is_dataclass
# from types import ModuleType
# from typing import List, Tuple, Dict, Any, Optional
# from hexfast import evaluate_hex
#
# Move = Tuple[int, int]
# MCTS_STAT_KEYS = [
#     "board_size",
#     "starting_moves",
#     "root_player",
#     "iterations_requested",
#     "iterations_performed",
#     "time_limit",
#     "duration",
#     "nodes_created",
#     "nodes_expanded",
#     "max_depth_reached",
# ]
#
# def load_player(module_name: str) -> ModuleType:
#     """Dynamically import a player module by name."""
#     try:
#         mod = importlib.import_module(module_name)
#     except ImportError as e:
#         raise SystemExit(f"Cannot import player module '{module_name}': {e}")
#     if not hasattr(mod, "choose_move"):
#         raise SystemExit(f"Player module '{module_name}' lacks a choose_move() function.")
#     return mod
#
# def _format_elapsed(seconds: float) -> str:
#     s = int(seconds)
#     h = s // 3600
#     m = (s % 3600) // 60
#     s = s % 60
#     return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
#
# def _call_choose_move(mod: ModuleType, size: int, moves: List[Move], rng: random.Random, iterations: int, time_limit: float) -> Move:
#     """
#     Call mod.choose_move with compatible args.
#     We always pass (size, moves, rng) positionally.
#     We pass iterations/time_limit as kwargs ONLY if the function accepts them.
#     """
#     fn = mod.choose_move
#     sig = inspect.signature(fn)
#     params = sig.parameters
#     kwargs: Dict[str, Any] = {}
#     if "iterations" in params:
#         kwargs["iterations"] = iterations
#     if "time_limit" in params:
#         kwargs["time_limit"] = time_limit
#
#     return fn(size, moves, rng, **kwargs)
#
# def _extract_last_stats(mod: ModuleType) -> Dict[str, Any]:
#     """
#     Extract stats from a player module if it exposes get_last_stats().
#     Returns a flat dict suitable for CSV, with keys like 'nodes_created', etc.
#     """
#     if not hasattr(mod, "get_last_stats"):
#         return {}
#
#     try:
#         st = mod.get_last_stats()
#     except Exception:
#         return {}
#
#     if st is None:
#         return {}
#
#     # common case: dataclass
#     if is_dataclass(st):
#         d = asdict(st)
#     elif isinstance(st, dict):
#         d = dict(st)
#     else:
#         # try generic object
#         d = {k: getattr(st, k) for k in dir(st) if not k.startswith("_")}
#
#     # normalize values to simple scalars
#     out: Dict[str, Any] = {}
#     for k, v in d.items():
#         if isinstance(v, (int, float, str, bool)) or v is None:
#             out[k] = v
#         else:
#             out[k] = str(v)
#     return out
#
# def _accum_init() -> Dict[str, Any]:
#     # store sums for numeric, last seen for others, max for depth
#     return {
#         "board_size": None,
#         "starting_moves": None,
#         "root_player": None,
#         "iterations_requested": None,
#         "time_limit": None,
#         "iterations_performed": 0,
#         "duration": 0.0,
#         "nodes_created": 0,
#         "nodes_expanded": 0,
#         "max_depth_reached": 0,
#     }
#
# def _accum_add(acc: Dict[str, Any], st: Dict[str, Any]) -> None:
#     # copy non-aggregated “identity” fields from latest stats (they should be consistent)
#     for k in ["board_size", "starting_moves", "root_player", "iterations_requested", "time_limit"]:
#         if k in st and st[k] not in ("", None):
#             acc[k] = st[k]
#
#     # sum numeric fields
#     if "iterations_performed" in st and st["iterations_performed"] not in ("", None):
#         acc["iterations_performed"] += int(st["iterations_performed"])
#     if "duration" in st and st["duration"] not in ("", None):
#         acc["duration"] += float(st["duration"])
#     if "nodes_created" in st and st["nodes_created"] not in ("", None):
#         acc["nodes_created"] += int(st["nodes_created"])
#     if "nodes_expanded" in st and st["nodes_expanded"] not in ("", None):
#         acc["nodes_expanded"] += int(st["nodes_expanded"])
#     if "max_depth_reached" in st and st["max_depth_reached"] not in ("", None):
#         acc["max_depth_reached"] = max(acc["max_depth_reached"], int(st["max_depth_reached"]))
#
# def play_single_game(size: int, black_mod: ModuleType, white_mod: ModuleType, rng: random.Random, iterations: int, time_limit: float) -> Dict[str, Any]:
#     moves: List[Move] = []
#     start = time.perf_counter()
#     black_last: Dict[str, Any] = {}
#     white_last: Dict[str, Any] = {}
#     black_game = _accum_init()
#     white_game = _accum_init()
#
#     while True:
#         to_move_mod = black_mod if len(moves) % 2 == 0 else white_mod
#         to_move_color = "Black" if len(moves) % 2 == 0 else "White"
#
#         try:
#             move = _call_choose_move(
#                 to_move_mod, size, moves, rng, iterations=iterations, time_limit=time_limit
#             )
#         except Exception as err:
#             loser_color = to_move_color
#             winner_color = "White" if loser_color == "Black" else "Black"
#             winner_player = black_mod.__name__ if winner_color == "Black" else white_mod.__name__
#             duration = time.perf_counter() - start
#             return {
#                 "termination": "crash",
#                 "error_type": err.__class__.__name__,
#                 "error_msg": str(err),
#                 "winner_color": winner_color,
#                 "winner_player": winner_player,
#                 "moves_played": len(moves),
#                 "game_time_sec": duration,
#                 "black_last": black_last,
#                 "white_last": white_last,
#                 "black_game": black_game,
#                 "white_game": white_game,
#             }
#
#         # after move, grab stats from that module
#         st = _extract_last_stats(to_move_mod)
#
#         if to_move_mod is black_mod:
#             black_last = st
#             _accum_add(black_game, st)
#         else:
#             white_last = st
#             _accum_add(white_game, st)
#
#         # legality check
#         r, c = move
#         if (not (0 <= r < size and 0 <= c < size)) or (move in moves):
#             loser_color = to_move_color
#             winner_color = "White" if loser_color == "Black" else "Black"
#             winner_player = black_mod.__name__ if winner_color == "Black" else white_mod.__name__
#             duration = time.perf_counter() - start
#             return {
#                 "termination": "illegal_move",
#                 "error_type": "",
#                 "error_msg": f"illegal move {move}",
#                 "winner_color": winner_color,
#                 "winner_player": winner_player,
#                 "moves_played": len(moves),
#                 "game_time_sec": duration,
#                 "black_last": black_last,
#                 "white_last": white_last,
#                 "black_game": black_game,
#                 "white_game": white_game,
#             }
#
#         moves.append(move)
#
#         winner, _ = evaluate_hex(size, moves)
#         if winner is not None:
#             if winner not in ("Black", "White"):
#                 raise RuntimeError(f"evaluate_hex returned unexpected winner {winner!r}")
#             duration = time.perf_counter() - start
#             winner_player = black_mod.__name__ if winner == "Black" else white_mod.__name__
#             return {
#                 "termination": "normal",
#                 "error_type": "",
#                 "error_msg": "",
#                 "winner_color": winner,
#                 "winner_player": winner_player,
#                 "moves_played": len(moves),
#                 "game_time_sec": duration,
#                 "black_last": black_last,
#                 "white_last": white_last,
#                 "black_game": black_game,
#                 "white_game": white_game,
#             }
#
#         if len(moves) == size * size:
#             raise RuntimeError("Full board reached with no winner – impossible in Hex.")
#
# def _clear_screen() -> None:
#     print("\033[H\033[J", end="")
#
# def print_progress_inplace(player_names: List[str], pair_stats: Dict[Tuple[int, int], Dict[str, int]], overall: List[Dict[str, int]], games_per_pair: int, current_round: int, start_time: float) -> None:
#     _clear_screen()
#     elapsed = time.perf_counter() - start_time
#
#     total_pairs = len(pair_stats)
#     total_games_played = sum(ps["games"] for ps in pair_stats.values())
#     total_games_max = total_pairs * games_per_pair
#
#     print("=== Hex Tournament Progress ===")
#     print(f"Round (per pair): {current_round}/{games_per_pair}")
#     print(f"Total games: {total_games_played}/{total_games_max} ({100.0 * total_games_played / total_games_max:.1f}%)")
#     print(f"Elapsed time: {_format_elapsed(elapsed)}")
#     print()
#
#     print("Per-pair (wins):")
#     for (i, j), ps in pair_stats.items():
#         g = ps["games"]
#         if g == 0:
#             continue
#         wi = ps["wins_i"]
#         wj = ps["wins_j"]
#         print(f"  {player_names[i]} vs {player_names[j]}: {g} | {wi}-{wj}  ({100.0*wi/g:.1f}% / {100.0*wj/g:.1f}%)")
#
#     print("\nOverall (wins/losses):")
#     for idx, st in enumerate(overall):
#         g = st["games"]
#         if g == 0:
#             continue
#         w = st["wins"]
#         l = st["losses"]
#         print(f"  {player_names[idx]}: {g} | W {w} ({100.0*w/g:.1f}%) | L {l} ({100.0*l/g:.1f}%)")
#
# def run_tournament(size: int, games_per_pair: int, player_modules: List[ModuleType], seed: int, update_interval_rounds: int, iterations: int, time_limit: float, out_dir: str, experiment_id: str) -> None:
#     os.makedirs(out_dir, exist_ok=True)
#     rng = random.Random(seed)
#     start_time = time.perf_counter()
#     n_players = len(player_modules)
#     if n_players < 2:
#         raise SystemExit("Need at least 2 players.")
#
#     player_names = [m.__name__ for m in player_modules]
#
#     # all pairs i<j
#     pairs: List[Tuple[int, int]] = [(i, j) for i in range(n_players) for j in range(i + 1, n_players)]
#
#     # stats
#     pair_stats: Dict[Tuple[int, int], Dict[str, int]] = {}
#     for (i, j) in pairs:
#         pair_stats[(i, j)] = {
#             "games": 0,
#             "wins_i": 0,
#             "wins_j": 0,
#             "games_i_black": 0,
#             "wins_i_black": 0,
#             "games_i_white": 0,
#             "wins_i_white": 0,
#             "games_j_black": 0,
#             "wins_j_black": 0,
#             "games_j_white": 0,
#             "wins_j_white": 0,
#         }
#
#     overall: List[Dict[str, int]] = []
#     for _ in range(n_players):
#         overall.append({
#             "games": 0,
#             "wins": 0,
#             "losses": 0,
#             "games_black": 0,
#             "wins_black": 0,
#             "games_white": 0,
#             "wins_white": 0,
#         })
#
#     # CSV output (one row per game)
#     games_csv_path = os.path.join(out_dir, f"{experiment_id}_games.csv")
#     pairs_csv_path = os.path.join(out_dir, f"{experiment_id}_pairs.csv")
#
#     fieldnames = [
#     "experiment_id",
#     "timestamp_start",
#     "seed",
#     "size",
#     "iterations",
#     "time_limit",
#     "round_index",
#     "pair_index",
#     "game_index_global",
#     "player_black",
#     "player_white",
#     "winner_color",
#     "winner_player",
#     "termination",
#     "error_type",
#     "error_msg",
#     "moves_played",
#     "game_time_sec",
#     ]
#
#     # Stats from the last decision made by each bot (optional but useful)
#     for k in MCTS_STAT_KEYS:
#         fieldnames.append(f"black_last_{k}")
#         fieldnames.append(f"white_last_{k}")
#
#     # Whole-game aggregated stats (what you asked for)
#     for k in MCTS_STAT_KEYS:
#         fieldnames.append(f"black_game_{k}")
#         fieldnames.append(f"white_game_{k}")
#
#     timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")
#
#     with open(games_csv_path, "w", newline="") as f_games:
#         writer = csv.DictWriter(f_games, fieldnames=fieldnames)
#         writer.writeheader()
#
#         game_index_global = 0
#
#         for round_idx in range(1, games_per_pair + 1):
#             for pair_idx, (i, j) in enumerate(pairs):
#                 # alternating colors for this pair
#                 if round_idx % 2 == 1:
#                     black_idx, white_idx = i, j
#                 else:
#                     black_idx, white_idx = j, i
#
#                 black_mod = player_modules[black_idx]
#                 white_mod = player_modules[white_idx]
#
#                 result = play_single_game(
#                     size=size,
#                     black_mod=black_mod,
#                     white_mod=white_mod,
#                     rng=rng,
#                     iterations=iterations,
#                     time_limit=time_limit,
#                 )
#
#                 game_index_global += 1
#
#                 winner_color = result["winner_color"]
#                 if winner_color not in ("Black", "White"):
#                     raise RuntimeError(f"Unexpected winner_color {winner_color!r}")
#
#                 # update stats
#                 ps = pair_stats[(i, j)]
#                 ps["games"] += 1
#
#                 # color game counts
#                 if black_idx == i:
#                     ps["games_i_black"] += 1
#                     ps["games_j_white"] += 1
#                 else:
#                     ps["games_j_black"] += 1
#                     ps["games_i_white"] += 1
#
#                 # overall games
#                 overall[black_idx]["games"] += 1
#                 overall[white_idx]["games"] += 1
#                 overall[black_idx]["games_black"] += 1
#                 overall[white_idx]["games_white"] += 1
#
#                 if winner_color == "Black":
#                     winner_idx = black_idx
#                     loser_idx = white_idx
#
#                     if winner_idx == i:
#                         ps["wins_i"] += 1
#                         ps["wins_i_black"] += 1
#                     else:
#                         ps["wins_j"] += 1
#                         ps["wins_j_black"] += 1
#
#                     overall[winner_idx]["wins"] += 1
#                     overall[winner_idx]["wins_black"] += 1
#                     overall[loser_idx]["losses"] += 1
#
#                 else:  # White
#                     winner_idx = white_idx
#                     loser_idx = black_idx
#
#                     if winner_idx == i:
#                         ps["wins_i"] += 1
#                         ps["wins_i_white"] += 1
#                     else:
#                         ps["wins_j"] += 1
#                         ps["wins_j_white"] += 1
#
#                     overall[winner_idx]["wins"] += 1
#                     overall[winner_idx]["wins_white"] += 1
#                     overall[loser_idx]["losses"] += 1
#
#                 # Build CSV row
#                 row: Dict[str, Any] = {
#                     "experiment_id": experiment_id,
#                     "timestamp_start": timestamp_start,
#                     "seed": seed,
#                     "size": size,
#                     "iterations": iterations,
#                     "time_limit": time_limit,
#                     "round_index": round_idx,
#                     "pair_index": pair_idx,
#                     "game_index_global": game_index_global,
#                     "player_black": black_mod.__name__,
#                     "player_white": white_mod.__name__,
#                     "winner_color": result["winner_color"],
#                     "winner_player": result["winner_player"],
#                     "termination": result["termination"],
#                     "error_type": result["error_type"],
#                     "error_msg": result["error_msg"],
#                     "moves_played": result["moves_played"],
#                     "game_time_sec": result["game_time_sec"],
#                 }
#
#                 # attach MCTS stats (last-decision + whole-game totals)
#                 blast = result.get("black_last", {}) or {}
#                 wlast = result.get("white_last", {}) or {}
#                 bgame = result.get("black_game", {}) or {}
#                 wgame = result.get("white_game", {}) or {}
#
#                 for k in MCTS_STAT_KEYS:
#                     row[f"black_last_{k}"] = blast.get(k, "")
#                     row[f"white_last_{k}"] = wlast.get(k, "")
#                     row[f"black_game_{k}"] = bgame.get(k, "")
#                     row[f"white_game_{k}"] = wgame.get(k, "")
#
#                 writer.writerow(row)
#
#             # progress update
#             if (update_interval_rounds > 0 and round_idx % update_interval_rounds == 0) or round_idx == games_per_pair:
#                 print_progress_inplace(player_names, pair_stats, overall, games_per_pair, round_idx, start_time)
#
#     # Write pair summary CSV
#     with open(pairs_csv_path, "w", newline="") as f_pairs:
#         pair_fields = [
#             "experiment_id",
#             "timestamp_start",
#             "seed",
#             "size",
#             "iterations",
#             "time_limit",
#             "player_i",
#             "player_j",
#             "games",
#             "wins_i",
#             "wins_j",
#             "games_i_black",
#             "wins_i_black",
#             "games_i_white",
#             "wins_i_white",
#             "games_j_black",
#             "wins_j_black",
#             "games_j_white",
#             "wins_j_white",
#         ]
#         w = csv.DictWriter(f_pairs, fieldnames=pair_fields)
#         w.writeheader()
#         for (i, j), ps in pair_stats.items():
#             w.writerow({
#                 "experiment_id": experiment_id,
#                 "timestamp_start": timestamp_start,
#                 "seed": seed,
#                 "size": size,
#                 "iterations": iterations,
#                 "time_limit": time_limit,
#                 "player_i": player_names[i],
#                 "player_j": player_names[j],
#                 **ps,
#             })
#
#     elapsed = time.perf_counter() - start_time
#     print("\n=== Done ===")
#     print(f"Wrote: {games_csv_path}")
#     print(f"Wrote: {pairs_csv_path}")
#     print(f"Elapsed: {_format_elapsed(elapsed)}")
#
# def main() -> None:
#     ap = argparse.ArgumentParser(description="Hex multi-arena: interleaved round-robin + CSV logging")
#     ap.add_argument("--players", nargs="+", required=True, help="player module names (importable)")
#     ap.add_argument("--size", type=int, default=7, help="board size n (n×n)")
#     ap.add_argument("--games-per-pair", type=int, default=100, help="games each pair plays")
#     ap.add_argument("--seed", type=int, default=42, help="RNG seed")
#     ap.add_argument("--iterations", type=int, default=1000, help="iterations passed to bots (if supported)")
#     ap.add_argument("--time-limit", type=float, default=1.0, help="time limit seconds passed to bots (if supported)")
#     ap.add_argument("--update-interval", type=int, default=1, help="rounds between display updates")
#     ap.add_argument("--out-dir", type=str, default="arena_results", help="directory to write CSV files")
#     ap.add_argument("--experiment-id", type=str, default="", help="identifier for this run (default auto)")
#     args = ap.parse_args()
#     mods = [load_player(name) for name in args.players]
#     experiment_id = args.experiment_id.strip() or time.strftime("exp_%Y%m%d_%H%M%S")
#
#     run_tournament(
#         size=args.size,
#         games_per_pair=args.games_per_pair,
#         player_modules=mods,
#         seed=args.seed,
#         update_interval_rounds=args.update_interval,
#         iterations=args.iterations,
#         time_limit=args.time_limit,
#         out_dir=args.out_dir,
#         experiment_id=experiment_id,
#     )
#
# if __name__ == "__main__":
#     main()

# hex_multi_arena.py

import argparse
import csv
import importlib
import inspect
import os
import random
import time
from dataclasses import asdict, is_dataclass
from types import ModuleType
from typing import List, Tuple, Dict, Any, Optional

from hexfast import evaluate_hex

import multiprocessing as mp  # add near imports

Move = Tuple[int, int]

MCTS_STAT_KEYS = [
    "board_size",
    "starting_moves",
    "root_player",
    "iterations_requested",
    "iterations_performed",
    "time_limit",
    "duration",
    "nodes_created",
    "nodes_expanded",
    "max_depth_reached",
]

def _play_game_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker entry point: imports player modules inside the process,
    plays one game, returns a dict with everything the parent needs.
    """
    import importlib
    import random

    black_name = payload["player_black"]
    white_name = payload["player_white"]

    black_mod = importlib.import_module(black_name)
    white_mod = importlib.import_module(white_name)

    rng = random.Random(payload["game_seed"])

    result = play_single_game(
        size=payload["size"],
        black_mod=black_mod,
        white_mod=white_mod,
        rng=rng,
        iterations=payload["iterations"],
        time_limit=payload["time_limit"],
    )

    # Return result plus identifying info the parent needs to update stats/CSV
    return {
        **payload,
        "result": result,
    }

def load_player(module_name: str) -> ModuleType:
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(f"Cannot import player module '{module_name}': {e}")
    if not hasattr(mod, "choose_move"):
        raise SystemExit(f"Player module '{module_name}' lacks a choose_move() function.")
    return mod


def _format_elapsed(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _clear_screen() -> None:
    print("\033[H\033[J", end="")


def _call_choose_move(
    mod: ModuleType,
    size: int,
    moves: List[Move],
    rng: random.Random,
    iterations: int,
    time_limit: float,
) -> Move:
    fn = mod.choose_move
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs: Dict[str, Any] = {}
    if "iterations" in params:
        kwargs["iterations"] = iterations
    if "time_limit" in params:
        kwargs["time_limit"] = time_limit

    return fn(size, moves, rng, **kwargs)


def _extract_last_stats(mod: ModuleType) -> Dict[str, Any]:
    if not hasattr(mod, "get_last_stats"):
        return {}
    try:
        st = mod.get_last_stats()
    except Exception:
        return {}
    if st is None:
        return {}

    if is_dataclass(st):
        d = asdict(st)
    elif isinstance(st, dict):
        d = dict(st)
    else:
        d = {k: getattr(st, k) for k in dir(st) if not k.startswith("_")}

    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _accum_init() -> Dict[str, Any]:
    return {
        "board_size": None,
        "starting_moves": None,
        "root_player": None,
        "iterations_requested": None,
        "time_limit": None,
        "iterations_performed": 0,
        "duration": 0.0,
        "nodes_created": 0,
        "nodes_expanded": 0,
        "max_depth_reached": 0,
    }


def _accum_add(acc: Dict[str, Any], st: Dict[str, Any]) -> None:
    for k in ["board_size", "starting_moves", "root_player", "iterations_requested", "time_limit"]:
        if k in st and st[k] not in ("", None):
            acc[k] = st[k]

    if "iterations_performed" in st and st["iterations_performed"] not in ("", None):
        acc["iterations_performed"] += int(st["iterations_performed"])
    if "duration" in st and st["duration"] not in ("", None):
        acc["duration"] += float(st["duration"])
    if "nodes_created" in st and st["nodes_created"] not in ("", None):
        acc["nodes_created"] += int(st["nodes_created"])
    if "nodes_expanded" in st and st["nodes_expanded"] not in ("", None):
        acc["nodes_expanded"] += int(st["nodes_expanded"])
    if "max_depth_reached" in st and st["max_depth_reached"] not in ("", None):
        acc["max_depth_reached"] = max(acc["max_depth_reached"], int(st["max_depth_reached"]))


def play_single_game(
    size: int,
    black_mod: ModuleType,
    white_mod: ModuleType,
    rng: random.Random,
    iterations: int,
    time_limit: float,
) -> Dict[str, Any]:
    moves: List[Move] = []
    start = time.perf_counter()

    black_last: Dict[str, Any] = {}
    white_last: Dict[str, Any] = {}
    black_game = _accum_init()
    white_game = _accum_init()

    while True:
        to_move_mod = black_mod if len(moves) % 2 == 0 else white_mod
        to_move_color = "Black" if len(moves) % 2 == 0 else "White"

        try:
            move = _call_choose_move(to_move_mod, size, moves, rng, iterations, time_limit)
        except Exception as err:
            loser_color = to_move_color
            winner_color = "White" if loser_color == "Black" else "Black"
            winner_player = black_mod.__name__ if winner_color == "Black" else white_mod.__name__
            return {
                "termination": "crash",
                "error_type": err.__class__.__name__,
                "error_msg": str(err),
                "winner_color": winner_color,
                "winner_player": winner_player,
                "moves_played": len(moves),
                "game_time_sec": time.perf_counter() - start,
                "black_last": black_last,
                "white_last": white_last,
                "black_game": black_game,
                "white_game": white_game,
            }

        st = _extract_last_stats(to_move_mod)
        if to_move_mod is black_mod:
            black_last = st
            _accum_add(black_game, st)
        else:
            white_last = st
            _accum_add(white_game, st)

        r, c = move
        if (not (0 <= r < size and 0 <= c < size)) or (move in moves):
            loser_color = to_move_color
            winner_color = "White" if loser_color == "Black" else "Black"
            winner_player = black_mod.__name__ if winner_color == "Black" else white_mod.__name__
            return {
                "termination": "illegal_move",
                "error_type": "",
                "error_msg": f"illegal move {move}",
                "winner_color": winner_color,
                "winner_player": winner_player,
                "moves_played": len(moves),
                "game_time_sec": time.perf_counter() - start,
                "black_last": black_last,
                "white_last": white_last,
                "black_game": black_game,
                "white_game": white_game,
            }

        moves.append(move)

        winner, _ = evaluate_hex(size, moves)
        if winner is not None:
            if winner not in ("Black", "White"):
                raise RuntimeError(f"evaluate_hex returned unexpected winner {winner!r}")
            winner_player = black_mod.__name__ if winner == "Black" else white_mod.__name__
            return {
                "termination": "normal",
                "error_type": "",
                "error_msg": "",
                "winner_color": winner,
                "winner_player": winner_player,
                "moves_played": len(moves),
                "game_time_sec": time.perf_counter() - start,
                "black_last": black_last,
                "white_last": white_last,
                "black_game": black_game,
                "white_game": white_game,
            }

        if len(moves) == size * size:
            raise RuntimeError("Full board reached with no winner – impossible in Hex.")


def print_progress_inplace(
    player_names: List[str],
    pair_wins: Dict[Tuple[int, int], Dict[str, int]],
    overall: List[Dict[str, int]],
    games_per_pair: int,
    round_idx: int,
    sweep_done: int,
    sweep_total: int,
    start_time: float,
    sweep_label: str,
) -> None:
    _clear_screen()
    elapsed = time.perf_counter() - start_time

    total_pairs = len(pair_wins)
    total_games_played = sum(ps["games"] for ps in pair_wins.values())
    total_games_max = total_pairs * games_per_pair

    print("=== Hex Sweep Progress ===")
    print(f"Sweep item: {sweep_done}/{sweep_total}   ({sweep_label})")
    print(f"Round (per pair): {round_idx}/{games_per_pair}")
    print(f"Total games (this sweep item): {total_games_played}/{total_games_max} ({100.0 * total_games_played / total_games_max:.1f}%)")
    print(f"Elapsed: {_format_elapsed(elapsed)}")
    print()

    print("Per-pair (wins):")
    for (i, j), ps in pair_wins.items():
        g = ps["games"]
        if g == 0:
            continue
        wi = ps["wins_i"]
        wj = ps["wins_j"]
        print(f"  {player_names[i]} vs {player_names[j]}: {g} | {wi}-{wj}  ({100.0*wi/g:.1f}% / {100.0*wj/g:.1f}%)")

    print("\nOverall (wins/losses):")
    for idx, st in enumerate(overall):
        g = st["games"]
        if g == 0:
            continue
        w = st["wins"]
        l = st["losses"]
        print(f"  {player_names[idx]}: {g} | W {w} ({100.0*w/g:.1f}%) | L {l} ({100.0*l/g:.1f}%)")


def _build_fieldnames() -> List[str]:
    base = [
        "results_prefix",
        "experiment_id",
        "run_index",
        "sweep_param",
        "sweep_value",
        "timestamp_start",
        "seed",
        "size",
        "iterations",
        "time_limit",
        "round_index",
        "pair_index",
        "game_index_global",
        "player_black",
        "player_white",
        "winner_color",
        "winner_player",
        "termination",
        "error_type",
        "error_msg",
        "moves_played",
        "game_time_sec",
    ]
    for k in MCTS_STAT_KEYS:
        base.append(f"black_last_{k}")
        base.append(f"white_last_{k}")
    for k in MCTS_STAT_KEYS:
        base.append(f"black_game_{k}")
        base.append(f"white_game_{k}")
    return base


def _open_csv_writer(path: str, fieldnames: List[str], append: bool) -> tuple[csv.DictWriter, Any]:
    exists = os.path.exists(path)
    mode = "a" if append else "w"
    f = open(path, mode, newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if (not exists) or (not append):
        writer.writeheader()
    return writer, f


def run_sweep(
    size: int,
    iterations: int,
    time_limit: float,
    games_per_pair: int,
    player_modules: List[ModuleType],
    seed: int,
    update_interval_rounds: int,
    out_dir: str,
    results_prefix: str,
    append: bool,
    sweep_param: str,
    sweep_values: List[str],
    workers: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    start_wall = time.perf_counter()

    player_names = [m.__name__ for m in player_modules]
    n = len(player_modules)
    if n < 2:
        raise SystemExit("Need at least 2 players.")

    pairs: List[Tuple[int, int]] = [(i, j) for i in range(n) for j in range(i + 1, n)]
    fieldnames = _build_fieldnames()

    games_path = os.path.join(out_dir, f"{results_prefix}_games.csv")
    pairs_path = os.path.join(out_dir, f"{results_prefix}_pairs.csv")

    games_writer, games_file = _open_csv_writer(games_path, fieldnames, append=append)
    # pair summary writer (simpler; we append one row per pair per sweep item)
    pair_fields = [
        "results_prefix", "experiment_id", "run_index", "sweep_param", "sweep_value",
        "timestamp_start", "seed", "size", "iterations", "time_limit",
        "player_i", "player_j",
        "games", "wins_i", "wins_j",
        "games_i_black", "wins_i_black",
        "games_i_white", "wins_i_white",
        "games_j_black", "wins_j_black",
        "games_j_white", "wins_j_white",
    ]
    pairs_writer, pairs_file = _open_csv_writer(pairs_path, pair_fields, append=append)

    game_index_global = 0
    timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        for run_index, sval in enumerate(sweep_values, start=1):
            # resolve per-run parameters
            run_size = size
            run_iterations = iterations
            run_time_limit = time_limit

            if sweep_param == "size":
                run_size = int(sval)
            elif sweep_param == "iterations":
                run_iterations = int(sval)
            elif sweep_param == "time_limit":
                run_time_limit = float(sval)
            else:
                raise ValueError("sweep_param must be one of: size, iterations, time_limit")

            experiment_id = f"{results_prefix}__{sweep_param}={sval}"

            # reset per-sweep-item progress stats
            pair_stats: Dict[Tuple[int, int], Dict[str, int]] = {}
            for (i, j) in pairs:
                pair_stats[(i, j)] = {
                    "games": 0, "wins_i": 0, "wins_j": 0,
                    "games_i_black": 0, "wins_i_black": 0,
                    "games_i_white": 0, "wins_i_white": 0,
                    "games_j_black": 0, "wins_j_black": 0,
                    "games_j_white": 0, "wins_j_white": 0,
                }
            overall = [{"games": 0, "wins": 0, "losses": 0} for _ in range(n)]




            # choose multiprocessing context (macOS works best with spawn)
            ctx = mp.get_context("spawn")

            pool = None
            if workers > 1:
                pool = ctx.Pool(processes=workers)

            # interleaved round-robin
            for round_idx in range(1, games_per_pair + 1):

                # ---- Build one task per pair for this round ----
                tasks: List[Dict[str, Any]] = []
                for pair_idx, (i, j) in enumerate(pairs):
                    if round_idx % 2 == 1:
                        black_idx, white_idx = i, j
                    else:
                        black_idx, white_idx = j, i

                    black_name = player_modules[black_idx].__name__
                    white_name = player_modules[white_idx].__name__

                    # deterministic per-game seed
                    # (use only ints; keep it stable across runs)
                    game_seed = (seed * 1_000_003 + run_index * 10_000 + round_idx * 100 + pair_idx) & 0xFFFFFFFF

                    tasks.append({
                        "results_prefix": results_prefix,
                        "experiment_id": experiment_id,
                        "run_index": run_index,
                        "sweep_param": sweep_param,
                        "sweep_value": sval,
                        "timestamp_start": timestamp_start,
                        "seed": seed,

                        "size": run_size,
                        "iterations": run_iterations,
                        "time_limit": run_time_limit,

                        "round_index": round_idx,
                        "pair_index": pair_idx,

                        "i": i, "j": j,
                        "black_idx": black_idx, "white_idx": white_idx,

                        "player_black": black_name,
                        "player_white": white_name,

                        "game_seed": game_seed,
                    })

                # ---- Execute this round (parallel if workers > 1) ----
                if workers <= 1:
                    results_payloads = [_play_game_task(t) for t in tasks]
                else:
                    results_payloads = pool.map(_play_game_task, tasks)

                # ---- Consume results in deterministic order (pair_index) ----
                results_payloads.sort(key=lambda d: d["pair_index"])

                for payload in results_payloads:
                    pair_idx = payload["pair_index"]
                    i = payload["i"]; j = payload["j"]
                    black_idx = payload["black_idx"]; white_idx = payload["white_idx"]

                    result = payload["result"]
                    game_index_global += 1

                    winner_color = result["winner_color"]
                    if winner_color not in ("Black", "White"):
                        raise RuntimeError(f"Unexpected winner_color {winner_color!r}")

                    ps = pair_stats[(i, j)]
                    ps["games"] += 1

                    if black_idx == i:
                        ps["games_i_black"] += 1
                        ps["games_j_white"] += 1
                    else:
                        ps["games_j_black"] += 1
                        ps["games_i_white"] += 1

                    overall[black_idx]["games"] += 1
                    overall[white_idx]["games"] += 1

                    if winner_color == "Black":
                        winner_idx = black_idx
                        loser_idx = white_idx
                        if winner_idx == i:
                            ps["wins_i"] += 1
                            ps["wins_i_black"] += 1
                        else:
                            ps["wins_j"] += 1
                            ps["wins_j_black"] += 1
                    else:
                        winner_idx = white_idx
                        loser_idx = black_idx
                        if winner_idx == i:
                            ps["wins_i"] += 1
                            ps["wins_i_white"] += 1
                        else:
                            ps["wins_j"] += 1
                            ps["wins_j_white"] += 1

                    overall[winner_idx]["wins"] += 1
                    overall[loser_idx]["losses"] += 1

                    # CSV row (note: now player names are strings in payload)
                    row: Dict[str, Any] = {
                        "results_prefix": results_prefix,
                        "experiment_id": experiment_id,
                        "run_index": run_index,
                        "sweep_param": sweep_param,
                        "sweep_value": sval,
                        "timestamp_start": timestamp_start,
                        "seed": seed,
                        "size": run_size,
                        "iterations": run_iterations,
                        "time_limit": run_time_limit,
                        "round_index": round_idx,
                        "pair_index": pair_idx,
                        "game_index_global": game_index_global,
                        "player_black": payload["player_black"],
                        "player_white": payload["player_white"],
                        "winner_color": result["winner_color"],
                        "winner_player": result["winner_player"],
                        "termination": result["termination"],
                        "error_type": result["error_type"],
                        "error_msg": result["error_msg"],
                        "moves_played": result["moves_played"],
                        "game_time_sec": result["game_time_sec"],
                    }

                    blast = result.get("black_last", {}) or {}
                    wlast = result.get("white_last", {}) or {}
                    bgame = result.get("black_game", {}) or {}
                    wgame = result.get("white_game", {}) or {}

                    for k in MCTS_STAT_KEYS:
                        row[f"black_last_{k}"] = blast.get(k, "")
                        row[f"white_last_{k}"] = wlast.get(k, "")
                        row[f"black_game_{k}"] = bgame.get(k, "")
                        row[f"white_game_{k}"] = wgame.get(k, "")

                    games_writer.writerow(row)

                if (update_interval_rounds > 0 and round_idx % update_interval_rounds == 0) or round_idx == games_per_pair:
                    label = f"{sweep_param}={sval}"
                    print_progress_inplace(
                        player_names, pair_stats, overall,
                        games_per_pair, round_idx,
                        run_index, len(sweep_values),
                        start_wall, label
                    )

            if pool is not None:
                pool.close()
                pool.join()






            # write pair summary rows for this sweep item
            for (i, j), ps in pair_stats.items():
                pairs_writer.writerow({
                    "results_prefix": results_prefix,
                    "experiment_id": experiment_id,
                    "run_index": run_index,
                    "sweep_param": sweep_param,
                    "sweep_value": sval,
                    "timestamp_start": timestamp_start,
                    "seed": seed,
                    "size": run_size,
                    "iterations": run_iterations,
                    "time_limit": run_time_limit,
                    "player_i": player_names[i],
                    "player_j": player_names[j],
                    **ps,
                })

    finally:
        games_file.close()
        pairs_file.close()

    _clear_screen()
    print("=== Sweep complete ===")
    print(f"Wrote/appended games to: {games_path}")
    print(f"Wrote/appended pairs to: {pairs_path}")
    print(f"Elapsed: {_format_elapsed(time.perf_counter() - start_wall)}")


def _parse_sweep_arg(s: str) -> tuple[str, List[str]]:
    """
    Parse:  --sweep time_limit 0.05,0.1,0.2
    Returns ("time_limit", ["0.05","0.1","0.2"])
    """
    raise RuntimeError("internal: use argparse to parse as two tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hex multi-arena: interleaved round-robin + CSV logging + sweeps")
    ap.add_argument("--players", nargs="+", required=True, help="player module names (importable)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")

    ap.add_argument("--games-per-pair", type=int, default=50, help="games each pair plays per sweep item")
    ap.add_argument("--update-interval", type=int, default=1, help="rounds between progress updates")

    # baseline params (used unless swept)
    ap.add_argument("--size", type=int, default=7, help="board size n (n×n)")
    ap.add_argument("--iterations", type=int, default=500, help="iterations passed to bots (if supported)")
    ap.add_argument("--time-limit", type=float, default=0.5, help="time limit seconds passed to bots (if supported)")

    # sweep: exactly one parameter at a time
    ap.add_argument(
        "--sweep",
        nargs=2,
        metavar=("PARAM", "VALUES"),
        help="Run multiple experiments varying one of: size, iterations, time_limit. "
             "VALUES is a comma-separated list, e.g. time_limit 0.05,0.1,0.2",
    )

    ap.add_argument("--out-dir", type=str, default="arena_results", help="directory to write CSV files")
    ap.add_argument("--results-prefix", type=str, default="study", help="shared CSV prefix for all sweep items")
    ap.add_argument("--append", action="store_true", help="append to existing CSVs instead of overwriting")
    ap.add_argument("--workers", type=int, default=1, help="number of worker processes (1 = no multiprocessing)")


    args = ap.parse_args()

    mods = [load_player(name) for name in args.players]

    if args.sweep is None:
        sweep_param = "time_limit"
        sweep_values = [str(args.time_limit)]
    else:
        sweep_param = args.sweep[0].strip()
        if sweep_param not in ("size", "iterations", "time_limit"):
            raise SystemExit("PARAM must be one of: size, iterations, time_limit")
        sweep_values = [v.strip() for v in args.sweep[1].split(",") if v.strip()]
        if not sweep_values:
            raise SystemExit("Sweep VALUES list is empty.")

    run_sweep(
        size=args.size,
        iterations=args.iterations,
        time_limit=args.time_limit,
        games_per_pair=args.games_per_pair,
        player_modules=mods,
        seed=args.seed,
        update_interval_rounds=args.update_interval,
        out_dir=args.out_dir,
        results_prefix=args.results_prefix,
        append=args.append,
        sweep_param=sweep_param,
        sweep_values=sweep_values,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
