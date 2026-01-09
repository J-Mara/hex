# summarize_results.py
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# Metrics we support from the CSV:
# We will read from black_game_<metric> / white_game_<metric>.
SUPPORTED_METRICS = {
    "winrate": None,  # special case
    "nodes_created": "nodes_created",
    "nodes_expanded": "nodes_expanded",
    "max_depth_reached": "max_depth_reached",
    "iterations_performed": "iterations_performed",
    "duration": "duration",  # seconds spent inside choose_move summed over the game
}


def read_games(path: str) -> List[dict]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _sorted_sweep_values(values: List[str]) -> List[str]:
    # Sort numerically if possible, else lexicographically.
    if all(_is_number(v) for v in values):
        return [v for v in sorted(values, key=lambda x: float(x))]
    return sorted(values)


def summarize_overall_winrate(rows: List[dict], sweep_param: str):
    # wins[sval][player] = wins
    # games[sval][player] = games
    wins = defaultdict(lambda: defaultdict(int))
    games = defaultdict(lambda: defaultdict(int))

    for row in rows:
        if row.get("sweep_param") != sweep_param:
            continue
        sval = row["sweep_value"]
        pb = row["player_black"]
        pw = row["player_white"]
        wp = row["winner_player"]

        games[sval][pb] += 1
        games[sval][pw] += 1
        wins[sval][wp] += 1

    return wins, games


def summarize_metric_means(
    rows: List[dict],
    sweep_param: str,
    metric_key: str,
    per_move: bool,
):
    """
    Return:
      sums[sval][player] = sum(metric over that player's games)
      counts[sval][player] = number of games included
    where metric is averaged per game (or per move if per_move=True).
    """
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    # We'll use black_game_<metric>, white_game_<metric>.
    col_black = f"black_game_{metric_key}"
    col_white = f"white_game_{metric_key}"

    for row in rows:
        if row.get("sweep_param") != sweep_param:
            continue
        sval = row["sweep_value"]

        pb = row["player_black"]
        pw = row["player_white"]

        # normalization factor
        denom = 1.0
        if per_move:
            try:
                denom = float(row.get("moves_played", "0") or "0")
            except Exception:
                denom = 0.0
            if denom <= 0:
                # skip if we can't normalize
                continue

        # black
        vb = row.get(col_black, "")
        if vb != "":
            try:
                sums[sval][pb] += float(vb) / denom
                counts[sval][pb] += 1
            except Exception:
                pass

        # white
        vw = row.get(col_white, "")
        if vw != "":
            try:
                sums[sval][pw] += float(vw) / denom
                counts[sval][pw] += 1
            except Exception:
                pass

    return sums, counts


def _collect_players_from_rows(rows: List[dict], sweep_param: str) -> List[str]:
    players = set()
    for row in rows:
        if row.get("sweep_param") != sweep_param:
            continue
        players.add(row["player_black"])
        players.add(row["player_white"])
    return sorted(players)


def print_table_winrate(wins, games):
    sweep_values = _sorted_sweep_values(list(games.keys()))
    players = sorted({p for sval in games for p in games[sval].keys()})

    print("=== Overall win rates by sweep value ===")
    header = ["sweep_value"] + players
    print("\t".join(header))

    for sval in sweep_values:
        row = [sval]
        for p in players:
            g = games[sval].get(p, 0)
            w = wins[sval].get(p, 0)
            row.append("n/a" if g == 0 else f"{100.0*w/g:.1f}% ({w}/{g})")
        print("\t".join(row))


def print_table_metric(sums, counts, metric: str, per_move: bool):
    sweep_values = _sorted_sweep_values(list({*sums.keys(), *counts.keys()}))
    players = sorted({p for sval in counts for p in counts[sval].keys()})

    label = f"{metric} per move" if per_move else f"{metric} per game"
    print(f"=== Mean {label} by sweep value ===")
    header = ["sweep_value"] + players
    print("\t".join(header))

    for sval in sweep_values:
        row = [sval]
        for p in players:
            c = counts[sval].get(p, 0)
            s = sums[sval].get(p, 0.0)
            row.append("n/a" if c == 0 else f"{(s/c):.4g} ({c}g)")
        print("\t".join(row))


def maybe_plot_winrate(rows: List[dict], sweep_param: str, out_png: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not available, skipping plot)")
        return

    wins, games = summarize_overall_winrate(rows, sweep_param)
    sval_list = _sorted_sweep_values(list(games.keys()))

    # x axis numeric if possible
    x = [float(s) if _is_number(s) else s for s in sval_list]
    players = sorted({p for sval in games for p in games[sval].keys()})

    for p in players:
        y = []
        for sval in sval_list:
            g = games[sval].get(p, 0)
            w = wins[sval].get(p, 0)
            y.append((w / g) if g else 0.0)
        plt.plot(x, y, marker="o", label=p)

    plt.xlabel(sweep_param)
    plt.ylabel("win rate")
    plt.title(f"Win rate vs {sweep_param}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"\nWrote plot: {out_png}")


def maybe_plot_metric(rows: List[dict], sweep_param: str, metric_key: str, per_move: bool, out_png: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not available, skipping plot)")
        return

    sums, counts = summarize_metric_means(rows, sweep_param, metric_key, per_move=per_move)
    sval_list = _sorted_sweep_values(list({*sums.keys(), *counts.keys()}))
    x = [float(s) if _is_number(s) else s for s in sval_list]
    players = _collect_players_from_rows(rows, sweep_param)

    for p in players:
        y = []
        for sval in sval_list:
            c = counts[sval].get(p, 0)
            s = sums[sval].get(p, 0.0)
            y.append((s / c) if c else 0.0)
        plt.plot(x, y, marker="o", label=p)

    plt.xlabel(sweep_param)
    plt.ylabel(f"mean {metric_key} per move" if per_move else f"mean {metric_key} per game")
    plt.title(f"{metric_key} vs {sweep_param}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"\nWrote plot: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Summarize Hex arena CSV results (no pandas).")
    ap.add_argument("--games-csv", required=True, help="path to <prefix>_games.csv")
    ap.add_argument("--sweep-param", choices=["size", "iterations", "time_limit"], required=True)

    ap.add_argument(
        "--metric",
        choices=list(SUPPORTED_METRICS.keys()),
        default="winrate",
        help="What to summarize/plot. winrate or an MCTS stat from *_game_* columns.",
    )
    ap.add_argument(
        "--per-move",
        action="store_true",
        help="Normalize stats by moves_played (useful when sweeping size).",
    )

    ap.add_argument("--pairwise", action="store_true", help="also print pairwise win summaries (winrate only)")
    ap.add_argument("--plot", type=str, default="", help="write a PNG plot (requires matplotlib)")

    args = ap.parse_args()

    rows = read_games(args.games_csv)

    if args.metric == "winrate":
        wins, games = summarize_overall_winrate(rows, args.sweep_param)
        print_table_winrate(wins, games)
        if args.plot:
            maybe_plot_winrate(rows, args.sweep_param, args.plot)
        if args.pairwise:
            # keep it simple: reuse your older pairwise logic if you want;
            # winrate pairwise printing isn't changed here.
            print("\n(pairwise not implemented in this version; you can keep your old pairwise function if needed)")
        return

    metric_key = SUPPORTED_METRICS[args.metric]
    assert metric_key is not None

    sums, counts = summarize_metric_means(rows, args.sweep_param, metric_key, per_move=args.per_move)
    print_table_metric(sums, counts, metric=args.metric, per_move=args.per_move)

    if args.plot:
        maybe_plot_metric(rows, args.sweep_param, metric_key, per_move=args.per_move, out_png=args.plot)


if __name__ == "__main__":
    main()
