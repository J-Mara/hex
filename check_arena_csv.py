import csv
import sys
from collections import Counter, defaultdict

def main(path: str) -> None:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        print("File:", path)
        print("Rows:", len(rows))
        print("Columns:", len(r.fieldnames or []))
        print("First 20 columns:", (r.fieldnames or [])[:20])
        print()

        if not rows:
            print("ERROR: CSV is empty.")
            sys.exit(1)

        # Basic required fields
        required = [
            "size", "iterations", "time_limit",
            "player_black", "player_white",
            "winner_color", "winner_player",
            "termination",
        ]
        missing = [c for c in required if c not in (r.fieldnames or [])]
        if missing:
            print("ERROR: Missing required columns:", missing)
            sys.exit(1)

        # Check winners and terminations
        winners = Counter(row["winner_color"] for row in rows)
        terms = Counter(row["termination"] for row in rows)
        print("Winner_color counts:", dict(winners))
        print("Termination counts:", dict(terms))

        bad_winners = [w for w in winners if w not in ("Black", "White")]
        if bad_winners:
            print("ERROR: Unexpected winner_color values:", bad_winners)
            sys.exit(1)

        # Check if any stats columns exist and are populated sometimes
        # stat_cols = [c for c in (r.fieldnames or []) if c.startswith("black_mcts_") or c.startswith("white_mcts_")]
        stat_cols = [
            c for c in (r.fieldnames or [])
            if c.startswith("black_last_") or c.startswith("white_last_")
            or c.startswith("black_game_") or c.startswith("white_game_")
        ]


        print("\nStats columns found:", len(stat_cols))
        if stat_cols:
            nonempty = 0
            for row in rows:
                for c in stat_cols:
                    if row.get(c, "") not in ("", None):
                        nonempty += 1
                        break
            print("Rows with at least one non-empty stat:", nonempty, "/", len(rows))
        else:
            print("NOTE: No mcts stats columns found. This is OK if players don't expose get_last_stats().")

        # Quick per-pair win counts
        pair_wins = defaultdict(lambda: Counter())
        for row in rows:
            pair = (row["player_black"], row["player_white"])
            pair_wins[pair][row["winner_player"]] += 1

        print("\nExample per-(black,white) win counts (first 5):")
        for i, (pair, cnt) in enumerate(pair_wins.items()):
            if i >= 5: break
            print(" ", pair, dict(cnt))

    print("\nOK: basic CSV checks passed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_arena_csv.py path/to/<exp>_games.csv")
        sys.exit(2)
    main(sys.argv[1])
