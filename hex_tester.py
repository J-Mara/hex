# hex_tester.py

import random
import argparse
from typing import List, Tuple, Optional, Set, Iterable
from hex_evaluator import evaluate_hex

Coord = Tuple[int, int]


# -------------------------------
# Core hex utilities (agnostic)
# -------------------------------

NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

def in_bounds(n: int, r: int, c: int) -> bool:
    return 0 <= r < n and 0 <= c < n

def neighbors(n: int, r: int, c: int) -> Iterable[Coord]:
    for dr, dc in NEIGHBORS:
        nr, nc = r + dr, c + dc
        if in_bounds(n, nr, nc):
            yield (nr, nc)


# ----------------------------------------
# Independent verifier (BFS from scratch)
# ----------------------------------------

def earliest_win_bfs(n: int, moves: List[Coord]) -> Tuple[Optional[str], Optional[int]]:
    """
    Independently verifies the game by checking, after each move i,
    whether the player who just moved has formed a connecting path.
    Returns ('Black'/'White'/None, move_index or None).
    """
    board = [[None] * n for _ in range(n)]  # 'B','W', or None

    def has_connection(color: str) -> bool:
        from collections import deque

        visited: Set[Coord] = set()
        q = deque()

        if color == 'B':
            # start frontier: all Black stones on top edge
            for c in range(n):
                if board[0][c] == 'B':
                    q.append((0, c))
                    visited.add((0, c))
            # target: reach any bottom row
            target_row = n - 1
            while q:
                r, c = q.popleft()
                if r == target_row:
                    return True
                for nr, nc in neighbors(n, r, c):
                    if (nr, nc) not in visited and board[nr][nc] == 'B':
                        visited.add((nr, nc))
                        q.append((nr, nc))
            return False

        else:  # 'W'
            # start frontier: all White stones on left edge
            for r in range(n):
                if board[r][0] == 'W':
                    q.append((r, 0))
                    visited.add((r, 0))
            # target: reach any rightmost column
            target_col = n - 1
            while q:
                r, c = q.popleft()
                if c == target_col:
                    return True
                for nr, nc in neighbors(n, r, c):
                    if (nr, nc) not in visited and board[nr][nc] == 'W':
                        visited.add((nr, nc))
                        q.append((nr, nc))
            return False

    for i, (r, c) in enumerate(moves, start=1):
        color = 'B' if i % 2 == 1 else 'W'
        if not in_bounds(n, r, c) or board[r][c] is not None:
            raise ValueError(f"Invalid move {i}: {(r, c)}")
        board[r][c] = color

        # Check if the just-moved player has a connection
        if has_connection(color):
            return ('Black' if color == 'B' else 'White', i)

    return (None, None)


# -------------------------------------------------------
# Generators for different categories of test sequences
# -------------------------------------------------------

def all_cells(n: int) -> List[Coord]:
    return [(r, c) for r in range(n) for c in range(n)]

def random_unoccupied_cell(n: int, occupied: Set[Coord], rng: random.Random) -> Coord:
    # Efficient sampling by rejection on small boards; fine for testing
    while True:
        r = rng.randrange(n)
        c = rng.randrange(n)
        if (r, c) not in occupied:
            return (r, c)

def make_path(n: int, color: str, rng: random.Random) -> List[Coord]:
    """
    Create a *simple path* (no repeats) across the board for the requested color.
    This is built by randomized DFS over the empty grid graph.
    The path consists of distinct cells that connect the required edges.
    """
    from collections import deque

    def goal_reached(coord: Coord) -> bool:
        r, c = coord
        if color == 'B':
            return r == n - 1
        else:
            return c == n - 1

    # Choose a randomized start on the relevant starting edge
    if color == 'B':
        starts = [(0, c) for c in range(n)]
    else:
        starts = [(r, 0) for r in range(n)]
    rng.shuffle(starts)

    visited: Set[Coord] = set()
    parent = dict()

    # Try multiple randomized DFS attempts to find a path to the opposite edge
    for start in starts:
        stack = [start]
        visited = {start}
        parent = {start: None}
        while stack:
            cur = stack.pop()
            if goal_reached(cur):
                # Reconstruct path
                path = []
                x = cur
                while x is not None:
                    path.append(x)
                    x = parent[x]
                path.reverse()
                return path
            nbrs = list(neighbors(n, cur[0], cur[1]))
            rng.shuffle(nbrs)
            for nxt in nbrs:
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = cur
                    stack.append(nxt)

    # As the grid is connected, this should not happen; fallback to straight line
    if color == 'B':
        c = rng.randrange(n)
        return [(r, c) for r in range(n)]
    else:
        r = rng.randrange(n)
        return [(r, c) for c in range(n)]

def interleave_for_win(n: int, path: List[Coord], color: str, rng: random.Random) -> List[Coord]:
    """
    Construct a move sequence where the given color eventually wins by placing
    stones along 'path'. Opponent moves are placed elsewhere.
    """
    occupied: Set[Coord] = set()
    moves: List[Coord] = []

    # Order in which the path cells will be played by winner
    path_order = path[:]  # copy
    rng.shuffle(path_order)

    winner_turn_is_odd = (color == 'B')  # Black moves on odd turns

    # Keep placing moves until the independent verifier says winner has connected
    for k in range(1, n*n + 1):
        if (winner_turn_is_odd and k % 2 == 1) or ((not winner_turn_is_odd) and k % 2 == 0):
            # Winner's turn: take next unoccupied path cell (guaranteed available)
            while path_order and path_order[0] in occupied:
                path_order.pop(0)
            if not path_order:
                # If path consumed (shouldn’t happen often), pick any free cell; win should already occur soon
                m = random_unoccupied_cell(n, occupied, rng)
            else:
                m = path_order.pop(0)
        else:
            # Opponent's turn: avoid occupying winner's remaining path cells when possible
            opp_try = 0
            while opp_try < 20:
                m = random_unoccupied_cell(n, occupied, rng)
                if m not in path:  # try not to block the intended path
                    break
                opp_try += 1

        moves.append(m)
        occupied.add(m)

        # Stop as soon as winner actually wins (per independent verifier)
        w, idx = earliest_win_bfs(n, moves)
        if w == ('Black' if color == 'B' else 'White'):
            return moves[:idx]

    return moves  # fallback (shouldn't be needed)

def generate_black_win(n: int, rng: random.Random) -> List[Coord]:
    path = make_path(n, 'B', rng)
    return interleave_for_win(n, path, 'B', rng)

def generate_white_win(n: int, rng: random.Random) -> List[Coord]:
    path = make_path(n, 'W', rng)
    return interleave_for_win(n, path, 'W', rng)

def generate_no_winner(n: int, rng: random.Random, max_len: Optional[int] = None) -> List[Coord]:
    """
    Generate a random sequence that *does not* produce a winner (yet).
    Stops early if about to win; retries as needed. By default tries up to ~n*n//2 moves.
    """
    target_len = max_len if max_len is not None else max(3, ((n-1) * (n-1)))
    attempts = 0
    while True:
        attempts += 1
        occupied: Set[Coord] = set()
        moves: List[Coord] = []
        for _ in range(min(target_len, n * n)):
            m = random_unoccupied_cell(n, occupied, rng)
            moves.append(m)
            occupied.add(m)
            w, _ = earliest_win_bfs(n, moves)
            if w is not None:
                break  # abort this try; we want "no winner yet"
        else:
            # successfully built a sequence with no winner
            return moves
        if attempts > 200:
            # On small boards, it can be hard—relax by shortening
            target_len = max(2, target_len - 1)

def generate_random_moves(n: int, rng: random.Random, length: int) -> List[Coord]:
    """
    Generate exactly 'length' random legal moves (alternating colors), regardless of outcome.
    """
    occupied: Set[Coord] = set()
    moves: List[Coord] = []
    for _ in range(min(length, n * n)):
        m = random_unoccupied_cell(n, occupied, rng)
        moves.append(m)
        occupied.add(m)
    return moves


# -----------------------------------------
# Batch runner with cross-verification
# -----------------------------------------

def run_suite(
    n: int,
    count_black_win: int,
    count_white_win: int,
    count_no_winner: int,
    count_random: int,
    random_len: int,
    seed: Optional[int] = None,
) -> None:
    rng = random.Random(seed)
    total = 0
    mismatches = 0

    def check_case(label: str, moves: List[Coord]) -> None:
        nonlocal total, mismatches
        total += 1
        # Black-box result
        bb_w, bb_idx = evaluate_hex(n, moves)
        # Independent verifier result
        ref_w, ref_idx = earliest_win_bfs(n, moves)
        ok = (bb_w == ref_w) and (bb_idx == ref_idx)

        status = "OK" if ok else "MISMATCH"
        print(f"[{label}] {status}  moves={len(moves)}  blackbox={bb_w, bb_idx}  ref={ref_w, ref_idx}")
        if not ok:
            mismatches += 1
            # On mismatch, show the sequence for debugging
            print("  Moves:", moves)

    # Generate and check each requested category
    for _ in range(count_black_win):
        check_case("BlackWin", generate_black_win(n, rng))

    for _ in range(count_white_win):
        check_case("WhiteWin", generate_white_win(n, rng))

    for _ in range(count_no_winner):
        check_case("NoWinner", generate_no_winner(n, rng))

    for _ in range(count_random):
        seq = generate_random_moves(n, rng, random_len)
        check_case("Random", seq)

    print(f"\nSummary: {total} cases, {mismatches} mismatches.")
    if mismatches > 0:
        raise SystemExit(1)


# ------------------------
# CLI entry point
# ------------------------

def main():
    p = argparse.ArgumentParser(description="Hex test generator + independent verifier")
    p.add_argument("--size", type=int, default=5, help="Board size n (default: 5)")
    p.add_argument("--black-win", type=int, default=5, help="# Black-win cases (default: 5)")
    p.add_argument("--white-win", type=int, default=5, help="# White-win cases (default: 5)")
    p.add_argument("--no-winner", type=int, default=5, help="# No-winner-yet cases (default: 5)")
    p.add_argument("--random", type=int, default=5, help="# Random cases (default: 5)")
    p.add_argument("--random-len", type=int, default=10, help="Length for random cases (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    args = p.parse_args()

    run_suite(
        n=args.size,
        count_black_win=args.black_win,
        count_white_win=args.white_win,
        count_no_winner=args.no_winner,
        count_random=args.random,
        random_len=args.random_len,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()

# run with: python3 hex_tester.py --size 7 --black-win 10 --white-win 10 --no-winner 5 --random 5 --random-len 15 --seed 123


# # Black wins on 5th move
# size = 3
# moves = [(0,1), (0,0), (1,1), (1,0), (2,1)]
# print(evaluate_hex(size, moves))  # -> ('Black', 5)
#
# # White wins on 6th move (connects left→right)
# moves = [(0,1), (1,0), (2,1), (1,1), (0,2), (1,2)]
# print(evaluate_hex(size, moves))  # -> ('White', 6)
#
# # No winner yet
# moves = [(0,0), (2,2), (0,1)]
# print(evaluate_hex(size, moves))  # -> (None, None)
#
# # larger tests
# size = 5
# moves = [(2,2), (0,3), (1,1), (3,2), (3,1), (2,1), (1,2), (4,0), (4,1), (0,1), (0,2)]
# print(evaluate_hex(size, moves)) # -> ('Black', 11)
#
# # Black wins on 5th move
# size = 3
# moves = [(0,1), (0,0), (1,1), (1,0), (2,1), (2,0), (0,2)]
# print(evaluate_hex(size, moves))  # -> ('Black', 5)
