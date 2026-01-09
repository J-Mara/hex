# look_ahead_player.py
from __future__ import annotations
from typing import List, Tuple, Dict
import random

# Prefer the fast extension if available
try:
    from hexfast import evaluate_hex as _eval_hex_raw  # returns (winner, path_idx_or_none)
except Exception:
    from hex_evaluator import evaluate_hex as _eval_hex_raw  # may return winner or (winner, ...)

Move = Tuple[int, int]

# -------- small helpers --------

def _side_to_move(move_count: int) -> str:
    # Black plays first, then alternate
    return "Black" if (move_count % 2 == 0) else "White"

def _opponent(color: str) -> str:
    return "White" if color == "Black" else "Black"

def _legal_moves(size: int, moves: List[Move]) -> List[Move]:
    occupied = set(moves)
    return [(r, c) for r in range(size) for c in range(size) if (r, c) not in occupied]

def _bitboards(size: int, moves: List[Move]) -> Tuple[int, int]:
    """
    Order-independent board key: two bitboards (black_mask, white_mask).
    Bit index = r*size + c.
    """
    bmask = 0
    wmask = 0
    for i, (r, c) in enumerate(moves):
        bit = 1 << (r * size + c)
        if i % 2 == 0:
            bmask |= bit
        else:
            wmask |= bit
    return bmask, wmask

def _winner_only(size: int, moves: List[Move]) -> str | None:
    """
    Normalize evaluate_hex output to just 'Black'/'White'/None.
    Works for either (winner, extra) or winner-only implementations.
    """
    res = _eval_hex_raw(size, moves)
    if isinstance(res, tuple):
        return res[0]
    return res

# -------- core search: "mate-in-N" (forced win within depth full moves) --------

def _can_root_force_win_within(
    size: int,
    moves: List[Move],
    depth_full_moves: int,
    root_color: str,
    cache: Dict[Tuple[int, str, int, int, int], bool],
) -> bool:
    """
    Returns True iff the player to move at 'moves' (which must be root_color)
    can force a win within 'depth_full_moves' *full* moves.
    A full move = root plays, then opponent plays (if game not already over).
    """
    # Terminal check (corrected)
    winner = _winner_only(size, moves)
    if winner is not None:
        return (winner == root_color)

    # If we’ve exhausted allowed full moves and haven’t already won, we cannot *guarantee* a win
    if depth_full_moves <= 0:
        return False

    # Memoization key: (size, root_color, black_mask, white_mask, depth_full_moves)
    bmask, wmask = _bitboards(size, moves)
    key = (size, root_color, bmask, wmask, depth_full_moves)
    if key in cache:
        return cache[key]

    # Sanity: it must be root_color to move here
    to_move = _side_to_move(len(moves))
    if to_move != root_color:
        # This should not happen; protect against misuse
        cache[key] = False
        return False

    # Try each root move: if we find one that *guarantees* a win within depth, return True
    legals_root = _legal_moves(size, moves)
    for mv in legals_root:
        # Root plays
        s1 = moves + [mv]
        winner1 = _winner_only(size, s1)
        if winner1 == root_color:
            cache[key] = True
            return True  # immediate win

        # If not yet terminal, opponent replies. To be a *guarantee*, the root must still
        # be able to force a win within (depth_full_moves - 1) *for every* opponent reply.
        opp_legals = _legal_moves(size, s1)
        opp_color = _opponent(root_color)
        forced = True  # assume this root move is a forced win until we find a refutation

        if not opp_legals:
            # Hex should always have a winner on full board, but be defensive:
            forced = (winner1 == root_color)

        for omv in opp_legals:
            s2 = s1 + [omv]
            winner2 = _winner_only(size, s2)
            if winner2 == opp_color:
                forced = False  # opponent can immediately win → not guaranteed
                break
            # After the opponent moves, it's root's turn again with one fewer full move remaining.
            if not _can_root_force_win_within(size, s2, depth_full_moves - 1, root_color, cache):
                forced = False  # opponent found a reply that avoids forced win within the depth
                break

        if forced:
            cache[key] = True
            return True

    cache[key] = False
    return False

# -------- public API --------

def choose_move(size: int, moves: List[Move], rng: random.Random | None = None, depth: int = 5) -> Move:
    """
    Look-ahead Hex player.
    - Searches up to 'depth' full moves (our move then theirs) for a *forced* win.
    - If it finds one, plays the first move that guarantees a win within that depth.
    #- Otherwise, plays a random legal move.
    - Otherwise, it now tries to avoid handing the opponent a forced win within
      (depth - 1) full moves before falling back to a random move.

    Assumptions:
    - Black always plays first; moves alternate strictly.
    - evaluate_hex(size, moves) returns 'Black'/'White'/None or (winner, extra).
    """
    legals = _legal_moves(size, moves)
    if not legals:
        # No legal moves left; return a dummy coordinate to avoid crashing (shouldn't happen)
        return (0, 0)

    # if (size - (len(moves)/2) > depth):
    #     print("to early for informed search")
    #     return rng.choice(legals)

    if rng is None:
        rng = random.Random()

    if (size - (len(moves)/2) > depth):
        print("too early for informed search, playing random move without lookahead")
        return rng.choice(legals)

    winner = _winner_only(size, moves)
    if winner is not None:
        print(f"board already has a winner ({winner}); selecting random legal move")
        return rng.choice(legals)

    root_color = _side_to_move(len(moves))
    print(
        f"look_ahead_player: {root_color} to move with depth={depth} on size {size} after {len(moves)} moves"
    )

    shared_cache: Dict[Tuple[int, str, int, int, int], bool] = {}

    # 1) Immediate win check (quick short-circuit)
    rng.shuffle(legals)  # randomize move order so multiple winning lines don't bias earlier cells
    print(f"initial randomized move order for immediate-win scan: {legals}")
    for mv in legals:
        s1 = moves + [mv]
        if _winner_only(size, s1) == root_color:
            print("winning move")
            return mv  # snap up the instant win
    print("no immediate winning move available")

    # 2) Forced win search up to 'depth' full moves
    if depth > 0:
        #cache: Dict[Tuple[int, str, int, int, int], bool] = {}
        print(f"searching for guaranteed win within {depth} full moves")
        for mv in legals:
            s1 = moves + [mv]
            # If opponent has an immediate win reply, this move can't be a guaranteed win
            opp_legals = _legal_moves(size, s1)
            opp_color = _opponent(root_color)
            forced = True
            for omv in opp_legals:
                s2 = s1 + [omv]
                w2 = _winner_only(size, s2)
                if w2 == opp_color:
                    forced = False
                    break
                #if not _can_root_force_win_within(size, s2, depth - 1, root_color, cache):
                if not _can_root_force_win_within(size, s2, depth - 1, root_color, shared_cache):
                    forced = False
                    break
            if forced:
                print("found guaranteed win")
                return mv  # first guaranteed winning move found
        print("no guaranteed win found within requested depth")
    else:
        print(f"depth {depth} too small for forced-win search; skipping that stage")

    # 3) Defensive filter: avoid handing opponent a forced win within (depth - 1)
    if depth > 0:
        defensive_depth = depth - 1
        opp_color = _opponent(root_color)
        defensive_candidates = legals[:]
        rng.shuffle(defensive_candidates)
        print(
            "evaluating moves to avoid opponent forced wins within "
            f"{defensive_depth} full moves; randomized order: {defensive_candidates}"
        )

        for mv in defensive_candidates:
            s1 = moves + [mv]
            print(f"checking defensive safety of move {mv}")

            # First, guard against immediate tactical losses
            immediate_loss = False
            opp_legals = _legal_moves(size, s1)
            for omv in opp_legals:
                s2 = s1 + [omv]
                if _winner_only(size, s2) == opp_color:
                    immediate_loss = True
                    print(
                        f" -> opponent wins immediately by replying with {omv}; move {mv} is unsafe"
                    )
                    break

            if immediate_loss:
                continue

            opponent_can_force = False
            if defensive_depth > 0:
                opponent_can_force = _can_root_force_win_within(
                    size, s1, defensive_depth, opp_color, shared_cache
                )
                print(
                    " -> opponent forced-win status within depth-1 "
                    f"({defensive_depth}): {opponent_can_force}"
                )
            else:
                print(
                    " -> defensive depth is 0; cannot look beyond immediate replies,"
                    " treating move as safe if no instant loss"
                )

            if not opponent_can_force:
                print(f"choosing move {mv} which does not hand opponent a forced win")
                return mv
            print(f"move {mv} rejected; opponent can force a win within {defensive_depth} moves")

        print(
            "all candidate moves allow the opponent a forced win within the defensive depth;"
            " selecting a random move anyway"
        )
        fallback_move = rng.choice(defensive_candidates)
        print(f"risk-accepting fallback move: {fallback_move}")
        return fallback_move

    # # 3) Fallback: random legal
    # print("random move")
    # return rng.choice(legals)
    # 4) Final fallback when depth <= 0 prevented defensive filtering
    print("no depth available for defensive filtering; choosing random move")
    fallback_move = rng.choice(legals)
    print(f"random fallback move: {fallback_move}")
    return fallback_move

# -------- strict version of move choice ------

def choose_strict_move(size: int, moves: List[Move], rng: random.Random | None = None, depth: int = 5) -> Optional[Move]:
    """
    Strict Look-ahead Hex player.
    - Searches up to 'depth' full moves (our move then theirs) for a *forced* win.
    - If it finds one, plays the first move that guarantees a win within that depth.
    - Otherwise, returns None.

    Assumptions:
    - Black always plays first; moves alternate strictly.
    - evaluate_hex(size, moves) returns 'Black'/'White'/None or (winner, extra).
    """
    legals = _legal_moves(size, moves)
    if not legals:
        # No legal moves left; return None
        return None

    if rng is None:
        rng = random.Random()

    winner = _winner_only(size, moves)
    if winner is not None:
        print(f"board already has a winner ({winner}); returning None")
        return None

    root_color = _side_to_move(len(moves))
    # print(
    #     f"look_ahead_player: {root_color} to move with depth={depth} on size {size} after {len(moves)} moves"
    # )

    shared_cache: Dict[Tuple[int, str, int, int, int], bool] = {}
    #Forced win search up to 'depth' full moves
    if depth > 0:
        #cache: Dict[Tuple[int, str, int, int, int], bool] = {}
        #print(f"searching for guaranteed win within {depth} full moves")
        for mv in legals:
            s1 = moves + [mv]
            # If opponent has an immediate win reply, this move can't be a guaranteed win
            opp_legals = _legal_moves(size, s1)
            opp_color = _opponent(root_color)
            forced = True
            for omv in opp_legals:
                s2 = s1 + [omv]
                w2 = _winner_only(size, s2)
                if w2 == opp_color:
                    forced = False
                    break
                #if not _can_root_force_win_within(size, s2, depth - 1, root_color, cache):
                if not _can_root_force_win_within(size, s2, depth - 1, root_color, shared_cache):
                    forced = False
                    break
            if forced:
                print("found guaranteed win")
                return mv  # first guaranteed winning move found
        print("no guaranteed win found within requested depth")
    else:
        print(f"depth {depth} too small for forced-win search")
    return None

# Optional quick self-test
if __name__ == "__main__":
    import sys
    # Smoke test: on 2x2, Black can win immediately from empty position.
    mv = choose_move(2, [], depth=1)
    print("Suggested move on empty 2x2:", mv)

# on size 3: optimal on black at depth 3
# on size 4: optimal on black at depth 5
