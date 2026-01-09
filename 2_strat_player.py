#MCTS with UCT for Hex, with simple triangle-based pruning of obviously bad edge moves.

#NOTES TO IMPROVE:
#I think I can clean up the filtering code and make it faster.

from __future__ import annotations
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Set
from hexfast import evaluate_hex

Move = Tuple[int, int]

# Hex neighbors in (row, col)
NEIGHBORS: List[Tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
]

@dataclass
class MCTSStats:
    board_size: int
    starting_moves: int
    root_player: str  # "Black" or "White"
    iterations_requested: int
    iterations_performed: int
    time_limit: float
    duration: float  # seconds
    nodes_created: int  # total nodes in the search tree
    nodes_expanded: int  # how many times we expanded a node
    max_depth_reached: int  # max tree depth (in plies) from root

_last_stats: Optional[MCTSStats] = None

def get_last_stats() -> Optional[MCTSStats]:
    """
    Return statistics from the last call to choose_move, or None if
    choose_move has not been called yet.
    """
    return _last_stats

def _neighbors(r: int, c: int, size: int) -> Iterable[Move]:
    """Yield all valid hex neighbors of (r, c)."""
    for dr, dc in NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc

def _touches_player_edge(pos: Move, size: int, player_is_black: bool) -> bool:
    """
    Return True if this position lies on one of the edges that matter
    for the given player.

    - Black connects TOP (row=0) to BOTTOM (row=size-1).
    - White connects LEFT (col=0) to RIGHT (col=size-1).
    """
    r, c = pos
    if player_is_black:
        return r == 0 or r == size - 1
    else:
        return c == 0 or c == size - 1

def _filter_bad_edge_triangle_moves(size: int, empties: Iterable[Move], moves: List[Move], player_is_black: bool) -> List[Move]:
    """
    Triangle-based pruning rules.

    Rule 1 (own-edge empty triangle):
        If there is a triangle of three adjacent empty spaces, two of which
        are touching an edge of the player whose turn it is, then that player
        should not play in either of those two edge spaces.

    Rule 2 (opponent-edge dominated triangle):
        You should never play a move on the opponent's border if it is part
        of a triangle where:
          - one other space is also on the opponent's border and is either
            empty or occupied by the opponent, and
          - the third space (not on that border) is occupied by the opponent.

        However, if all legal moves would be pruned away by these rules, we
        fall back to allowing all empties.

    This function takes the set of empty cells and returns a list of allowed
    moves with those "obviously bad" edge cells removed, unless that would
    remove all moves.
    """
    empties_list: List[Move] = list(empties)
    empties_set: Set[Move] = set(empties_list)
    bad: Set[Move] = set()

    # Build a simple board from moves so we know who occupies what.
    # Board entries: None, "Black", or "White".
    board: List[List[Optional[str]]] = [
        [None for _ in range(size)] for _ in range(size)
    ]
    for idx, (r, c) in enumerate(moves):
        board[r][c] = "Black" if idx % 2 == 0 else "White"

    player_name = "Black" if player_is_black else "White"
    opp_name = "White" if player_is_black else "Black"

    # --- Rule 1: own-edge empty triangles -----------------------------------

    # Only cells that touch the relevant edges can be forbidden by this rule.
    edge_cells = [pos for pos in empties_list if _touches_player_edge(pos, size, player_is_black)]

    for e1 in edge_cells:
        if e1 in bad:
            continue
        r1, c1 = e1

        # Look for an adjacent edge cell e2 (also empty)
        for dr, dc in NEIGHBORS:
            r2, c2 = r1 + dr, c1 + dc
            e2 = (r2, c2)
            if e2 not in empties_set:
                continue
            if not _touches_player_edge(e2, size, player_is_black):
                continue

            # We have two adjacent empty edge cells e1 and e2.
            # Now see if there's a third empty cell t that is adjacent to both.
            neighbors_e1 = list(_neighbors(r1, c1, size))
            neighbors_e2 = set(_neighbors(r2, c2, size))

            found_triangle = False
            for t in neighbors_e1:
                if t in empties_set and t in neighbors_e2:
                    # Three empty cells, each adjacent to the others:
                    # e1, e2 (both on player's edge) and t.
                    found_triangle = True
                    break

            if found_triangle:
                bad.add(e1)
                bad.add(e2)
                # No need to keep searching for other triangles with e1/e2
                # for rule 1; continue with next edge cell.

    # --- Rule 2: opponent-edge dominated triangle ---------------------------

    # Opponent's "player edge"
    def touches_opponent_edge(pos: Move) -> bool:
        return _touches_player_edge(pos, size, not player_is_black)

    # Candidate cells on opponent's border (must be empty, because in empties_list)
    opp_edge_cells = [pos for pos in empties_list if touches_opponent_edge(pos)]

    for e1 in opp_edge_cells:
        if e1 in bad:
            continue
        r1, c1 = e1

        # Look for another border cell e2 (empty or occupied by opponent)
        for dr, dc in NEIGHBORS:
            r2, c2 = r1 + dr, c1 + dc
            if not (0 <= r2 < size and 0 <= c2 < size):
                continue
            e2 = (r2, c2)

            if not touches_opponent_edge(e2):
                continue

            # e2 must be either empty or occupied by opponent.
            if e2 in empties_set:
                e2_ok = True
            else:
                # occupied: only OK if by the opponent
                e2_ok = (board[r2][c2] == opp_name)
            if not e2_ok:
                continue

            # Now see if there's a third cell t that:
            #  - is adjacent to both e1 and e2,
            #  - is NOT on the opponent's border,
            #  - is occupied by the opponent.
            neighbors_e1 = list(_neighbors(r1, c1, size))
            neighbors_e2 = set(_neighbors(r2, c2, size))

            found_triangle = False
            for tr, tc in neighbors_e1:
                t = (tr, tc)
                if t not in neighbors_e2:
                    continue
                if touches_opponent_edge(t):
                    continue  # must be the non-border vertex
                if board[tr][tc] != opp_name:
                    continue  # must be occupied by opponent
                found_triangle = True
                break

            if found_triangle:
                # e1 is always empty (candidate), so we can mark it bad.
                bad.add(e1)
                # If e2 is also empty, it too is a bad candidate move.
                if e2 in empties_set:
                    bad.add(e2)
                # Done with this e1 once we've found one such triangle.
                break

    # Apply pruning, but respect the caveat: don't prune away all moves.
    allowed_moves = [pos for pos in empties_list if pos not in bad]
    if not allowed_moves and empties_list:
        # If pruning rules eliminate all legal moves, fall back to allowing all.
        allowed_moves = empties_list

    return allowed_moves

class MCTSNode:
    """
    A single node in the MCTS tree.

    - `moves` is the full history leading to this position.
    - `move` is the move that led from parent to this node (None for root).
    - `depth` is the number of moves from the root position.
    """

    def __init__(
        self,
        size: int,
        moves: List[Move],
        all_positions: Iterable[Move],
        parent: Optional["MCTSNode"] = None,
        move: Optional[Move] = None,
    ):
        self.size = size
        self.moves = moves
        self.parent = parent
        self.move = move

        # Which player is to move at this node?
        # Black moves on turns 0,2,4,...; White on 1,3,5,...
        self.player_is_black = (len(moves) % 2 == 0)

        # Tree statistics
        self.wins: float = 0.0
        self.visits: int = 0

        # Depth in the tree
        self.depth: int = 0 if parent is None else parent.depth + 1

        # Terminal state?
        winner, _ = evaluate_hex(size, moves)
        self.is_terminal: bool = winner is not None
        self.winner: Optional[str] = winner  # "Black" / "White" / None

        # Legal moves still untried from this node
        if self.is_terminal:
            self.untried_moves: List[Move] = []
        else:
            occupied = set(moves)
            all_empties = [pos for pos in all_positions if pos not in occupied]

            # Apply triangle-based pruning: remove obviously bad edge moves.
            self.untried_moves = _filter_bad_edge_triangle_moves(
                size=size,
                empties=all_empties,
                moves=moves,
                player_is_black=self.player_is_black,
            )

        self.children: List["MCTSNode"] = []

    def best_child(self, c_param: float = math.sqrt(2.0)) -> "MCTSNode":
        """
        Select child with highest UCT score.
        Assumes all children have visits > 0.
        """
        assert self.children, "best_child called on node with no children"
        parent_log = math.log(self.visits)

        def uct_score(child: "MCTSNode") -> float:
            exploit = child.wins / child.visits
            explore = c_param * math.sqrt(parent_log / child.visits)
            return exploit + explore

        return max(self.children, key=uct_score)

def _reward_for_winner(winner: Optional[str], root_player_is_black: bool) -> float:
    """
    Convert a game result into a reward from the root player's perspective.

    - 1.0 for a win for the root player
    - 0.0 for a loss
    - 0.5 for a draw/unknown (should basically never happen in Hex)
    """
    if winner is None:
        return 0.5
    if winner == "Black":
        return 1.0 if root_player_is_black else 0.0
    if winner == "White":
        return 0.0 if root_player_is_black else 1.0
    # Unexpected string; treat as draw
    return 0.5

def _rollout(size: int, start_moves: List[Move], all_positions: Iterable[Move], root_player_is_black: bool, rng) -> float:
    """
    Fast random playout from the given state:

    - Take the current moves.
    - Compute the remaining empty positions.
    - Shuffle the remaining positions once.
    - Append them all to form a complete random game.
    - Call evaluate_hex ONCE on that full game.

    We rely on the fact that evaluate_hex:
    - Returns the correct winner even if extra moves are played after the
      actual winning move.
    """
    moves = list(start_moves)
    used = set(moves)
    available = [pos for pos in all_positions if pos not in used]

    # If there are no available moves left, just evaluate the current position.
    if not available:
        winner, _ = evaluate_hex(size, moves)
        return _reward_for_winner(winner, root_player_is_black)

    rng.shuffle(available)
    full_game = moves + available

    winner, _ = evaluate_hex(size, full_game)
    return _reward_for_winner(winner, root_player_is_black)

def choose_move(size: int, moves: List[Move], rng: Optional[random.Random] = None, iterations: int = 1000000, time_limit: float = 1.0) -> Move:
    """
    Choose the next move for the current player using plain MCTS with UCT,
    but with triangle-based pruning of obviously bad edge moves.

    Signature is compatible with your arena:
        choose_move(size, moves, rng)

    You can also call it manually with extra control:
        choose_move(size, moves, rng, iterations=500, time_limit=0.5)

    Args:
        size: board size.
        moves: list of moves played so far (0-based (row, col)), Black first.
        rng: random.Random instance used for all randomness (for reproducibility).
             If None, uses the global `random` module.
        iterations: maximum number of MCTS iterations.
        time_limit: soft time limit in seconds for this move. If <= 0, time
                    is ignored and only `iterations` matters.

    Returns:
        The chosen move as a (row, col) tuple.
    """
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer")

    # Decide which player is to move at the root.
    # Black moves on turns 0,2,4,...; White on 1,3,5,...
    root_player_is_black = (len(moves) % 2 == 0)
    root_player = "Black" if root_player_is_black else "White"

    # Random source (either passed-in RNG or the random module)
    if rng is None:
        rng = random

    # Precompute all board positions once
    all_positions: List[Move] = [
        (r, c) for r in range(size) for c in range(size)
    ]

    # Build root node
    root = MCTSNode(size=size, moves=list(moves), all_positions=all_positions)

    # Game should not be over when arena calls us, but just in case:
    if root.is_terminal:
        raise ValueError("Cannot choose a move: the game is already over.")

    # Stats
    nodes_created = 1  # root
    nodes_expanded = 0
    max_depth = 0

    start_time = time.perf_counter()
    deadline = start_time + time_limit if time_limit > 0 else float("inf")
    iterations_done = 0

    # MCTS main loop
    for _ in range(iterations):
        if time.perf_counter() >= deadline:
            #print("2 strat time limit hit")
            break

        node = root

        # SELECTION: follow UCT until we hit a node we can expand or a terminal node.
        while (
            not node.is_terminal
            and not node.untried_moves
            and node.children
        ):
            node = node.best_child()

        # EXPANSION: if this node is non-terminal and has untried moves, expand one.
        if (not node.is_terminal) and node.untried_moves:
            # Here we just pick a random untried move (no heuristic bias),
            # but the list itself has already been pruned by the triangle rules.
            idx = rng.randrange(len(node.untried_moves))
            move = node.untried_moves.pop(idx)

            new_moves = list(node.moves)
            new_moves.append(move)

            child = MCTSNode(
                size=size,
                moves=new_moves,
                all_positions=all_positions,
                parent=node,
                move=move,
            )
            node.children.append(child)

            node = child
            nodes_created += 1
            nodes_expanded += 1
            if node.depth > max_depth:
                max_depth = node.depth

        # SIMULATION or TERMINAL REWARD
        if node.is_terminal:
            # We already know the winner at this node.
            reward = _reward_for_winner(node.winner, root_player_is_black)
        else:
            reward = _rollout(
                size=size,
                start_moves=node.moves,
                all_positions=all_positions,
                root_player_is_black=root_player_is_black,
                rng=rng,
            )

        # BACKPROPAGATION
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

        iterations_done += 1

    duration = time.perf_counter() - start_time

    # Choose the child of the root with the highest visit count.
    if not root.children:
        # No legal moves available (should only happen on a full board).
        raise ValueError("No legal moves available.")

    best_child = max(root.children, key=lambda c: c.visits)
    best_move = best_child.move  # type: ignore[assignment]

    # Record stats for this search
    global _last_stats
    _last_stats = MCTSStats(
        board_size=size,
        starting_moves=len(moves),
        root_player=root_player,
        iterations_requested=iterations,
        iterations_performed=iterations_done,
        time_limit=time_limit,
        duration=duration,
        nodes_created=nodes_created,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth,
    )

    return best_move
