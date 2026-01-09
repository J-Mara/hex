#MCTS with UCT for Hex, using a "risk-of-not-playing" heuristic in expansion.

#NOTES TO IMPROVE: Improve efficiency by finding a shortest path from the
#current state (not just the length of the path). Everything not in this path
#trivially has no risk of increasing the shortest path length.
#Also, try to save the ordering of possible moves like in 1

from __future__ import annotations
import math
import time
import random
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
from hexfast import evaluate_hex

Move = Tuple[int, int]
EMPTY = 0
BLACK = 1
WHITE = 2

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

# --- Hex geometry helpers ----------------------------------------------------

NEIGHBORS: List[Tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
]

def _neighbors(r: int, c: int, size: int) -> Iterable[Move]:
    """Yield valid hex neighbors of (r, c)."""
    for dr, dc in NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc

def _build_board(size: int, moves: List[Move]) -> List[List[int]]:
    """
    Build a board representation from the move history.

    0 = EMPTY, 1 = BLACK, 2 = WHITE.
    Black plays first, then White, etc.
    """
    board = [[EMPTY for _ in range(size)] for _ in range(size)]
    for i, (r, c) in enumerate(moves):
        is_black_turn = (i % 2 == 0)
        board[r][c] = BLACK if is_black_turn else WHITE
    return board

def _cell_state_for_player(board: List[List[int]], r: int, c: int, player_is_black: bool, blocked_move: Optional[Move]) -> str:
    """
    Classify a cell from the perspective of a given player, assuming that
    `blocked_move` (if not None) is occupied by the opponent.

    Returns: "us", "opp", or "empty".
    """
    if blocked_move is not None and (r, c) == blocked_move:
        return "opp"

    val = board[r][c]
    if player_is_black:
        if val == BLACK:
            return "us"
        elif val == WHITE:
            return "opp"
    else:
        if val == WHITE:
            return "us"
        elif val == BLACK:
            return "opp"
    return "empty"

def _shortest_connection_cost(size: int, board: List[List[int]], player_is_black: bool, blocked_move: Optional[Move]) -> int:
    """
    Heuristic: minimal number of *additional* stones this player needs
    (ignoring future opponent moves) to complete a winning connection.

    Implementation: Dijkstra on the hex graph with cell costs:
        cost = 0 for own stones
        cost = 1 for empty cells
        cost = INF (blocked) for opponent stones

    For Black: connect TOP (row=0) to BOTTOM (row=size-1).
    For White: connect LEFT (col=0) to RIGHT (col=size-1).

    We treat `blocked_move` as if it were already a stone of the opponent.
    """
    INF = size * size + 10
    dist = [[INF for _ in range(size)] for _ in range(size)]
    heap: List[Tuple[int, int, int]] = []

    if player_is_black:
        # Start from top edge (row=0), goal is any cell on bottom edge (row=size-1)
        start_row = 0
        for c in range(size):
            state = _cell_state_for_player(board, start_row, c, player_is_black, blocked_move)
            if state == "opp":
                continue  # cannot pass opponent stones
            cost = 0 if state == "us" else 1
            dist[start_row][c] = cost
            heapq.heappush(heap, (cost, start_row, c))

        while heap:
            d, r, c = heapq.heappop(heap)
            if d != dist[r][c]:
                continue
            if r == size - 1:
                return d  # reached bottom edge

            for nr, nc in _neighbors(r, c, size):
                state = _cell_state_for_player(board, nr, nc, player_is_black, blocked_move)
                if state == "opp":
                    continue
                step_cost = 0 if state == "us" else 1
                nd = d + step_cost
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

        return INF

    else:
        # White: connect left (col=0) to right (col=size-1)
        start_col = 0
        for r in range(size):
            state = _cell_state_for_player(board, r, start_col, player_is_black, blocked_move)
            if state == "opp":
                continue
            cost = 0 if state == "us" else 1
            dist[r][start_col] = cost
            heapq.heappush(heap, (cost, r, start_col))

        while heap:
            d, r, c = heapq.heappop(heap)
            if d != dist[r][c]:
                continue
            if c == size - 1:
                return d  # reached right edge

            for nr, nc in _neighbors(r, c, size):
                state = _cell_state_for_player(board, nr, nc, player_is_black, blocked_move)
                if state == "opp":
                    continue
                step_cost = 0 if state == "us" else 1
                nd = d + step_cost
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

        return INF

# --- MCTS core ---------------------------------------------------------------

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
            self.untried_moves = [pos for pos in all_positions if pos not in occupied]

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

# --- Top-level MCTS with risk-based expansion --------------------------------

def choose_move(size: int, moves: List[Move], rng: Optional[random.Random] = None, iterations: int = 1000000, time_limit: float = 0.5) -> Move:
    """
    Choose the next move for the current player using MCTS with UCT,
    where the expansion step is guided by a "risk-of-not-playing" heuristic.

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
            #print("5 strat time limit hit")
            break

        node = root

        # SELECTION: follow UCT until we hit a node we can expand or a terminal node.
        while (
            not node.is_terminal
            and not node.untried_moves
            and node.children
        ):
            node = node.best_child()

        # EXPANSION: if this node is non-terminal and has untried moves, expand
        # the move whose hypothetical opponent play there would hurt us the most.
        if (not node.is_terminal) and node.untried_moves:
            player_is_black = (len(node.moves) % 2 == 0)
            board = _build_board(size, node.moves)

            # Base shortest connection cost for the player to move.
            base_cost = _shortest_connection_cost(
                size=size,
                board=board,
                player_is_black=player_is_black,
                blocked_move=None,
            )

            best_moves: List[Move] = []
            best_risk = float("-inf")

            for m in node.untried_moves:
                # Imagine the OPPONENT plays at m: treat m as blocked/opp stone.
                blocked_cost = _shortest_connection_cost(
                    size=size,
                    board=board,
                    player_is_black=player_is_black,
                    blocked_move=m,
                )
                risk = blocked_cost - base_cost

                if risk > best_risk:
                    best_risk = risk
                    best_moves = [m]
                elif risk == best_risk:
                    best_moves.append(m)

            # If all risks are equal (including possibly negative),
            # best_moves will just be all moves with max risk value.
            move = rng.choice(best_moves)

            # Remove chosen move from untried_moves
            node.untried_moves.remove(move)

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
