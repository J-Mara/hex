#MCTS with UCT for Hex, using a 'two-ahead' heuristic in rollouts.

#NOTES TO IMPROVE: right now I use two ahead random for rollout games, but I
#should get something better that is also hyper optimized. Maybe a neural net,
#maybe some c code, maybe random moves influenced by a hand crafted system of rules.

from __future__ import annotations
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
from hexfast import evaluate_hex, two_ahead_choose_move

Move = Tuple[int, int]

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
    'Two-ahead' rollout:

    Starting from start_moves, repeatedly:
      - check for a winner;
      - if no winner and board not full, let two_ahead_choose_move(size, moves, rng)
        choose the next move (for the current player);
      - append the move and continue.

    This uses two_ahead_choose_move for *both* players' moves.
    """
    moves = list(start_moves)
    max_moves = size * size

    while True:
        winner, _ = evaluate_hex(size, moves)
        if winner is not None:
            return _reward_for_winner(winner, root_player_is_black)

        if len(moves) == max_moves:
            # Full board, no winner: treat as draw (should be rare in Hex).
            return 0.5

        # Let the heuristic policy pick the next move for the current player.
        move = two_ahead_choose_move(size, moves, rng)
        # As a safety net, you could assert legality; we assume hexfast's
        # implementation always returns a legal move.
        moves.append(move)

def choose_move(size: int, moves: List[Move], rng: Optional[random.Random] = None, iterations: int = 1000000, time_limit: float = 1.0) -> Move:
    """
    Choose the next move for the current player using MCTS with UCT,
    where rollouts use two_ahead_choose_move for both players.

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
            #print("3 strat time limit hit")
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
