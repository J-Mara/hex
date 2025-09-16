# hex_env.py
#
# A *very* small OpenAI-Gym-like environment for 5x5 Hex.
# It exposes reset() and step(move) and keeps track of whose turn it is.
# Uses evaluate_hex from hex_evaluator (black-box).

from typing import List, Tuple, Optional
import numpy as np

from hex_evaluator import evaluate_hex

Move = Tuple[int, int]

class HexEnv:
    """
    A minimal reinforcement-learning environment for Hex.
    The *current* player always acts (so the agent can play either colour
    transparently). States are 3-plane 5×5 tensors:
        plane 0: 1 where Black stones, else 0
        plane 1: 1 where White stones, else 0
        plane 2: 1 everywhere if the current player is Black, 0 if White
    Reward: +1 for win, −1 for loss, 0 for every non-terminal move.
    Episode ends at win or full board.
    """
    SIZE = 5
    N_CELLS = SIZE * SIZE

    def __init__(self):
        self.moves: List[Move] = []
        self.done = False
        self.winner: Optional[str] = None

    # -------- Gym-style API ---------
    def reset(self):
        self.moves.clear()
        self.done = False
        self.winner = None
        return self._board_tensor()

    def legal_moves(self) -> List[Move]:
        occupied = set(self.moves)
        return [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)
                if (r, c) not in occupied]

    def step(self, move: Move):
        if self.done:
            raise ValueError("Game already finished")
        if move not in self.legal_moves():
            # illegal → immediate loss
            self.done = True
            self.winner = 'White' if self._current_player() == 'Black' else 'Black'
            reward = -1.0
            return self._board_tensor(), reward, self.done, {}
        self.moves.append(move)

        self.winner, _ = evaluate_hex(self.SIZE, self.moves)
        if self.winner is not None or len(self.moves) == self.N_CELLS:
            self.done = True
            reward = 1.0 if self.winner == self._current_player() else -1.0
        else:
            reward = 0.0
        return self._board_tensor(), reward, self.done, {}

    # -------- helpers --------
    def _current_player(self) -> str:
        return 'Black' if len(self.moves) % 2 == 0 else 'White'

    def _board_tensor(self) -> np.ndarray:
        """Return a 3×5×5 float32 numpy array."""
        b = np.zeros((3, self.SIZE, self.SIZE), dtype=np.float32)
        for i, (r, c) in enumerate(self.moves, start=1):
            plane = 0 if i % 2 == 1 else 1  # Black plane 0, White plane 1
            b[plane, r, c] = 1.0
        if self._current_player() == 'Black':
            b[2, :, :] = 1.0
        return b
