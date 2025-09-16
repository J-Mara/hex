# hex_env.py
from typing import List, Tuple, Optional
import numpy as np

Move = Tuple[int, int]

# Prefer the C accelerator if present
try:
    from hexfast import evaluate_hex as _eval
except Exception:
    from hex_evaluator import evaluate_hex as _eval  # your existing Python version

class HexEnv5:
    SIZE = 5
    def __init__(self):
        self.moves: List[Move] = []
        self.done = False
        self.winner: Optional[str] = None

    def reset(self):
        self.moves.clear()
        self.done = False
        self.winner = None
        return self._obs()

    def current_player(self) -> str:
        return 'Black' if (len(self.moves) % 2 == 0) else 'White'

    def legal_moves(self) -> List[Move]:
        occ = set(self.moves)
        return [(r, c) for r in range(self.SIZE) for c in range(self.SIZE) if (r, c) not in occ]

    def step(self, move: Move):
        if self.done:
            raise ValueError("Episode finished.")
        if move in set(self.moves):
            # illegal: immediate loss for side to move
            self.done = True
            self.winner = 'White' if self.current_player() == 'Black' else 'Black'
            return self._obs(), -1.0, True, {}
        r, c = move
        if not (0 <= r < self.SIZE and 0 <= c < self.SIZE):
            self.done = True
            self.winner = 'White' if self.current_player() == 'Black' else 'Black'
            return self._obs(), -1.0, True, {}
        self.moves.append(move)
        self.winner, _ = _eval(self.SIZE, self.moves)
        if self.winner is not None or len(self.moves) == self.SIZE * self.SIZE:
            self.done = True
            reward = 1.0  # reward is for the player who JUST moved
        else:
            reward = 0.0
        return self._obs(), reward, self.done, {}

    def _obs(self) -> np.ndarray:
        """3×5×5 float32: [black, white, to_move_is_black]"""
        n = self.SIZE
        b = np.zeros((3, n, n), dtype=np.float32)
        for i, (r, c) in enumerate(self.moves, start=1):
            if i % 2 == 1: b[0, r, c] = 1.0  # Black stones
            else:          b[1, r, c] = 1.0  # White stones
        if self.current_player() == 'Black':
            b[2, :, :] = 1.0
        return b
