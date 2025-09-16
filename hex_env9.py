# hex_env9.py
from typing import List, Tuple, Optional
import numpy as np

Move = Tuple[int, int]

try:
    from hexfast import evaluate_hex as _eval
except Exception:
    from hex_evaluator import evaluate_hex as _eval  # fallback

class HexEnv9:
    SIZE = 9

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
        n = self.SIZE
        return [(r, c) for r in range(n) for c in range(n) if (r, c) not in occ]

    def step(self, move: Move):
        if self.done:
            raise ValueError("Episode finished.")
        r, c = move
        if move in set(self.moves) or not (0 <= r < self.SIZE and 0 <= c < self.SIZE):
            self.done = True
            self.winner = 'White' if self.current_player() == 'Black' else 'Black'
            return self._obs(), -1.0, True, {}
        self.moves.append(move)
        self.winner, _ = _eval(self.SIZE, self.moves)
        if self.winner is not None or len(self.moves) == self.SIZE * self.SIZE:
            self.done = True
            reward = 1.0  # for the player who just moved
        else:
            reward = 0.0
        return self._obs(), reward, self.done, {}

    def _obs(self) -> np.ndarray:
        """3×9×9 float32: [black, white, to_move_is_black]"""
        n = self.SIZE
        b = np.zeros((3, n, n), dtype=np.float32)
        for i, (r, c) in enumerate(self.moves, start=1):
            if i % 2 == 1: b[0, r, c] = 1.0
            else:          b[1, r, c] = 1.0
        if self.current_player() == 'Black':
            b[2, :, :] = 1.0
        return b
