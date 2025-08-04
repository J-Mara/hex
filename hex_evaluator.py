from typing import List, Optional, Tuple

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        #print(n)
        #print(self.parent)
        #print(self.rank)
        #print('\n')

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class HexEvaluator:
    """
    Evaluate a game of Hex given board size and a sequence of moves.
    Each move is a (row, col) pair using 0-based coordinates.
    """
    # Neighbor deltas for a rhombus-shaped Hex board using (row, col)
    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("Board size must be a positive integer.")
        self.n = size
        self.board = [[None] * size for _ in range(size)]  # 'B'/'W'/None

        # DSUs for each color (with their own virtual nodes)
        self.cells = self.n * self.n
        # Black tries Top↔Bottom; add two virtual nodes for top and bottom
        self.black = DSU(self.cells + 2)
        self.BLK_TOP = self.cells
        self.BLK_BOT = self.cells + 1

        # White tries Left↔Right; add two virtual nodes for left and right
        self.white = DSU(self.cells + 2)
        self.WHT_LEFT = self.cells
        self.WHT_RIGHT = self.cells + 1

    def _idx(self, r: int, c: int) -> int:
        return r * self.n + c

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def _place_and_connect(self, r: int, c: int, color: str) -> None:
        self.board[r][c] = color
        idx = self._idx(r, c)

        if color == 'B':
            # Connect to virtual edges if on top/bottom rows
            if r == 0:
                self.black.union(idx, self.BLK_TOP)
            if r == self.n - 1:
                self.black.union(idx, self.BLK_BOT)

            # Union with adjacent Black stones
            for dr, dc in self.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and self.board[nr][nc] == 'B':
                    self.black.union(idx, self._idx(nr, nc))

        else:  # color == 'W'
            # Connect to virtual edges if on left/right columns
            if c == 0:
                self.white.union(idx, self.WHT_LEFT)
            if c == self.n - 1:
                self.white.union(idx, self.WHT_RIGHT)

            # Union with adjacent White stones
            for dr, dc in self.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and self.board[nr][nc] == 'W':
                    self.white.union(idx, self._idx(nr, nc))

    def play(self, moves: List[Tuple[int, int]]) -> Tuple[Optional[str], Optional[int]]:
        """
        Plays moves in order. Returns:
          (winner, move_index)
          - winner: 'Black', 'White', or None if no one has won yet
          - move_index: 1-based move number at which the win occurred, or None
        Raises ValueError on illegal inputs (out-of-bounds or repeated move).
        """
        for i, (r, c) in enumerate(moves, start=1):
            if not self._in_bounds(r, c):
                raise ValueError(f"Move {i} out of bounds: {(r, c)}")
            if self.board[r][c] is not None:
                raise ValueError(f"Move {i} repeats an occupied cell: {(r, c)}")

            color = 'B' if i % 2 == 1 else 'W'  # Black starts
            self._place_and_connect(r, c, color)

            # Check for win after this move
            if color == 'B':
                if self.black.find(self.BLK_TOP) == self.black.find(self.BLK_BOT):
                    return ("Black", i)
            else:
                if self.white.find(self.WHT_LEFT) == self.white.find(self.WHT_RIGHT):
                    return ("White", i)

        return (None, None)


def evaluate_hex(size: int, moves: List[Tuple[int, int]]) -> Tuple[Optional[str], Optional[int]]:
    """
    Convenience function: returns ('Black' or 'White' or None, move_number or None)
    """
    return HexEvaluator(size).play(moves)


if __name__ == "__main__":
    # Simple CLI demo: run `python hex_evaluator.py`
    # Example 1: Black wins on a 3x3
    example1 = [(0,1), (0,0), (1,1), (1,0), (2,1)]
    print("Example 1 (3x3):", evaluate_hex(3, example1))

    # Example 2: White wins on a 3x3
    #example2 = [(0,1), (1,0), (2,1), (1,1), (0,2), (1,2)]
    #print("Example 2 (3x3):", evaluate_hex(3, example2))
