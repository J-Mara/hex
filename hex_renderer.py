# hex_renderer.py
#
# Public, black-box API:
#   render_hex_game(size: int, moves: list[tuple[int,int]],
#                   output: str | None = None, show: bool = False,
#                   annotate_order: bool = True) -> None
#
# Example:
#   from hex_renderer import render_hex_game
#   render_hex_game(7, [(0,3),(0,2),(1,3), ...], output="game.png")
#
# Notes:
#   - 0-based coordinates (row, col)
#   - Black = odd move numbers, connects Top↔Bottom
#   - White = even move numbers, connects Left↔Right
#   - No dependency on your evaluator; it only draws.

from __future__ import annotations

import math
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle

Move = Tuple[int, int]

# ------------- geometry helpers (pointy-top hex layout) ----------------

def _hex_centers(size: int, R: float = 1.0) -> Dict[Move, Tuple[float, float]]:
    """Map (row, col) -> (x, y) center coordinates for an n×n rhombus of hex cells."""
    w = math.sqrt(3.0) * R     # horizontal spacing
    v = 1.5 * R                # vertical spacing
    centers: Dict[Move, Tuple[float, float]] = {}
    for r in range(size):
        for c in range(size):
            #x = c * w + r * (w / 2.0)
            #y = r * v
            x = c * w + r * (w / 2.0)
            y = -1 * r * v
            centers[(r, c)] = (x, y)
    return centers

def _board_bounds(size: int, R: float = 1.0, pad: float = 1.4) -> Tuple[float, float, float, float]:
    centers = _hex_centers(size, R)
    xs = [p[0] for p in centers.values()]
    ys = [p[1] for p in centers.values()]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

# --------------------- PUBLIC API ---------------------

def render_hex_game(
    size: int,
    moves: List[Move],
    *,
    output: str | None = None,
    show: bool = False,
    annotate_order: bool = True,
    highlight_last: bool = True,
    scale: float = 1.0,
    dpi: int = 300,
) -> None:
    """
    Render a Hex position.

    Parameters
    ----------
    size : int
        Board dimension n (n×n).
    moves : list[(row, col)]
        Move list (0-based). Must be legal (in-bounds, no duplicates).
    output : str | None
        If provided, save the image to this path (e.g., 'board.png', '.pdf', '.svg').
    show : bool
        If True, show an interactive window.
    annotate_order : bool
        If True, label each stone with its move number (1..len(moves)).
    highlight_last : bool
        If True, draw a ring around the final move.
    scale : float
        Visual scale (1.0 = default size). Increase for larger tiles.
    dpi : int
        Resolution when saving.

    Returns
    -------
    None
    """
    # ---- validate input (black-box friendly) ----
    if size <= 0:
        raise ValueError("size must be positive")
    seen = set()
    for i, (r, c) in enumerate(moves, start=1):
        if not (0 <= r < size and 0 <= c < size):
            raise ValueError(f"Move {i} out of bounds: {(r, c)}")
        if (r, c) in seen:
            raise ValueError(f"Move {i} repeats an occupied cell: {(r, c)}")
        seen.add((r, c))

    # ---- layout parameters ----
    R = 1.0 * scale
    centers = _hex_centers(size, R)
    xmin, xmax, ymin, ymax = _board_bounds(size, R)

    # ---- canvas ----
    # figsize tuned for readability; scale influences tile size
    fig, ax = plt.subplots(figsize=(6*scale, 6*scale))
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')

    # ---- base cells ----
    for (r, c), (x, y) in centers.items():
        tile = RegularPolygon(
            (x, y), numVertices=6, radius=R, orientation=0.0,
            facecolor="#f2f2f2", edgecolor="#888888", linewidth=1.0
        )
        ax.add_patch(tile)

    # subtle side hints (optional; not intrusive)
    ax.text((xmin + xmax) / 2, ymax + 0.25*scale, "White: Left ↔ Right",
            ha='center', va='bottom', fontsize=9*scale)
    ax.text(xmin - 0.25*scale, (ymin + ymax) / 2, "Black: Top ↔ Bottom",
            ha='right', va='center', rotation=90, fontsize=9*scale)

    # ---- stones ----
    stone_r = 0.58 * R
    for i, (r, c) in enumerate(moves, start=1):
        x, y = centers[(r, c)]
        is_black = (i % 2 == 1)
        face = 'black' if is_black else 'white'
        edge = 'white' if is_black else 'black'
        stone = Circle((x, y), radius=stone_r, facecolor=face, edgecolor=edge, linewidth=1.5)
        ax.add_patch(stone)

        if annotate_order:
            ax.text(
                x, y, str(i),
                color=('white' if is_black else 'black'),
                ha='center', va='center',
                fontsize=max(7, int(9*scale)), weight='bold'
            )

    if highlight_last and moves:
        rL, cL = moves[-1]
        xL, yL = centers[(rL, cL)]
        ring = Circle((xL, yL), radius=stone_r * 1.15,
                      facecolor=(0, 0, 0, 0), edgecolor='#2b8cbe', linewidth=2.0)
        ax.add_patch(ring)

    # small coordinate ticks (top row cols, left col rows)
    for c in range(size):
        xt, yt = centers[(0, c)]
        ax.text(xt, yt + (R * 0.95), f"{c}", ha='center', va='bottom', fontsize=7*scale, color='#555555')
    for r in range(size):
        xl, yl = centers[(r, 0)]
        ax.text(xl - (math.sqrt(3)*R*0.65), yl, f"{r}", ha='right', va='center', fontsize=7*scale, color='#555555')

    if output:
        plt.savefig(output, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
