#!/usr/bin/env python3
# hex_gui.py — simple Tkinter Hex GUI to play vs any bot module
# Dependencies: standard library (tkinter), your evaluator (hexfast or hex_evaluator)
# Usage examples:
#   python hex_gui.py --bot look_ahead_player --size 5 --human black
#   RL_WEIGHTS=best_model.npz python hex_gui.py --bot rl_player --size 5 --human white

from __future__ import annotations
import argparse
import importlib
import math
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import sys

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    print("Tkinter not available. On macOS, install Python from python.org or `brew install tcl-tk` "
          "and ensure Python links against it.", file=sys.stderr)
    raise

# --- evaluator: prefer C, else Python ---
try:
    from hexfast import evaluate_hex as evaluate_hex
except Exception:
    from hex_evaluator import evaluate_hex as evaluate_hex

Move = Tuple[int, int]

DEFAULT_BOT_CHOICES = [
    "rl_player",
    "two_ahead_random_player",
    "one_ahead_random_player",
    "random_player",
    # C-accelerated wrappers (if you created them)
    "two_ahead_random_player_c",
    "one_ahead_random_player_c",
    "random_player_c",
    "monte_carlo_player",
    "monte_carlo_player_c",
]

@dataclass
class BotRef:
    name: str
    module: object
    choose: object

# ---------------- Geometry helpers (pointy-top hex grid) -----------------

def hex_centers(size: int, R: float = 1.0) -> dict[Move, Tuple[float, float]]:
    """Return centers in axial layout (pointy-top), rhombus n×n."""
    w = math.sqrt(3.0) * R
    v = 1.5 * R
    centers: dict[Move, Tuple[float, float]] = {}
    for r in range(size):
        for c in range(size):
            x = c * w + r * (w / 2.0)
            y = r * v
            centers[(r, c)] = (x, y)
    return centers

def bounds_from_centers(centers: dict[Move, Tuple[float, float]], pad: float = 1.0):
    xs = [x for (x, y) in centers.values()]
    ys = [y for (x, y) in centers.values()]
    return min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad

def hex_corners(x: float, y: float, R: float) -> List[float]:
    """Return flat list [x1,y1,x2,y2,...] of the 6 polygon vertices (pointy-top)."""
    pts = []
    for k in range(6):
        ang = math.radians(60*k - 30)  # pointy
        pts.extend([x + R*math.cos(ang), y + R*math.sin(ang)])
    return pts

# ---------------- Core GUI -----------------

class HexGUI:
    def __init__(self, size: int, bot_name: str, human_color: str, seed: Optional[int] = None):
        self.size = int(size)

        # 1) CREATE ROOT FIRST
        self.root = tk.Tk()
        self.root.title("Hex GUI")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 2) THEN MAKE Tk variables (tie them to the root)
        self.bot_name_var   = tk.StringVar(self.root, value=bot_name)
        self.human_color_var= tk.StringVar(self.root, value=human_color.capitalize())  # "Black"/"White"
        self.seed_var       = tk.StringVar(self.root, value="" if seed is None else str(seed))
        self.status_var     = tk.StringVar(self.root, value="Ready.")
        self.move_count_var = tk.StringVar(self.root, value="Moves: 0")

        self.is_thinking = False
        self.bot_thread = None
        self.rng = random.Random(seed if seed is not None else time.time_ns())

        # Sidebar controls
        self.sidebar = ttk.Frame(self.root, padding=8)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        # Bot picker
        ttk.Label(self.sidebar, text="Bot module:").grid(row=0, column=0, sticky="w")
        self.bot_combo = ttk.Combobox(self.sidebar, textvariable=self.bot_name_var, width=32, values=DEFAULT_BOT_CHOICES)
        self.bot_combo.grid(row=1, column=0, sticky="we", pady=(0,6))
        ttk.Button(self.sidebar, text="Load Bot", command=self.load_bot).grid(row=1, column=1, padx=(6,0))

        # Human side
        ttk.Label(self.sidebar, text="Human plays:").grid(row=2, column=0, sticky="w")
        self.side_combo = ttk.Combobox(self.sidebar, textvariable=self.human_color_var, values=["Black","White"], width=10)
        self.side_combo.grid(row=3, column=0, sticky="w", pady=(0,6))

        # Size
        ttk.Label(self.sidebar, text="Board size (n×n):").grid(row=4, column=0, sticky="w")
        self.size_var = tk.IntVar(value=self.size)
        size_entry = ttk.Spinbox(self.sidebar, from_=3, to=13, textvariable=self.size_var, width=6)
        size_entry.grid(row=5, column=0, sticky="w", pady=(0,6))

        # Seed
        ttk.Label(self.sidebar, text="RNG seed (optional):").grid(row=6, column=0, sticky="w")
        ttk.Entry(self.sidebar, textvariable=self.seed_var, width=12).grid(row=7, column=0, sticky="w", pady=(0,6))
        ttk.Button(self.sidebar, text="Apply Seed", command=self.apply_seed).grid(row=7, column=1, padx=(6,0))

        # Buttons
        btn_row = ttk.Frame(self.sidebar)
        btn_row.grid(row=8, column=0, columnspan=2, pady=(8,4), sticky="we")
        ttk.Button(btn_row, text="New Game", command=self.new_game).grid(row=0, column=0, padx=(0,6))
        ttk.Button(btn_row, text="Undo", command=self.undo_move).grid(row=0, column=1, padx=(0,6))

        # Status
        ttk.Label(self.sidebar, textvariable=self.status_var, wraplength=240, foreground="#444").grid(row=9, column=0, columnspan=2, sticky="we", pady=(10,2))
        ttk.Label(self.sidebar, textvariable=self.move_count_var, foreground="#444").grid(row=10, column=0, columnspan=2, sticky="we", pady=(0,4))

        # Canvas
        self.canvas = tk.Canvas(self.root, width=720, height=640, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Key-n>", lambda e: self.new_game())
        self.root.bind("<Key-u>", lambda e: self.undo_move())

        # Model state
        self.moves: List[Move] = []
        self.centers: dict[Move, Tuple[float,float]] = {}
        self.R = 1.0
        self.last_move: Optional[Move] = None

        # Load bot
        self.bot: Optional[BotRef] = None
        self.load_bot(init=True)

        # Build geometry & start
        self.relayout()
        self.new_game(first_draw=False)  # set up and optionally let bot move first

    # inside HexGUI
    def _compute_transform(self):
        """Return (s, tx, ty, W, H) to map model coords (x,y) -> screen: (s*x + tx, s*y + ty)."""
        # Canvas size (guard for the '1' px initial layout)
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W <= 1: W = 720
        if H <= 1: H = 640

        # Model bbox
        xs = [x for (x, y) in self.centers.values()]
        ys = [y for (x, y) in self.centers.values()]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

        pad = 30
        sx = (W - 2*pad) / (xmax - xmin + 1e-6)
        sy = (H - 2*pad) / (ymax - ymin + 1e-6)
        s = min(sx, sy)

        # Center inside the canvas (same as draw_board)
        tx = (W - s*(xmin + xmax)) / 2.0
        ty = (H - s*(ymin + ymax)) / 2.0
        return s, tx, ty, W, H


    # ------------- Bot loading -------------
    def load_bot(self, init: bool=False):
        name = self.bot_name_var.get().strip()
        try:
            mod = importlib.import_module(name)
            choose = getattr(mod, "choose_move", None)
            if choose is None:
                raise AttributeError(f"Module {name} has no choose_move")
            self.bot = BotRef(name=name, module=mod, choose=choose)
            self.status_var.set(f"Loaded bot: {name}")
        except Exception as e:
            self.bot = None
            self.status_var.set(f"Failed to load bot '{name}': {e}")
            if not init:
                messagebox.showerror("Load Bot", f"Could not load bot '{name}':\n{e}")

    def apply_seed(self):
        s = self.seed_var.get().strip()
        if s == "":
            self.rng = random.Random(time.time_ns())
            self.status_var.set("Seed cleared (randomized).")
            return
        try:
            val = int(s, 0)
            self.rng = random.Random(val)
            self.status_var.set(f"Seed set to {val}.")
        except Exception:
            messagebox.showerror("Seed", "Seed must be an integer (decimal or 0x... hex).")

    # ------------- Game state / drawing -------------

    def relayout(self):
        """Compute centers and radius to fit canvas size."""
        self.canvas.delete("all")
        W = self.canvas.winfo_width() or 720
        H = self.canvas.winfo_height() or 640
        n = self.size_var.get()
        # determine radius so that grid fits nicely
        # grid width ≈ (sqrt(3)*R)*(n-1) + (sqrt(3)/2*R)*(n-1) + 2*pad
        pad = 30
        # rough fit: iterate R candidates
        R = 20.0
        for _ in range(40):
            centers = hex_centers(n, R)
            xmin, xmax, ymin, ymax = bounds_from_centers(centers, pad=2.0)
            scale_x = (W - 2*pad) / (xmax - xmin + 1e-6)
            scale_y = (H - 2*pad) / (ymax - ymin + 1e-6)
            s = min(scale_x, scale_y)
            if s >= 1.0:
                R *= 1.1
            else:
                R *= 0.95
        self.R = R
        self.centers = hex_centers(n, R)
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        n = self.size_var.get()

        s, tx, ty, W, H = self._compute_transform()

        def T(pt):
            x, y = pt
            return (s*x + tx, s*y + ty)

        # cells
        for (r,c), (x,y) in self.centers.items():
            X, Y = T((x,y))
            poly = hex_corners(X, Y, self.R*s)
            self.canvas.create_polygon(*poly, fill="#f4f4f4", outline="#888888", width=1)

        # edge labels
        self.canvas.create_text((W/2), ty-12, text="Black: Top ↔ Bottom", fill="#333333")
        self.canvas.create_text(tx-12, (H/2), text="White: Left ↔ Right", angle=90, fill="#333333")

        # stones
        rstone = 0.58*self.R*s
        for i,(r,c) in enumerate(self.moves, start=1):
            x,y = self.centers[(r,c)]
            X,Y = T((x,y))
            color = "black" if (i%2==1) else "white"
            edge  = "white" if (i%2==1) else "black"
            self.canvas.create_oval(X-rstone, Y-rstone, X+rstone, Y+rstone, fill=color, outline=edge, width=2)
            self.canvas.create_text(X, Y, text=str(i), fill=("white" if (i%2==1) else "black"))
        if self.moves:
            r,c = self.moves[-1]
            x,y = self.centers[(r,c)]
            X,Y = T((x,y))
            self.canvas.create_oval(X-rstone*1.15, Y-rstone*1.15, X+rstone*1.15, Y+rstone*1.15, outline="#2b8cbe", width=2)

        self.move_count_var.set(f"Moves: {len(self.moves)}")

    # ------------- Interaction -------------

    def on_click(self, event):
        if self.is_thinking:
            return
        if self.game_over():
            self.status_var.set("Game over. Click New Game to play again.")
            return

        # SAME transform as draw_board
        s, tx, ty, _, _ = self._compute_transform()

        # nearest center in *screen space*
        best = None; bestd2 = 1e18
        for (r,c), (x,y) in self.centers.items():
            X = s*x + tx
            Y = s*y + ty
            d2 = (X - event.x)**2 + (Y - event.y)**2
            if d2 < bestd2:
                bestd2 = d2; best = (r,c, X, Y)
        if best is None:
            return

        r,c, X, Y = best

        # Optionally, ignore clicks far from any hex (prevents off-board picks)
        # threshold = (self.R * s) * 0.9
        # if bestd2 > threshold*threshold:
        #     return

        if (r,c) in set(self.moves):
            self.status_var.set("Cell already occupied.")
            return

        human_color = self.human_color_var.get()
        to_move = 'Black' if (len(self.moves)%2==0) else 'White'
        if to_move != human_color:
            self.status_var.set(f"Not {human_color}'s turn.")
            return

        self.moves.append((r,c))
        self.draw_board()
        self.check_winner_and_maybe_bot()


    def check_winner_and_maybe_bot(self):
        n = self.size_var.get()
        winner, _ = evaluate_hex(n, self.moves)
        if winner:
            self.status_var.set(f"Winner: {winner}.")
            self.draw_board()
            return
        # Bot to move?
        to_move = 'Black' if (len(self.moves)%2==0) else 'White'
        human_color = self.human_color_var.get()
        if to_move == human_color:
            return
        self.start_bot_turn()

    def start_bot_turn(self):
        if self.bot is None:
            self.status_var.set("No bot loaded.")
            return
        # Guard: RL bot currently supports size=5 only (it will raise ValueError otherwise)
        self.is_thinking = True
        self.status_var.set(f"{self.bot.name} thinking...")
        moves_snapshot = list(self.moves)
        size_snapshot = int(self.size_var.get())
        rng_snapshot = self.rng

        def worker():
            try:
                mv = self.bot.choose(size_snapshot, moves_snapshot, rng_snapshot)
                self.root.after(0, lambda: self.apply_bot_move(mv))
            except Exception as e:
                self.root.after(0, lambda: self.on_bot_error(e))
        self.bot_thread = threading.Thread(target=worker, daemon=True)
        self.bot_thread.start()

    def apply_bot_move(self, move: Move):
        self.is_thinking = False
        # validate move (must be legal)
        if move in set(self.moves):
            self.status_var.set(f"Bot played illegal (occupied) move {move}; bot loses.")
            # Winner is other side (the human)
            self.draw_board()
            return
        n = self.size_var.get()
        r,c = move
        if not (0 <= r < n and 0 <= c < n):
            self.status_var.set(f"Bot played out-of-bounds {move}; bot loses.")
            self.draw_board()
            return
        self.moves.append(move)
        self.draw_board()
        winner, _ = evaluate_hex(n, self.moves)
        if winner:
            self.status_var.set(f"Winner: {winner}.")
        else:
            self.status_var.set("Your turn.")

    def on_bot_error(self, e: Exception):
        self.is_thinking = False
        messagebox.showerror("Bot error", f"The bot raised an exception:\n{e}")
        self.status_var.set("Bot error. See message.")

    def game_over(self) -> bool:
        n = self.size_var.get()
        winner, _ = evaluate_hex(n, self.moves)
        return winner is not None or len(self.moves) == n*n

    def new_game(self, first_draw: bool=True):
        if self.is_thinking:
            return
        self.moves.clear()
        self.status_var.set("New game.")
        # If human is White, bot (Black) moves first
        human_color = self.human_color_var.get()
        self.draw_board()
        if human_color == 'White':
            self.start_bot_turn()

    def undo_move(self):
        if self.is_thinking:
            return
        if not self.moves:
            return
        # Undo last move; if that was bot's, it becomes human's turn
        self.moves.pop()
        # Also undo human move if you'd like undo pairs:
        # if self.moves: self.moves.pop()
        self.draw_board()
        self.status_var.set("Undid last move.")

    def _on_close(self):
        self.root.destroy()

    def run(self):
        # Redraw on resize
        def on_resize(_e):
            self.relayout()
        self.canvas.bind("<Configure>", on_resize)
        self.root.mainloop()


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Play Hex vs any bot module.")
    ap.add_argument("--bot", type=str, default="two_ahead_random_player",
                    help="Module name exposing choose_move(size, moves, rng=None)")
    ap.add_argument("--size", type=int, default=5, help="Board size n (n×n)")
    ap.add_argument("--human", type=str, default="Black", choices=["Black","White","black","white"])
    ap.add_argument("--seed", type=str, default="", help="Optional RNG seed (int).")
    args = ap.parse_args()

    seed = None
    if args.seed.strip():
        try:
            seed = int(args.seed, 0)
        except Exception:
            print("Seed must be an integer (e.g., 42 or 0x1234). Ignoring.", file=sys.stderr)

    gui = HexGUI(size=args.size, bot_name=args.bot, human_color=args.human, seed=seed)
    gui.run()

if __name__ == "__main__":
    main()
