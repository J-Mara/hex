# train_rl_numpy_7.py
from __future__ import annotations
import os, csv, argparse, importlib
import numpy as np
from typing import List, Tuple, Dict, Any
from rl_model_numpy_7 import PolicyMLP7
from hex_env7 import HexEnv7
from types import ModuleType

Move = Tuple[int, int]

def play_episode(model: PolicyMLP7, rng: np.random.Generator,
                 random_open_k: int,
                 opponent: PolicyMLP7 | ModuleType | None) -> Dict[str, Any]:
    env = HexEnv7(); env.reset()
    # random opening to diversify
    for _ in range(random_open_k):
        legals = env.legal_moves()
        if not legals: break
        env.step(legals[rng.integers(0, len(legals))])

    rl_as_black = bool(rng.integers(0, 2))
    steps = []
    while True:
        to_black = (len(env.moves) % 2 == 0)
        am_rl_turn = (to_black and rl_as_black) or ((not to_black) and (not rl_as_black))
        if am_rl_turn:
            x = env._obs().reshape(-1)
            legal = np.ones(49, dtype=bool)
            for (r,c) in set(env.moves): legal[r*7+c] = False
            a = model.sample(x, legal, temperature=1.0, rng=rng)
            _, _, done, _ = env.step((a//7, a%7))
            side = 1 if to_black else -1
            steps.append({"x": x, "mask": legal, "a": a, "side": side})
        else:
            x = env._obs().reshape(-1)
            legal = np.ones(49, dtype=bool)
            for (r,c) in set(env.moves): legal[r*7+c] = False
            #a = (model if opponent is None else opponent).greedy(x, legal)
            if(opponent is None):
                a = model.greedy(x, legal)
                _, _, done, _ = env.step((a//7, a%7))
            else:
                if(opponent is PolicyMLP7):
                    a = opponent.greedy(x, legal)
                    _, _, done, _ = env.step((a//7, a%7))
                else:
                    _, _, done, _ = env.step(opponent.choose_move(7, env.moves, None))
            #_, _, done, _ = env.step((a//7, a%7))
        if done:
            win_as_rl = (env.winner == 'Black') == rl_as_black
            ret = 1.0 if win_as_rl else -1.0
            for st in steps: st["ret"] = ret
            return {"steps": steps, "winner": env.winner, "rl_as_black": rl_as_black}

def batch_from_eps(eps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    batch = []
    for ep in eps:
        for st in ep["steps"]:
            batch.append({
                "x": st["x"],
                "mask": st["mask"],
                "a": st["a"],
                "side": st["side"],
                "ret": st["ret"],
            })
    return batch

def eval_vs_module(model_eval: PolicyMLP7, opponent_mod: str, games: int, seed: int = 0):
    opp = importlib.import_module(opponent_mod)
    wins = 0
    black_wins = 0
    white_wins = 0
    rl_as_black = True
    for g in range(games):
        env = HexEnv7(); env.reset()
        while True:
            to_black = (len(env.moves) % 2 == 0)
            if (to_black and rl_as_black) or ((not to_black) and (not rl_as_black)):
                x = env._obs().reshape(-1)
                legal = np.ones(49, dtype=bool)
                for (r,c) in set(env.moves): legal[r*7+c] = False
                a = model_eval.greedy(x, legal)
                move = (a//7, a%7)
            else:
                move = opp.choose_move(7, env.moves, None)
            _, _, done, _ = env.step(move)
            if done:
                if (env.winner == 'Black') == rl_as_black:
                    wins += 1
                    if(rl_as_black):
                        black_wins += 1
                    else:
                        white_wins += 1
                break
        rl_as_black = not rl_as_black
    return (wins / games), (black_wins / games), (white_wins / games)

def main():
    ap = argparse.ArgumentParser(description="NumPy RL for 7x7 Hex (self-play with frozen opponent)")
    ap.add_argument("--batch-games", type=int, default=96)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-games", type=int, default=600)
    ap.add_argument("--opponent1", type=str, default="random_player_c")
    ap.add_argument("--opponent2", type=str, default="two_ahead_random_player_c")
    ap.add_argument("--frozen-prob", type=float, default=0.4)
    ap.add_argument("--benchmark-active", type=bool, default=False)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs("checkpoints7", exist_ok=True)
    os.makedirs("metrics7", exist_ok=True)

    best_path = "best7_model.npz"
    if os.path.exists(best_path):
        model = PolicyMLP7.load(best_path)
        print(f"Loaded {best_path}")
    else:
        model = PolicyMLP7(seed=args.seed)
        model.save(best_path, step=0)

    metrics_path = "metrics7/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["step","games_seen","loss","entropy","wr_op1","wr_op2"])

    step = -1
    games_seen = 0
    best_vs_op2 = 0.0
    frozen = PolicyMLP7.load(best_path)
    benchmark = importlib.import_module('two_ahead_random_player_c')

    while True:
        # 1) collect a batch
        if not args.benchmark_active:
            episodes: List[Dict[str, Any]] = []
            while len(episodes) < args.batch_games:
                k = int(rng.integers(0, 7))  # 0..6 random opening moves
                opp_model = frozen if (rng.random() < args.frozen_prob) else None
                ep = play_episode(model, rng, random_open_k=k, opponent=opp_model)
                episodes.append(ep)
            batch = batch_from_eps(episodes)
            games_seen += len(episodes)
        else:
            episodes: List[Dict[str, Any]] = []
            while len(episodes) < args.batch_games:
                k = int(rng.integers(0, 7))
                opp_player = benchmark
                ep = play_episode(model, rng, random_open_k=k, opponent=opp_player)
                episodes.append(ep)
            batch = batch_from_eps(episodes)
            games_seen += len(episodes)

        # 2) update
        stats = model.update(batch, lr=args.lr, clip=2.0, weight_decay=1e-5)
        step += 1

        # 3) save
        if (step % args.save_every) == 0:
            path = f"checkpoints7/rl7_{games_seen:06d}.npz"
            model.save(path, step=games_seen)
            print(f"[{games_seen}] saved {path}  loss={stats['loss']:.3f}  H={stats['entropy']:.3f}  bB={stats['baseline_black']:.3f} bW={stats['baseline_white']:.3f}")

        # 4) eval
        if (step % args.eval_every) == 0:
            eval_model = model
            wr1, wrb1, wrw1 = eval_vs_module(eval_model, args.opponent1, args.eval_games, seed=123)
            wr2, wrb2, wrw2 = eval_vs_module(eval_model, args.opponent2, args.eval_games, seed=456)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([step, games_seen, stats["loss"], stats["entropy"], wr1, wr2])
            #print(f"[{games_seen}] win% vs {args.opponent1}: {100*wr1:.1f}  vs {args.opponent2}: {100*wr2:.1f}")
            print(f"[{games_seen}] win% vs {args.opponent1}: {100*wr1:.1f}, black: {200*wrb1:.1f}, white: {200*wrw1:.1f} vs {args.opponent2}: {100*wr2:.1f}, black: {200*wrb2:.1f}, white: {200*wrw2:.1f}")
            if wr2 > best_vs_op2:
                best_vs_op2 = wr2
                model.save(best_path, step=games_seen)
                frozen = PolicyMLP7.load(best_path)
                print(f"  ↑ new best7_model.npz ({100*wr2:.1f}% vs {args.opponent2})")

if __name__ == "__main__":
    main()

# time python train_rl_numpy_7.py --batch-games 1024 --eval-every 300 --save-every 300 --lr 0.0015 --opponent1 two_ahead_random_player_c --opponent2 rl_player7 --eval-games 5000 --seed 252 --benchmark-active False
