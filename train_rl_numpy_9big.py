# train_rl_numpy_9big.py
from __future__ import annotations
import os, csv, argparse, importlib
import numpy as np
from typing import List, Tuple, Dict, Any
from rl_model_numpy_9big import PolicyMLP9Big
from hex_env9 import HexEnv9

Move = Tuple[int, int]

def play_episode(model: PolicyMLP9Big, rng: np.random.Generator,
                 random_open_k: int, opponent: PolicyMLP9Big | None) -> Dict[str, Any]:
    env = HexEnv9(); env.reset()
    for _ in range(random_open_k):
        legals = env.legal_moves()
        if not legals: break
        env.step(legals[rng.integers(0, len(legals))])
    rl_as_black = bool(rng.integers(0, 2))
    steps = []
    while True:
        to_black = (len(env.moves) % 2 == 0)
        am_rl = (to_black and rl_as_black) or ((not to_black) and (not rl_as_black))
        if am_rl:
            x = env._obs().reshape(-1)
            legal = np.ones(81, dtype=bool)
            for (r,c) in set(env.moves): legal[r*9+c] = False
            a = model.sample(x, legal, temperature=1.0, rng=rng)
            _, _, done, _ = env.step((a//9, a%9))
            steps.append({"x": x, "mask": legal, "a": a, "side": (1 if to_black else -1)})
        else:
            x = env._obs().reshape(-1)
            legal = np.ones(81, dtype=bool)
            for (r,c) in set(env.moves): legal[r*9+c] = False
            a = (model if opponent is None else opponent).greedy(x, legal)
            _, _, done, _ = env.step((a//9, a%9))
        if done:
            ret = 1.0 if ((env.winner == 'Black') == rl_as_black) else -1.0
            for st in steps: st["ret"] = ret
            return {"steps": steps, "winner": env.winner, "rl_as_black": rl_as_black}

def batch_from_eps(eps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    batch = []
    for ep in eps:
        for st in ep["steps"]:
            batch.append({"x": st["x"], "mask": st["mask"], "a": st["a"], "side": st["side"], "ret": st["ret"]})
    return batch

def eval_vs_module(model_eval: PolicyMLP9Big, opponent_mod: str, games: int, seed: int = 0) -> float:
    opp = importlib.import_module(opponent_mod)
    wins = 0; rl_as_black = True
    for g in range(games):
        env = HexEnv9(); env.reset()
        while True:
            to_black = (len(env.moves) % 2 == 0)
            if (to_black and rl_as_black) or ((not to_black) and (not rl_as_black)):
                x = env._obs().reshape(-1)
                legal = np.ones(81, dtype=bool)
                for (r,c) in set(env.moves): legal[r*9+c] = False
                a = model_eval.greedy(x, legal); move = (a//9, a%9)
            else:
                move = opp.choose_move(9, env.moves, None)
            _, _, done, _ = env.step(move)
            if done:
                if (env.winner == 'Black') == rl_as_black: wins += 1
                break
        rl_as_black = not rl_as_black
    return wins / games

def main():
    ap = argparse.ArgumentParser(description="NumPy RL (large) for 9x9 Hex")
    ap.add_argument("--batch-games", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--eval-games", type=int, default=600)
    ap.add_argument("--opponent1", type=str, default="random_player_c")
    ap.add_argument("--opponent2", type=str, default="two_ahead_random_player_c")
    ap.add_argument("--frozen-prob", type=float, default=0.5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs("checkpoints9big", exist_ok=True)
    os.makedirs("metrics9big", exist_ok=True)

    best_path = "best9big_model.npz"
    if os.path.exists(best_path):
        model = PolicyMLP9Big.load(best_path); print(f"Loaded {best_path}")
    else:
        # If you already have best9_model.npz from the smaller net, you can warm start:
        if os.path.exists("best9_model.npz"):
            model = PolicyMLP9Big.load("best9_model.npz")  # migrated into bigger net
            model.save(best_path, step=0); print("Migrated small 9x9 → big and saved best9big_model.npz")
        else:
            model = PolicyMLP9Big(seed=args.seed); model.save(best_path, step=0)

    metrics_path = "metrics9big/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["step","games_seen","loss","entropy","wr_random","wr_op2"])

    step = 0; games_seen = 0; best_vs_op2 = 0.0
    frozen = PolicyMLP9Big.load(best_path)

    while True:
        # 1) collect batch
        episodes: List[Dict[str, Any]] = []
        while len(episodes) < args.batch_games:
            k = int(rng.integers(0, 9))  # 0..8 random opening moves
            opp_model = frozen if (rng.random() < args.frozen_prob) else None

            episodes.append(play_episode(model, rng, random_open_k=k, opponent=opp_model))
        batch = batch_from_eps(episodes); games_seen += len(episodes)

        # 2) update
        stats = model.update(batch, lr=args.lr, clip=3.0, weight_decay=1e-5,
                             baseline_alpha=0.05, entropy_coef=args.entropy)
        step += 1

        # 3) save
        if (step % args.save_every) == 0:
            path = f"checkpoints9big/rl9big_{games_seen:06d}.npz"
            model.save(path, step=games_seen)
            print(f"[{games_seen}] saved {path}  loss={stats['loss']:.3f}  H={stats['entropy']:.3f}  bB={stats['baseline_black']:.3f} bW={stats['baseline_white']:.3f}")

        # 4) eval
        if (step % args.eval_every) == 0:
            wr1 = eval_vs_module(model, args.opponent1, args.eval_games, seed=123)
            wr2 = eval_vs_module(model, args.opponent2, args.eval_games, seed=456)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([step, games_seen, stats["loss"], stats["entropy"], wr1, wr2])
            print(f"[{games_seen}] win% vs {args.opponent1}: {100*wr1:.1f}  vs {args.opponent2}: {100*wr2:.1f}")
            if wr2 > best_vs_op2:
                best_vs_op2 = wr2
                model.save(best_path, step=games_seen)
                frozen = PolicyMLP9Big.load(best_path)
                print(f"  ↑ new best9big_model.npz ({100*wr2:.1f}% vs {args.opponent2})")

if __name__ == "__main__":
    main()
