# train_rl_numpy.py
from __future__ import annotations
import os, csv, time, argparse, importlib, io
import numpy as np
from typing import List, Tuple, Dict, Any
from rl_model_numpy import PolicyMLP
from hex_env import HexEnv5

Move = Tuple[int, int]

def obs_and_mask(moves: List[Move]) -> tuple[np.ndarray, np.ndarray]:
    env = HexEnv5(); env.moves = list(moves)
    x = env._obs().reshape(-1)  # [75]
    legal = np.ones(25, dtype=bool)
    for (r, c) in set(moves): legal[r*5 + c] = False
    return x, legal

def play_episode(model: PolicyMLP, rng: np.random.Generator,
                 random_open_k: int,
                 opponent: PolicyMLP | None) -> Dict[str, Any]:
    """
    Self-play. If opponent is None, mirror self-play (but we still only learn from
    one colour per episode). If opponent is provided, that side plays greedy.
    We only record steps (x, mask, a, side) for our agent's turns.
    """
    env = HexEnv5(); env.reset()
    # Random opening
    for _ in range(random_open_k):
        legals = env.legal_moves()
        if not legals: break
        env.step(legals[rng.integers(0, len(legals))])

    # Choose which colour our agent plays this episode
    rl_as_black = bool(rng.integers(0, 2))  # 50% chance

    steps = []  # only our actions
    while True:
        to_black = (len(env.moves) % 2 == 0)
        am_rl_turn = (to_black and rl_as_black) or ((not to_black) and (not rl_as_black))
        if am_rl_turn:
            x = env._obs().reshape(-1); mask = np.ones(25, dtype=bool)
            for (r,c) in set(env.moves): mask[r*5+c] = False
            a = model.sample(x, mask, temperature=1.0, rng=rng)
            side = 1 if to_black else -1
            _, _, done, _ = env.step((a//5, a%5))
            steps.append({"x": x, "mask": mask, "a": a, "side": side})
        else:
            # opponent acts (greedy). If None, use our model greedy to keep games sensible.
            x = env._obs().reshape(-1); mask = np.ones(25, dtype=bool)
            for (r,c) in set(env.moves): mask[r*5+c] = False
            if opponent is None:
                a = model.greedy(x, mask)
            else:
                a = opponent.greedy(x, mask)
            _, _, done, _ = env.step((a//5, a%5))

        if done:
            winner = env.winner  # 'Black' or 'White'
            # Assign returns for our steps only
            ret_for_us = 1.0 if ((winner == 'Black') == rl_as_black) else -1.0
            for st in steps:
                st["ret"] = ret_for_us
            return {"steps": steps, "winner": winner, "rl_as_black": rl_as_black}

def batch_from_eps(eps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    batch = []
    for ep in eps:
        for st in ep["steps"]:
            batch.append({
                "x": st["x"],
                "mask": st["mask"],
                "a": st["a"],
                "side": st["side"],     # +1 black, -1 white
                "ret": st["ret"],       # +1/-1 from our agent's perspective
            })
    return batch

#def eval_vs_module(model_eval: PolicyMLP, opponent_mod: str, games: int, seed: int = 0) -> float:
def eval_vs_module(model_eval: PolicyMLP, opponent_mod: str, games: int, seed: int = 0):
    """Greedy RL vs opponent over `games`, alternating colours."""
    opp = importlib.import_module(opponent_mod)
    rng = np.random.default_rng(seed)
    wins = 0
    black_wins = 0
    white_wins = 0
    rl_as_black = True
    for g in range(games):
        env = HexEnv5(); env.reset()
        while True:
            to_black = (len(env.moves) % 2 == 0)
            if (to_black and rl_as_black) or ((not to_black) and (not rl_as_black)):
                x = env._obs().reshape(-1); mask = np.ones(25, dtype=bool)
                for (r,c) in set(env.moves): mask[r*5+c] = False
                a = model_eval.greedy(x, mask)
                move = (a//5, a%5)
            else:
                move = opp.choose_move(5, env.moves, None)
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
    ap = argparse.ArgumentParser(description="NumPy RL for 5x5 Hex (self-play with frozen opponent)")
    ap.add_argument("--batch-games", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=522)
    ap.add_argument("--eval-games", type=int, default=800)
    ap.add_argument("--opponent1", type=str, default="two_ahead_random_player_c")
    ap.add_argument("--opponent2", type=str, default="rl_player")
    ap.add_argument("--frozen-prob", type=float, default=0.5, help="Prob. an episode is vs frozen best")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # init / resume
    best_path = "best_model.npz"
    if os.path.exists(best_path):
        model = PolicyMLP.load(best_path)
        print(f"Loaded {best_path}")
    else:
        model = PolicyMLP(seed=args.seed)
        model.save(best_path, step=0)

    metrics_path = "metrics/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["step","games_seen","loss","entropy","wr_op1","wr_op2"])

    step = 0
    games_seen = 0
    best_vs_op2 = 0.0
    frozen = PolicyMLP.load(best_path)  # start with best

    while True:
        # 1) Collect a batch
        episodes: List[Dict[str, Any]] = []
        while len(episodes) < args.batch_games:
            k = int(rng.integers(0, 5))  # 0..4 random opening moves
            opp_model = frozen if (rng.random() < args.frozen_prob) else None
            ep = play_episode(model, rng, random_open_k=k, opponent=opp_model)
            episodes.append(ep)
        batch = batch_from_eps(episodes)
        games_seen += len(episodes)

        # 2) Update policy (learn only from our agent's decisions)
        stats = model.update(batch, lr=args.lr, clip=2.0, weight_decay=1e-5)
        step += 1

        # 3) Periodic save
        if (step % args.save_every) == 0:
            path = f"checkpoints/rl_{games_seen:06d}.npz"
            model.save(path, step=games_seen)
            print(f"[{games_seen}] saved {path}  loss={stats['loss']:.3f}  H={stats['entropy']:.3f}  bB={stats['baseline_black']:.3f} bW={stats['baseline_white']:.3f}")

        # 4) Periodic eval (bigger samples to reduce noise)
        if (step % args.eval_every) == 0:
            eval_model = model
            wr1, wrb1, wrw1 = eval_vs_module(eval_model, args.opponent1, args.eval_games, seed=123)
            wr2, wrb2, wrw2 = eval_vs_module(eval_model, args.opponent2, args.eval_games, seed=456)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([step, games_seen, stats["loss"], stats["entropy"], wr1, wr2])
            #print(f"[{games_seen}] win% vs {args.opponent1}: {100*wr1:.1f}  vs {args.opponent2}: {100*wr2:.1f}")
            print(f"[{games_seen}] win% vs {args.opponent1}: {100*wr1:.1f}, black: {200*wrb1:.1f}, white: {200*wrw1:.1f} vs {args.opponent2}: {100*wr2:.1f}, black: {200*wrb2:.1f}, white: {200*wrw2:.1f}")
            # Update best on smoothed criterion (vs op2)
            if wr2 > best_vs_op2:
                best_vs_op2 = wr2
                model.save(best_path, step=games_seen)
                frozen = PolicyMLP.load(best_path)
                print(f"  ↑ new best_model.npz ({100*wr2:.1f}% vs {args.opponent2})")

if __name__ == "__main__":
    main()


# # train_rl_numpy.py
# from __future__ import annotations
# import os, csv, time, argparse, importlib
# import numpy as np
# from typing import List, Tuple, Dict, Any
# from rl_model_numpy import PolicyMLP
# from hex_env import HexEnv5
#
# Move = Tuple[int, int]
#
# # Prefer the fast C evaluator inside env; already handled by HexEnv5.
#
# def state_and_mask_from_moves(moves: List[Move]) -> tuple[np.ndarray, np.ndarray]:
#     env = HexEnv5(); env.moves = list(moves)
#     x = env._obs().reshape(-1)  # [75]
#     legal = np.ones(25, dtype=bool)
#     for (r, c) in set(moves): legal[r*5 + c] = False
#     return x, legal
#
# def play_selfplay_episode(model: PolicyMLP, rng: np.random.Generator,
#                           random_opening_k: int = 0, temperature: float = 1.0) -> Dict[str, Any]:
#     env = HexEnv5(); env.reset()
#     # random opening (optional)
#     for _ in range(random_opening_k):
#         legals = env.legal_moves()
#         if not legals: break
#         env.step(legals[rng.integers(0, len(legals))])
#
#     steps = []
#     while True:
#         x = env._obs().reshape(-1)  # [75]
#         legal = np.ones(25, dtype=bool)
#         for (r, c) in set(env.moves): legal[r*5 + c] = False
#         a = model.sample(x, legal_mask=legal, temperature=temperature, rng=rng)
#         s2, r, done, _ = env.step((a // 5, a % 5))
#         steps.append({"x": x, "mask": legal, "a": a, "player": env.current_player() if done else None})
#         if done:
#             # reward r is for the player who JUST moved; we need to assign returns per move:
#             # moves made by the final mover get +1, the other side −1.
#             # We can determine sides by move parity.
#             winner = env.winner  # 'Black' or 'White'
#             returns = []
#             for i in range(len(env.moves)):
#                 mover = 'Black' if (i % 2 == 0) else 'White'
#                 returns.append(1.0 if mover == winner else -1.0)
#             return {"steps": steps, "returns": returns, "winner": winner}
#
# def batch_from_episodes(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     batch = []
#     for ep in episodes:
#         for t, step in enumerate(ep["steps"]):
#             batch.append({
#                 "x": step["x"],
#                 "mask": step["mask"],
#                 "a": step["a"],
#                 "adv": ep["returns"][t],  # raw ±1 (we'll normalize per batch below)
#             })
#     # normalize advantages per batch (variance reduction)
#     advs = np.array([b["adv"] for b in batch], dtype=np.float32)
#     advs = (advs - advs.mean()) / (advs.std() + 1e-6)
#     for i in range(len(batch)):
#         batch[i]["adv"] = float(advs[i])
#     return batch
#
# def play_match(model_eval: PolicyMLP, opponent_mod: str, games: int, seed: int = 12345) -> float:
#     """Evaluate greedy RL vs opponent module (alternate colors). Returns win-rate for RL."""
#     opp = importlib.import_module(opponent_mod)
#     rl_first_black = True
#     wins = 0
#     rng = np.random.default_rng(seed)
#     for g in range(games):
#         env = HexEnv5(); env.reset()
#         while True:
#             to_black = (len(env.moves) % 2 == 0)
#             if to_black:
#                 # player to move is Black
#                 if rl_first_black:
#                     # RL acts
#                     x = env._obs().reshape(-1)
#                     legal = np.ones(25, dtype=bool)
#                     for (r, c) in set(env.moves): legal[r*5 + c] = False
#                     a = model_eval.greedy(x, legal)
#                     move = (a // 5, a % 5)
#                 else:
#                     move = opp.choose_move(5, env.moves, None)
#             else:
#                 if rl_first_black:
#                     move = opp.choose_move(5, env.moves, None)
#                 else:
#                     x = env._obs().reshape(-1)
#                     legal = np.ones(25, dtype=bool)
#                     for (r, c) in set(env.moves): legal[r*5 + c] = False
#                     a = model_eval.greedy(x, legal)
#                     move = (a // 5, a % 5)
#             _, _, done, _ = env.step(move)
#             if done:
#                 if env.winner == 'Black':
#                     wins += 1 if rl_first_black else 0
#                 elif env.winner == 'White':
#                     wins += 1 if not rl_first_black else 0
#                 break
#         rl_first_black = not rl_first_black  # alternate colors
#     return wins / games
#
# def main():
#     ap = argparse.ArgumentParser(description="NumPy RL for 5x5 Hex (self-play)")
#     ap.add_argument("--batch-games", type=int, default=128)
#     ap.add_argument("--eval-every", type=int, default=500)
#     ap.add_argument("--save-every", type=int, default=500)
#     ap.add_argument("--lr", type=float, default=3e-3)
#     ap.add_argument("--temperature", type=float, default=1.0)
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--eval-random-games", type=int, default=200)
#     #ap.add_argument("--eval-oneahead-games", type=int, default=200)
#     ap.add_argument("--eval-twoahead-games", type=int, default=200)
#     ap.add_argument("--opponent1", type=str, default="random_player_c")
#     #ap.add_argument("--opponent2", type=str, default="one_ahead_random_player_c")
#     ap.add_argument("--opponent2", type=str, default="two_ahead_random_player_c")
#     args = ap.parse_args()
#
#     rng = np.random.default_rng(args.seed)
#     os.makedirs("checkpoints", exist_ok=True)
#     os.makedirs("metrics", exist_ok=True)
#
#     # init or resume
#     best_path = "best_model.npz"
#     if os.path.exists(best_path):
#         model = PolicyMLP.load(best_path)
#         print(f"Loaded {best_path}")
#     else:
#         model = PolicyMLP(seed=args.seed)
#         model.save(best_path, step=0)
#
#     metrics_path = "metrics/metrics.csv"
#     if not os.path.exists(metrics_path):
#         with open(metrics_path, "w", newline="") as f:
#             #csv.writer(f).writerow(["step","games_seen","loss","entropy","win_vs_random","win_vs_oneahead"])
#             csv.writer(f).writerow(["step","games_seen","loss","entropy","win_vs_random","win_vs_twoahead"])
#
#     step = 0
#     games_seen = 0
#     #best_vs_oneahead = 0.0
#     best_vs_twoahead = 0.0
#
#     while True:
#         # 1) Collect a batch of self-play episodes
#         episodes: List[Dict[str, Any]] = []
#         while len(episodes) < args.batch_games:
#             k = int(rng.integers(0, 5))  # 0..4 random opening moves
#             ep = play_selfplay_episode(model, rng, random_opening_k=k, temperature=args.temperature)
#             episodes.append(ep)
#         batch = batch_from_episodes(episodes)
#         games_seen += len(episodes)
#
#         # 2) Policy update
#         stats = model.update(batch, lr=args.lr)
#         step += 1
#
#         # 3) Periodic save
#         if (step % args.save_every) == 0:
#             path = f"checkpoints/rl_{games_seen:06d}.npz"
#             model.save(path, step=games_seen)
#             print(f"[{games_seen}] saved {path}  loss={stats['loss']:.3f}  H={stats['entropy']:.3f}")
#
#         # 4) Periodic evaluation
#         if (step % args.eval_every) == 0:
#             eval_model = model  # greedy eval
#             w1 = play_match(eval_model, args.opponent1, args.eval_random_games, seed=123)
#             #w2 = play_match(eval_model, args.opponent2, args.eval_oneahead_games, seed=456)
#             w2 = play_match(eval_model, args.opponent2, args.eval_twoahead_games, seed=456)
#             with open(metrics_path, "a", newline="") as f:
#                 csv.writer(f).writerow([step, games_seen, stats["loss"], stats["entropy"], w1, w2])
#             print(f"[{games_seen}] win% vs {args.opponent1}: {100*w1:.1f}  vs {args.opponent2}: {100*w2:.1f}")
#             # keep best
#             # if w2 > best_vs_oneahead:
#             #     best_vs_oneahead = w2
#             #     model.save(best_path, step=games_seen)
#             #     print(f"  ↑ new best_model.npz (vs {args.opponent2}: {100*w2:.1f}%)")
#             if w2 > best_vs_twoahead:
#                 best_vs_twoahead = w2
#                 model.save(best_path, step=games_seen)
#                 print(f"  ↑ new best_model.npz (vs {args.opponent2}: {100*w2:.1f}%)")
#
# if __name__ == "__main__":
#     main()
