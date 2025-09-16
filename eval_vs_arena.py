# eval_vs_arena.py
import argparse, importlib, os
import numpy as np
from rl_model_numpy import PolicyMLP
from hex_env import HexEnv5

def play_match(model_eval: PolicyMLP, opponent_mod: str, games: int, seed: int = 0) -> float:
    opp = importlib.import_module(opponent_mod)
    wins = 0
    rl_first_black = True
    rng = np.random.default_rng(seed)
    for g in range(games):
        env = HexEnv5(); env.reset()
        while True:
            to_black = (len(env.moves) % 2 == 0)
            if to_black:
                if rl_first_black:
                    x = env._obs().reshape(-1)
                    legal = np.ones(25, dtype=bool)
                    for (r,c) in set(env.moves): legal[r*5+c] = False
                    a = model_eval.greedy(x, legal)
                    move = (a//5, a%5)
                else:
                    move = opp.choose_move(5, env.moves, None)
            else:
                if rl_first_black:
                    move = opp.choose_move(5, env.moves, None)
                else:
                    x = env._obs().reshape(-1)
                    legal = np.ones(25, dtype=bool)
                    for (r,c) in set(env.moves): legal[r*5+c] = False
                    a = model_eval.greedy(x, legal)
                    move = (a//5, a%5)
            _, _, done, _ = env.step(move)
            if done:
                if env.winner == 'Black':
                    wins += 1 if rl_first_black else 0
                else:
                    wins += 1 if not rl_first_black else 0
                break
        rl_first_black = not rl_first_black
    return wins / games

def main():
    ap = argparse.ArgumentParser(description="Evaluate saved RL checkpoint vs any opponent")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--opponent", required=True, help="e.g., random_player or one_ahead_random_player")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model = PolicyMLP.load(args.weights)
    wr = play_match(model, args.opponent, args.games, args.seed)
    print(f"Win-rate vs {args.opponent}: {100*wr:.2f}% over {args.games} games")

if __name__ == "__main__":
    main()
