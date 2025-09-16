# eval_vs_arena_9.py
import argparse, importlib
import numpy as np
from rl_model_numpy_9 import PolicyMLP9
from hex_env9 import HexEnv9

def play_match(model_eval: PolicyMLP9, opponent_mod: str, games: int, seed: int = 0) -> float:
    opp = importlib.import_module(opponent_mod)
    wins = 0
    rl_as_black = True
    for g in range(games):
        env = HexEnv9(); env.reset()
        while True:
            to_black = (len(env.moves) % 2 == 0)
            if (to_black and rl_as_black) or ((not to_black) and (not rl_as_black)):
                x = env._obs().reshape(-1)
                legal = np.ones(81, dtype=bool)
                for (r,c) in set(env.moves): legal[r*9+c] = False
                a = model_eval.greedy(x, legal)
                move = (a//9, a%9)
            else:
                move = opp.choose_move(9, env.moves, None)
            _, _, done, _ = env.step(move)
            if done:
                if (env.winner == 'Black') == rl_as_black:
                    wins += 1
                break
        rl_as_black = not rl_as_black
    return wins / games

def main():
    ap = argparse.ArgumentParser(description="Evaluate 9x9 RL checkpoint vs any opponent")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = PolicyMLP9.load(args.weights)
    wr = play_match(model, args.opponent, args.games, args.seed)
    print(f"Win-rate vs {args.opponent}: {100*wr:.2f}% over {args.games} games")

if __name__ == "__main__":
    main()
