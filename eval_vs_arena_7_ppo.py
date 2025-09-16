# eval_vs_arena_7_ppo.py
import argparse, importlib, os
import torch
from typing import Tuple
from ppo_models_7 import PolicyValueNet7, get_device
from hex_env7 import HexEnv7

Move = Tuple[int, int]

@torch.no_grad()
def play_match(model: PolicyValueNet7, opponent_mod: str, games: int, device: torch.device) -> float:
    opp = importlib.import_module(opponent_mod)
    wins = 0; rl_as_black = True
    for _ in range(games):
        env = HexEnv7(); env.reset()
        while True:
            to_black = (len(env.moves) % 2 == 0)
            if (to_black and rl_as_black) or ((not to_black) and (not rl_as_black)):
                obs = torch.from_numpy(env._obs()).float().unsqueeze(0).to(device)
                legal = torch.ones(1,49, dtype=torch.bool, device=device)
                for (r,c) in set(env.moves): legal[0, r*7+c] = False
                logits, _ = model(obs)
                neg_inf = torch.finfo(logits.dtype).min
                logits[~legal] = neg_inf
                a = int(torch.argmax(logits, dim=-1).item()); move = (a//7, a%7)
            else:
                move = opp.choose_move(7, env.moves, None)
            _, _, done, _ = env.step(move)
            if done:
                if (env.winner == 'Black') == rl_as_black: wins += 1
                break
        rl_as_black = not rl_as_black
    return wins / games

def main():
    ap = argparse.ArgumentParser(description="Evaluate 7x7 PPO checkpoint vs any opponent")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--games", type=int, default=1000)
    args = ap.parse_args()

    device = get_device()
    model = PolicyValueNet7().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    wr = play_match(model, args.opponent, args.games, device)
    print(f"Win-rate vs {args.opponent}: {100*wr:.2f}% over {args.games} games")

if __name__ == "__main__":
    main()
