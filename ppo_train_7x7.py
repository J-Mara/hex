# ppo_train_7x7.py
from __future__ import annotations
import os, csv, argparse, importlib
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo_models_7 import PolicyValueNet7, get_device, masked_softmax_stable
from ppo_buffer_7 import PPOBuffer

# Use your existing environment
from hex_env7 import HexEnv7

Move = Tuple[int, int]

def obs_and_mask_from_moves(moves: List[Move]) -> tuple[torch.Tensor, torch.Tensor]:
    env = HexEnv7(); env.moves = list(moves)
    obs = torch.from_numpy(env._obs()).float()         # [3,7,7]
    legal = torch.ones(49, dtype=torch.bool)
    for (r,c) in set(env.moves): legal[r*7 + c] = False
    return obs, legal

@torch.no_grad()
def eval_vs_module(model: PolicyValueNet7, opponent_mod: str, games: int, device: torch.device, seed: int = 0) -> float:
    opp = importlib.import_module(opponent_mod)
    wins = 0
    rl_as_black = True
    for _ in range(games):
        env = HexEnv7(); env.reset()
        while True:
            to_black = (len(env.moves) % 2 == 0)
            if (to_black and rl_as_black) or ((not to_black) and (not rl_as_black)):
                obs = torch.from_numpy(env._obs()).float().unsqueeze(0).to(device)   # [1,3,7,7]
                legal = torch.ones(1,49, dtype=torch.bool, device=device)
                for (r,c) in set(env.moves): legal[0, r*7 + c] = False
                act, _, _ = model.act(obs, legal, deterministic=True)
                a = int(act.item()); move = (a//7, a%7)
            else:
                move = opp.choose_move(7, env.moves, None)
            _, _, done, _ = env.step(move)
            if done:
                if (env.winner == 'Black') == rl_as_black: wins += 1
                break
        rl_as_black = not rl_as_black
    return wins / games

def main():
    ap = argparse.ArgumentParser(description="PPO self-play for 7x7 Hex (PyTorch)")
    ap.add_argument("--total-steps", type=int, default=2_000_000, help="Number of policy-action steps to collect")
    ap.add_argument("--batch-steps", type=int, default=20_000, help="Steps per PPO update")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gae-lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--save-every", type=int, default=20_000)
    ap.add_argument("--log-interval", type=int, default=5_000)
    ap.add_argument("--eval-games", type=int, default=800)
    ap.add_argument("--opponent1", type=str, default="random_player_c")
    ap.add_argument("--opponent2", type=str, default="two_ahead_random_player_c")
    ap.add_argument("--frozen-prob", type=float, default=0.5, help="Probability an episode uses a frozen opponent")
    ap.add_argument("--random-open-max", type=int, default=6, help="Random opening moves at episode start (0..k)")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = get_device()
    print("Using device:", device)
    os.makedirs("checkpoints7ppo", exist_ok=True)
    os.makedirs("metrics7ppo", exist_ok=True)

    model = PolicyValueNet7().to(device)
    best_path = "best7ppo_model.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded {best_path}")
    else:
        torch.save(model.state_dict(), best_path)
        print("Initialized best7ppo_model.pt")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = "metrics7ppo/metrics.csv"
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["steps","policy_loss","value_loss","entropy","wr_random","wr_op2"])
        print(f"Wrote metrics header to {metrics_path}")

    # frozen opponent (greedy) starts equal to current model/best
    frozen = PolicyValueNet7().to(device)
    frozen.load_state_dict(model.state_dict())

    # Baseline eval at step 0 so files update immediately
    wr1_0 = eval_vs_module(model, args.opponent1, max(50, args.eval_games // 8), device, seed=101)
    wr2_0 = eval_vs_module(model, args.opponent2, max(50, args.eval_games), device, seed=args.seed)
    with open(metrics_path, "a", newline="") as f:
        csv.writer(f).writerow([0, 0.0, 0.0, 0.0, wr1_0, wr2_0])
    print(f"[0] baseline wr vs {args.opponent1}: {100*wr1_0:.1f}%, vs {args.opponent2}: {100*wr2_0:.1f}%")

    total_steps = 0
    next_save = args.save_every
    next_eval = args.eval_every
    best_vs_op2 = wr2_0

    buffer = PPOBuffer(device)

    def finish_episode_and_push(episode_steps: List[dict], winner: Optional[str]):
        """Assign +1/-1 rewards to OUR policy moves in the episode and push to buffer."""
        if winner not in ("Black", "White"):
            # draw or full board: give 0 rewards
            for i, st in enumerate(episode_steps):
                buffer.add_step(st["obs"], st["mask"], st["action"], st["logprob"], st["value"],
                                reward=0.0, done=(i == len(episode_steps)-1))
            return
        win_black = (winner == "Black")
        for i, st in enumerate(episode_steps):
            # side +1 for Black, -1 for White
            reward = 1.0 if (st["side_black"] == win_black) else -1.0
            buffer.add_step(st["obs"], st["mask"], st["action"], st["logprob"], st["value"],
                            reward=reward, done=(i == len(episode_steps)-1))

    # ---- main collection & update loop ----
    while total_steps < args.total_steps:
        # 1) Collect a batch of self-play action steps
        buffer.clear()
        while len(buffer) < args.batch_steps:
            env = HexEnv7(); env.reset()
            # Optional random opening
            k = int(np.random.randint(0, args.random_open_max + 1))
            for _ in range(k):
                legals = env.legal_moves()
                if not legals: break
                env.step(legals[np.random.randint(0, len(legals))])

            # Decide episode opponent mode
            use_frozen = (np.random.rand() < args.frozen_prob)
            episode_steps: List[dict] = []
            rl_as_black = bool(np.random.randint(0, 2)) if use_frozen else True  # if mirror, we'll record every move

            while True:
                to_black = (len(env.moves) % 2 == 0)

                # Whose policy acts?
                if (not use_frozen) or (use_frozen and ((to_black and rl_as_black) or (not to_black and not rl_as_black))):
                    # OUR policy acts (record transition)
                    obs_np = env._obs()                      # [3,7,7]
                    obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
                    mask = torch.ones(1,49, dtype=torch.bool, device=device)
                    for (r,c) in set(env.moves): mask[0, r*7+c] = False
                    # action, logprob, value = model.act(obs, mask, deterministic=False)
                    # a = int(action.item()); move = (a//7, a%7)

                    action, logprob, value = model.act(obs, mask, deterministic=False)
                    a = int(action.item())
                    if not bool(mask[0, a]):
                        # snap to the best legal move
                        a = int(torch.nonzero(mask[0], as_tuple=False)[0].item())
                        # recompute logprob for the snapped action
                        logits, _ = model(obs)
                        probs = masked_softmax_stable(logits, mask)
                        logprob = torch.log(probs[0, a].clamp_min(1e-12))
                    move = (a//7, a%7)

                    # if(move in set(env.moves)):
                    #     raise Exception ("wait, that's illegal")
                    _, _, done, _ = env.step(move)
                    episode_steps.append({
                        "obs": obs.squeeze(0),
                        "mask": mask.squeeze(0),
                        "action": action.squeeze(0),
                        "logprob": logprob.squeeze(0),
                        "value": value.squeeze(0),
                        "side_black": to_black,  # side at time of decision
                    })
                else:
                    # Frozen opponent acts (greedy, not stored)
                    obs = torch.from_numpy(env._obs()).float().unsqueeze(0).to(device)
                    mask = torch.ones(1,49, dtype=torch.bool, device=device)
                    for (r,c) in set(env.moves): mask[0, r*7+c] = False
                    act, _, _ = frozen.act(obs, mask, deterministic=True)
                    a = int(act.item()); move = (a//7, a%7)
                    _, _, done, _ = env.step(move)

                if done:
                    finish_episode_and_push(episode_steps, env.winner)
                    break

            # heartbeat while collecting
            if len(buffer) % max(1, args.log_interval // 2) == 0:
                print(f"collecting… {len(buffer)}/{args.batch_steps}", end="\r")

        total_steps += len(buffer)

        # 2) Compute GAE / returns
        tensors = buffer.compute_advantages(gamma=args.gamma, lam=args.gae_lam)

        # 3) PPO update
        policy_losses, value_losses, entropies = [], [], []
        for _ in range(args.epochs):
            for mb in PPOBuffer.iterate_minibatches(tensors, args.minibatch_size):
                new_logprob, entropy, values = model.evaluate_actions(
                    mb["obs"], mb["mask"], mb["actions"]
                )
                ratio = torch.exp(new_logprob - mb["old_logprobs"])
                surr1 = ratio * mb["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * mb["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((mb["returns"] - values) ** 2).mean()

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        # 4) Save for every crossed threshold
        while total_steps >= next_save:
            ckpt = f"checkpoints7ppo/ppo7_step{next_save:08d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"[{total_steps}] saved {ckpt}")
            next_save += args.save_every

        # 5) Evaluate for every crossed threshold
        while total_steps >= next_eval:
            wr1 = eval_vs_module(model, args.opponent1, args.eval_games, device, seed=123)
            wr2 = eval_vs_module(model, args.opponent2, args.eval_games, device, seed=456)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    next_eval,  # log at scheduled eval step
                    np.mean(policy_losses) if policy_losses else 0.0,
                    np.mean(value_losses) if value_losses else 0.0,
                    np.mean(entropies)    if entropies    else 0.0,
                    wr1, wr2
                ])
            print(f"[{next_eval}] wr vs {args.opponent1}: {100*wr1:.1f}%, vs {args.opponent2}: {100*wr2:.1f}%")
            if wr2 > best_vs_op2:
                best_vs_op2 = wr2
                torch.save(model.state_dict(), best_path)
                frozen.load_state_dict(model.state_dict())
                print(f"  ↑ new best7ppo_model.pt ({100*wr2:.1f}% vs {args.opponent2})")
            next_eval += args.eval_every

if __name__ == "__main__":
    main()
