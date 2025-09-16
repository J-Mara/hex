# train_hex_rl.py
#
# Self-play training script.
#   * plays N self-play games
#   * every eval_interval games, measures win-rate vs random & two_ahead players
#   * logs everything to TensorBoard

import random
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from hex_env import HexEnv
from rl_agent import RLAgent
import random_player, two_ahead_random_player          # your earlier bots

from hex_arena import run_match, load_player
from contextlib import redirect_stdout
import io

import os
os.environ["TORCH_LOGS"] = "none"

SAVE_DIR = Path("runs_hex_rl")
SAVE_DIR.mkdir(exist_ok=True)
writer = SummaryWriter(SAVE_DIR/(f"run_{int(time.time())}"))

def play_selfplay_game(agent: RLAgent):
    env = HexEnv()
    state = env.reset()
    while True:
        move = agent.choose_move(env.SIZE, env.moves)
        _, reward, done, _ = env.step(move)
        agent.rewards.append(reward)
        if done:
            # final reward is +1 (win) for the player who *just moved*,
            # but from the *other* player's perspective it's −1.
            # REINFORCE buffer already recorded +1/−1 properly because env.step
            # gave reward from current player's POV.
            agent.finish_episode()
            return reward > 0   # True if current player won

# ----------- new evaluate() helper ----------------

def evaluate(agent_modname: str, opponent_modname: str,
             games: int = 100) -> float:
    """
    Returns win-rate of agent vs opponent across `games`, alternating colours.
    Uses run_match's printed summary rather than re-implementing the loop.
    """
    f = io.StringIO()
    with redirect_stdout(f):        # silence arena prints
        run_match(
            size=5,
            games=games,
            player1_mod=load_player(agent_modname),
            player2_mod=load_player(opponent_modname),
            mode="alternate",
            seed=12345,
        )
    text = f.getvalue()
    # Parse the line "Player1 wins     : 48 (48.0%)"
    for line in text.splitlines():
        if line.startswith("Player1 wins"):
            pct = float(line.split("(")[1].split("%")[0])
            return pct / 100.0
    raise RuntimeError("Could not parse arena output")


def main():
    random.seed(0)
    torch.manual_seed(0)

    agent = RLAgent(lr=3e-4)
    TOTAL_GAMES = 5000
    EVAL_INTERVAL = 250

    for game_idx in range(1, TOTAL_GAMES + 1):
        win = play_selfplay_game(agent)

        if game_idx % 10 == 0:
            writer.add_scalar("selfplay/win_as_current_player", int(win), game_idx)

        if game_idx % EVAL_INTERVAL == 0:
            torch.save(agent.net.state_dict(), f"checkpoint_{game_idx}.pth")
            win_vs_rand   = evaluate("rl_agent", "random_player", 100)
            win_vs_two_ahead = evaluate("rl_agent", "two_ahead_random_player", 100)
            writer.add_scalar("eval/winrate_vs_random", win_vs_rand, game_idx)
            writer.add_scalar("eval/winrate_vs_two_ahead", win_vs_two_ahead, game_idx)
            print(f"[{game_idx}] win% vs random: {100*win_vs_rand:.1f} "
                  f"vs two_ahead: {100*win_vs_two_ahead:.1f}")

    writer.close()

if __name__ == "__main__":
    main()
