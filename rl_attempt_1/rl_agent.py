# rl_agent.py
#
# A tiny CNN policy + REINFORCE buffer + choose_move() adapter
# so *this* file can be imported by hex_arena just like your other bots.

from __future__ import annotations
from typing import List, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hex_env import HexEnv, Move

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels = 3, out shape 5×5
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.head  = nn.Conv2d(64, 1, kernel_size=1)     # logits per cell

        #1x1 value head for baseline (scalar)
        self.vhead = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # [B,64,1,1]
        nn.Flatten(),     # [B,64]
        nn.Linear(64, 1)
        )

    def forward(self, x):            # x: [B,3,5,5]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        logits = self.head(x).squeeze(1)  # [B,5,5]
        value = self.vhead(x)  #[B,1]
        return logits, value

class RLAgent:
    """
    Wraps PolicyNet and keeps a buffer for REINFORCE.
    call choose_move to act *and* store (state, move) for learning.
    """
    def __init__(self, lr=1e-3):
        self.net = PolicyNet().to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.saved_states: List[torch.Tensor] = []
        self.saved_actions: List[int] = []
        self.rewards: List[float] = []

    # ----- arena interface -----
    def choose_move(self, size: int, moves: List[Move],
                    rng: random.Random | None = None) -> Move:
        assert size == 5, "Only 5x5 implemented in this starter agent"
        env = HexEnv()
        env.moves = moves.copy()           # mirror current position
        state = torch.from_numpy(env._board_tensor()).unsqueeze(0).to(DEVICE)  # [1,3,5,5]

        logits, _ = self.net(state)
        logits = logits.flatten()          # 25
        legal_mask = torch.zeros(25, dtype=torch.bool, device=DEVICE)
        for (r, c) in env.legal_moves():
            legal_mask[r * 5 + c] = True
        logits[~legal_mask] = -1e9         # impossible moves → very low log-prob
        probs = F.softmax(logits, dim=0)
        m = torch.distributions.Categorical(probs)
        action_idx = m.sample()

        # save for learning
        self.saved_states.append(state)
        self.saved_actions.append(action_idx)

        r, c = divmod(action_idx.item(), 5)
        return (r, c)

    # ----- REINFORCE update after a full game -----
    def finish_episode(self, gamma=1.0):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_losses = []; value_losses = []
        for logit_state, action, ret in zip(self.saved_states,
                                            self.saved_actions, returns):
            logits, value = self.net(logit_state)
            value = value.squeeze(0)
            advantage = ret - value.detach()
            value_losses.append(F.mse_loss(value, ret))

            logits = logits.flatten()
            log_prob = F.log_softmax(logits, dim=0)[action]
            policy_losses.append(-log_prob * advantage)

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # clear buffers
        self.saved_states.clear()
        self.saved_actions.clear()
        self.rewards.clear()

# -------- convenience global instance for arena ----------
_agent_singleton: RLAgent | None = None

def choose_move(size: int, moves: List[Move],
                rng: random.Random | None = None) -> Move:
    """
    Black-box hook for hex_arena:
    >>> from rl_agent import choose_move
    """
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = RLAgent()
    return _agent_singleton.choose_move(size, moves, rng)
