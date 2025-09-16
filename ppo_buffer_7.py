# ppo_buffer_7.py
from __future__ import annotations
from typing import List, Dict, Iterator
import torch

class PPOBuffer:
    """
    Stores transitions where OUR policy acted. For terminal Hex, we assign +1/-1
    per step after an episode ends, based on the eventual winner and the side to move.
    Then we compute GAE over those rewards.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.data: Dict[str, List[torch.Tensor]] = {k: [] for k in [
            "obs", "mask", "actions", "logprobs", "values", "rewards", "dones"
        ]}

    def __len__(self): return len(self.data["actions"])

    def add_step(self, obs, mask, action, logprob, value, reward, done):
        self.data["obs"].append(obs.detach())
        self.data["mask"].append(mask.detach())
        self.data["actions"].append(action.detach())
        self.data["logprobs"].append(logprob.detach())
        self.data["values"].append(value.detach())
        self.data["rewards"].append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
        self.data["dones"].append(torch.as_tensor(done, dtype=torch.bool, device=self.device))

    def to_tensors(self):
        out = {}
        for k, lst in self.data.items():
            out[k] = torch.stack(lst, dim=0).to(self.device)
        return out

    def clear(self):
        for k in self.data:
            self.data[k].clear()

    def compute_advantages(self, gamma: float = 1.0, lam: float = 0.95):
        """Return dict with obs, mask, actions, old_logprobs, returns, advantages, values."""
        T = len(self)
        batch = self.to_tensors()
        rewards = batch["rewards"]          # [T]
        values  = batch["values"]           # [T]
        dones   = batch["dones"]            # [T]
        adv = torch.zeros_like(rewards)
        lastgaelam = 0.0
        next_value = torch.tensor(0.0, device=self.device)
        for t in reversed(range(T)):
            nonterminal = (~dones[t]).float()
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        ret = adv + values
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": batch["obs"],
            "mask": batch["mask"],
            "actions": batch["actions"],
            "old_logprobs": batch["logprobs"],
            "returns": ret,
            "advantages": adv,
            "values": values,
        }

    @staticmethod
    def iterate_minibatches(tensors: Dict[str, torch.Tensor], mb_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        N = tensors["actions"].size(0)
        idx = torch.randperm(N)
        for i in range(0, N, mb_size):
            sel = idx[i:i+mb_size]
            yield {k: v[sel] for k, v in tensors.items()}
