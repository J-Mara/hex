# ppo_models_7.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- masking patch

def masked_softmax_stable(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable masked softmax.
    logits: [B, A], legal_mask: [B, A] (bool)
    Returns probs with illegal actions exactly zero and rows normalized to 1.
    """
    # Large negative for illegal, but keep it finite and then re-mask after softmax.
    masked = torch.where(legal_mask, logits, logits.new_full(logits.shape, -1e9))
    z = masked - masked.max(dim=-1, keepdim=True).values
    e = torch.exp(z)
    e = e * legal_mask.float()
    denom = e.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return e / denom

def pick_greedy_masked(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(legal_mask, logits, logits.new_full(logits.shape, -1e9))
    return masked.argmax(dim=-1)

# ---- small utils ----

def get_device() -> torch.device:
    try:
        #if torch.backends.mps.is_available():
        if torch.cuda.is_available():
            return torch.device("cuda")
            #return torch.device("cpu")
    except Exception:
        pass
    return torch.device("cpu")

# def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
#     """Set illegal moves to a large negative number so softmax ignores them."""
#     # logits: [B, 49], legal_mask: [B, 49] (bool)
#     neg_inf = torch.finfo(logits.dtype).min
#     out = logits.clone()
#     out[~legal_mask] = neg_inf
#     return out

# ---- policy-value net ----

class PolicyValueNet7(nn.Module):
    """
    Input:  (B, 3, 7, 7)  planes: [black, white, to_move_is_black]
    Policy: (B, 49) logits for actions a = r*7 + c (masked for illegals)
    Value:  (B, 1)  scalar V(s)
    """
    def __init__(self):
        super().__init__()
        C = 3
        # lightweight conv encoder
        self.enc = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head_dim = 128 * 7 * 7
        self.pi_head = nn.Sequential(
            nn.Linear(self.head_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 49),
        )
        self.v_head = nn.Sequential(
            nn.Linear(self.head_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs: [B,3,7,7]
        h = self.enc(obs)
        h = h.view(h.size(0), -1)
        logits = self.pi_head(h)      # [B,49]
        value  = self.v_head(h)       # [B,1]
        return logits, value

    @torch.no_grad()
    # def act(self, obs: torch.Tensor, legal_mask: torch.Tensor, deterministic: bool = False):
    #     logits, value = self.forward(obs)
    #     logits = masked_logits(logits, legal_mask)
    #     if deterministic:
    #         # Greedy: argmax over legal
    #         action = torch.argmax(logits, dim=-1)
    #         # logprob from masked softmax
    #         probs = F.softmax(logits, dim=-1)
    #         logprob = torch.log(torch.gather(probs, 1, action.view(-1,1)).clamp_min(1e-12)).squeeze(1)
    #         return action, logprob, value.squeeze(1)
    #     else:
    #         probs = F.softmax(logits, dim=-1)
    #         # sampling with masked probs (illegal are 0 after masked softmax)
    #         dist = torch.distributions.Categorical(probs=probs)
    #         action = dist.sample()
    #         logprob = dist.log_prob(action)
    #         return action, logprob, value.squeeze(1)
    #
    # def evaluate_actions(self, obs: torch.Tensor, legal_mask: torch.Tensor, actions: torch.Tensor):
    #     logits, value = self.forward(obs)
    #     logits = masked_logits(logits, legal_mask)
    #     probs = F.softmax(logits, dim=-1)
    #     dist = torch.distributions.Categorical(probs=probs)
    #     logprob = dist.log_prob(actions)
    #     entropy = dist.entropy()
    #     return logprob, entropy, value.squeeze(1)

    def act(self, obs: torch.Tensor, legal_mask: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        if deterministic:
            action = pick_greedy_masked(logits, legal_mask)
            probs = masked_softmax_stable(logits, legal_mask)
            logprob = torch.log(torch.gather(probs, 1, action.view(-1,1)).clamp_min(1e-12)).squeeze(1)
        else:
            probs = masked_softmax_stable(logits, legal_mask)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
        # belt & suspenders: if anything went wrong, force a legal action
        illegal = ~legal_mask.gather(1, action.view(-1,1)).squeeze(1)
        if illegal.any():
            fix = legal_mask.float().argmax(dim=-1)  # first legal
            action = torch.where(illegal, fix, action)
            # recompute logprob for fixed actions
            probs = masked_softmax_stable(logits, legal_mask)
            logprob = torch.log(torch.gather(probs, 1, action.view(-1,1)).clamp_min(1e-12)).squeeze(1)
        return action, logprob, value.squeeze(1)

    def evaluate_actions(self, obs: torch.Tensor, legal_mask: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        probs = masked_softmax_stable(logits, legal_mask)
        # As above, if an action is illegal due to any data bug, clamp it to a legal index
        illegal = ~legal_mask.gather(1, actions.view(-1,1)).squeeze(1)
        if illegal.any():
            fix = legal_mask.float().argmax(dim=-1)
            actions = torch.where(illegal, fix, actions)
        dist = torch.distributions.Categorical(probs=probs)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value.squeeze(1)
