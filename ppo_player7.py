# ppo_player7.py
import os
from typing import List, Tuple
import torch
import torch.nn.functional as F
from ppo_models_7 import PolicyValueNet7, get_device
from hex_env7 import HexEnv7

Move = Tuple[int, int]
_MODEL = None
_DEVICE = None

def _load_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    _DEVICE = get_device()
    path = os.environ.get("PPO7_WEIGHTS", "best7ppo_model.pt")
    model = PolicyValueNet7().to(_DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=_DEVICE))
    model.eval()
    _MODEL = model
    return _MODEL

# @torch.no_grad()
# def choose_move(size: int, moves: List[Move], rng=None) -> Move:
#     if size != 7:
#         raise ValueError("ppo_player7 supports size=7 only.")
#     model = _load_model()
#     env = HexEnv7(); env.moves = list(moves)
#     obs = torch.from_numpy(env._obs()).float().unsqueeze(0).to(_DEVICE)   # [1,3,7,7]
#     legal = torch.ones(1,49, dtype=torch.bool, device=_DEVICE)
#     for (r,c) in set(moves): legal[0, r*7 + c] = False
#     logits, _ = model(obs)
#     neg_inf = torch.finfo(logits.dtype).min
#     logits[~legal] = neg_inf
#     a = int(torch.argmax(logits, dim=-1).item())
#     return (a // 7, a % 7)

@torch.no_grad()
def choose_move(size, moves, rng=None):
    ...
    logits, _ = model(obs)
    legal = torch.ones(1,49, dtype=torch.bool, device=_DEVICE)
    for (r,c) in set(moves): legal[0, r*7+c] = False
    # robust greedy
    masked = torch.where(legal, logits, logits.new_full(logits.shape, -1e9))
    a = int(masked.argmax(dim=-1).item())
    if not bool(legal[0, a]):  # ultra-rare fallback
        a = int(torch.nonzero(legal[0], as_tuple=False)[0].item())
    return (a // 7, a % 7)
