# rl_model_numpy_7.py
# checked for 9 artifacts
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import os, math
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over legal actions only; supports 1D or 2D input."""
    z = logits.copy()
    z[~mask] = -1e9
    if z.ndim == 1:
        zmax = z.max()
        e = np.exp(z - zmax)
        e[~mask] = 0.0
        s = e.sum()
        return e / (s + 1e-12)
    else:
        zmax = z.max(axis=1, keepdims=True)
        e = np.exp(z - zmax)
        e[~mask] = 0.0
        s = e.sum(axis=1, keepdims=True)
        return e / (s + 1e-12)

@dataclass
class AdamState:
    mW1: np.ndarray; vW1: np.ndarray
    mb1: np.ndarray; vb1: np.ndarray
    mW2: np.ndarray; vW2: np.ndarray
    mb2: np.ndarray; vb2: np.ndarray
    mW3: np.ndarray; vW3: np.ndarray
    mb3: np.ndarray; vb3: np.ndarray
    t: int = 0

class PolicyMLP7:
    """
    7x7 Hex policy:
      input  : 3*7*7 = 147
      hidden : 512 → 512
      output : 49 logits (r*7 + c)
    Uses per-colour EMA baselines to reduce variance.
    """
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.07, size=(147, 512)).astype(np.float32)
        self.b1 = np.zeros((512,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.07, size=(512, 512)).astype(np.float32)
        self.b2 = np.zeros((512,), dtype=np.float32)
        self.W3 = rng.normal(0, 0.07, size=(512, 49)).astype(np.float32)
        self.b3 = np.zeros((49,), dtype=np.float32)

        self.opt = AdamState(
            mW1=np.zeros_like(self.W1), vW1=np.zeros_like(self.W1),
            mb1=np.zeros_like(self.b1), vb1=np.zeros_like(self.b1),
            mW2=np.zeros_like(self.W2), vW2=np.zeros_like(self.W2),
            mb2=np.zeros_like(self.b2), vb2=np.zeros_like(self.b2),
            mW3=np.zeros_like(self.W3), vW3=np.zeros_like(self.W3),
            mb3=np.zeros_like(self.b3), vb3=np.zeros_like(self.b3),
            t=0
        )
        self.base_black = 0.0
        self.base_white = 0.0

    def _relu(self, x): return np.maximum(x, 0.0)

    # ----- forward -----
    def logits(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            h1 = self._relu(x @ self.W1 + self.b1)
            h2 = self._relu(h1 @ self.W2 + self.b2)
            z  = h2 @ self.W3 + self.b3
            return z
        else:
            h1 = self._relu(x @ self.W1 + self.b1[None, :])
            h2 = self._relu(h1 @ self.W2 + self.b2[None, :])
            z  = h2 @ self.W3 + self.b3[None, :]
            return z

    def greedy(self, x: np.ndarray, legal_mask: np.ndarray) -> int:
        z = self.logits(x)
        z[~legal_mask] = -1e9
        return int(np.argmax(z))

    def sample(self, x: np.ndarray, legal_mask: np.ndarray, temperature: float = 1.0,
               rng: np.random.Generator | None = None) -> int:
        z = self.logits(x)
        if temperature != 1.0:
            z = z / max(1e-6, float(temperature))
        p = masked_softmax(z, legal_mask)
        if rng is None: rng = np.random.default_rng()
        return int(rng.choice(49, p=p))

    # ----- update (REINFORCE with per-colour EMA baselines) -----
    def update(self, batch: List[Dict[str, Any]], lr: float = 2e-3,
               clip: float = 2.0, weight_decay: float = 1e-5,
               baseline_alpha: float = 0.05) -> Dict[str, float]:
        """
        batch items: { 'x':[147], 'mask':[49] bool, 'a':int(0..48), 'side':+1/-1, 'ret':+1/-1 }
        Only our agent's decisions are included.
        """
        N = len(batch)
        X   = np.stack([b['x'] for b in batch]).astype(np.float32)
        Msk = np.stack([b['mask'] for b in batch])
        A   = np.array([b['a'] for b in batch], dtype=np.int64)
        Side= np.array([b['side'] for b in batch], dtype=np.int8)
        Ret = np.array([b['ret'] for b in batch], dtype=np.float32)

        # EMA baselines per colour
        maskB = (Side == 1); maskW = (Side == -1)
        if maskB.any():
            self.base_black = (1 - baseline_alpha) * self.base_black + baseline_alpha * float(Ret[maskB].mean())
        if maskW.any():
            self.base_white = (1 - baseline_alpha) * self.base_white + baseline_alpha * float(Ret[maskW].mean())
        baselines = np.where(Side == 1, self.base_black, self.base_white).astype(np.float32)
        Adv = Ret - baselines

        # forward
        H1 = self._relu(X @ self.W1 + self.b1[None, :])
        H2 = self._relu(H1 @ self.W2 + self.b2[None, :])
        Z  = H2 @ self.W3 + self.b3[None, :]
        P  = masked_softmax(Z, Msk)

        # loss
        logp_a = np.log(P[np.arange(N), A] + 1e-12)
        loss = -np.mean(Adv * logp_a)

        # grads
        Gz = P.copy()
        Gz[np.arange(N), A] -= 1.0
        Gz *= (Adv / max(1, N))[:, None]

        gW3 = H2.T @ Gz
        gb3 = Gz.sum(axis=0)
        Gh2 = Gz @ self.W3.T
        Gh2 *= (H2 > 0).astype(np.float32)

        gW2 = H1.T @ Gh2
        gb2 = Gh2.sum(axis=0)
        Gh1 = Gh2 @ self.W2.T
        Gh1 *= (H1 > 0).astype(np.float32)

        gW1 = X.T @ Gh1
        gb1 = Gh1.sum(axis=0)

        if weight_decay > 0.0:
            gW1 += weight_decay * self.W1
            gW2 += weight_decay * self.W2
            gW3 += weight_decay * self.W3

        total_norm = math.sqrt(
            (gW1**2).sum() + (gb1**2).sum() +
            (gW2**2).sum() + (gb2**2).sum() +
            (gW3**2).sum() + (gb3**2).sum()
        )
        if total_norm > clip:
            scale = clip / (total_norm + 1e-8)
            gW1 *= scale; gb1 *= scale; gW2 *= scale; gb2 *= scale; gW3 *= scale; gb3 *= scale

        self._adam_step(gW1, gb1, gW2, gb2, gW3, gb3, lr)

        avg_entropy = float(-(P * np.log(P + 1e-12)).sum(axis=1).mean())
        return {
            "loss": float(loss),
            "entropy": avg_entropy,
            "grad_norm": float(total_norm),
            "baseline_black": float(self.base_black),
            "baseline_white": float(self.base_white),
        }

    def _adam_step(self, gW1, gb1, gW2, gb2, gW3, gb3, lr):
        b1, b2, eps = 0.9, 0.999, 1e-8
        st = self.opt; st.t += 1
        def upd(param, g, m, v):
            m[:] = b1*m + (1-b1)*g
            v[:] = b2*v + (1-b2)*(g*g)
            mhat = m / (1 - b1**st.t)
            vhat = v / (1 - b2**st.t)
            param -= lr * mhat / (np.sqrt(vhat) + eps)
        upd(self.W1, gW1, st.mW1, st.vW1); upd(self.b1, gb1, st.mb1, st.vb1)
        upd(self.W2, gW2, st.mW2, st.vW2); upd(self.b2, gb2, st.mb2, st.vb2)
        upd(self.W3, gW3, st.mW3, st.vW3); upd(self.b3, gb3, st.mb3, st.vb3)

    # ----- persistence -----
    def save(self, path: str, step: int | None = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3,
                 mW1=self.opt.mW1, vW1=self.opt.vW1, mb1=self.opt.mb1, vb1=self.opt.vb1,
                 mW2=self.opt.mW2, vW2=self.opt.vW2, mb2=self.opt.mb2, vb2=self.opt.vb2,
                 mW3=self.opt.mW3, vW3=self.opt.vW3, mb3=self.opt.mb3, vb3=self.opt.vb3,
                 t=self.opt.t,
                 base_black=self.base_black, base_white=self.base_white,
                 step=0 if step is None else step)

    @classmethod
    def load(cls, path: str) -> "PolicyMLP7":
        d = np.load(path, allow_pickle=False)
        obj = cls(seed=0)
        obj.W1, obj.b1 = d["W1"], d["b1"]
        obj.W2, obj.b2 = d["W2"], d["b2"]
        obj.W3, obj.b3 = d["W3"], d["b3"]
        obj.opt.t = int(d.get("t", 0))
        for name in ["mW1","vW1","mb1","vb1","mW2","vW2","mb2","vb2","mW3","vW3","mb3","vb3"]:
            if name in d.files: setattr(obj.opt, name, d[name])
        obj.base_black = float(d.get("base_black", 0.0))
        obj.base_white = float(d.get("base_white", 0.0))
        return obj
