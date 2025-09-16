# rl_model_numpy_9big.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import os, math
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = logits.copy()
    z[~mask] = -1e9
    if z.ndim == 1:
        zmax = z.max()
        e = np.exp(z - zmax); e[~mask] = 0.0
        return e / (e.sum() + 1e-12)
    zmax = z.max(axis=1, keepdims=True)
    e = np.exp(z - zmax); e[~mask] = 0.0
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

@dataclass
class AdamState:
    mW1: np.ndarray; vW1: np.ndarray
    mb1: np.ndarray; vb1: np.ndarray
    mW2: np.ndarray; vW2: np.ndarray
    mb2: np.ndarray; vb2: np.ndarray
    mW3: np.ndarray; vW3: np.ndarray
    mb3: np.ndarray; vb3: np.ndarray
    mW4: np.ndarray; vW4: np.ndarray
    mb4: np.ndarray; vb4: np.ndarray
    t: int = 0

class PolicyMLP9Big:
    """
    9x9 Hex policy (large):
      input  : 243  (3x9x9)
      hidden : 512 → 512 → 512 (ReLU)
      output : 81 logits
    Per-colour EMA baselines; optional entropy bonus in update().
    """
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.06, size=(243, 512)).astype(np.float32); self.b1 = np.zeros((512,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.06, size=(512, 512)).astype(np.float32); self.b2 = np.zeros((512,), dtype=np.float32)
        self.W3 = rng.normal(0, 0.06, size=(512, 512)).astype(np.float32); self.b3 = np.zeros((512,), dtype=np.float32)
        self.W4 = rng.normal(0, 0.06, size=(512,  81)).astype(np.float32); self.b4 = np.zeros(( 81,), dtype=np.float32)

        self.opt = AdamState(
            mW1=np.zeros_like(self.W1), vW1=np.zeros_like(self.W1),
            mb1=np.zeros_like(self.b1), vb1=np.zeros_like(self.b1),
            mW2=np.zeros_like(self.W2), vW2=np.zeros_like(self.W2),
            mb2=np.zeros_like(self.b2), vb2=np.zeros_like(self.b2),
            mW3=np.zeros_like(self.W3), vW3=np.zeros_like(self.W3),
            mb3=np.zeros_like(self.b3), vb3=np.zeros_like(self.b3),
            mW4=np.zeros_like(self.W4), vW4=np.zeros_like(self.W4),
            mb4=np.zeros_like(self.b4), vb4=np.zeros_like(self.b4),
            t=0
        )
        self.base_black = 0.0
        self.base_white = 0.0

    # ----- forward -----
    def _relu(self, x): return np.maximum(x, 0.0)

    def logits(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            h1 = self._relu(x @ self.W1 + self.b1)
            h2 = self._relu(h1 @ self.W2 + self.b2)
            h3 = self._relu(h2 @ self.W3 + self.b3)
            return h3 @ self.W4 + self.b4
        h1 = self._relu(x @ self.W1 + self.b1[None, :])
        h2 = self._relu(h1 @ self.W2 + self.b2[None, :])
        h3 = self._relu(h2 @ self.W3 + self.b3[None, :])
        return h3 @ self.W4 + self.b4[None, :]

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
        return int(rng.choice(81, p=p))

    # ----- update (REINFORCE + per-colour EMA baselines + entropy) -----
    def update(self, batch: List[Dict[str, Any]], lr: float = 1.5e-3,
               clip: float = 3.0, weight_decay: float = 1e-5,
               baseline_alpha: float = 0.05, entropy_coef: float = 0.01) -> Dict[str, float]:
        """
        batch items: { 'x':[243], 'mask':[81] bool, 'a':int(0..80), 'side':+1/-1, 'ret':+1/-1 }
        """
        N = len(batch)
        X   = np.stack([b['x'] for b in batch]).astype(np.float32)
        Msk = np.stack([b['mask'] for b in batch])
        A   = np.array([b['a'] for b in batch], dtype=np.int64)
        Side= np.array([b['side'] for b in batch], dtype=np.int8)
        Ret = np.array([b['ret'] for b in batch], dtype=np.float32)

        # EMA baselines per colour
        maskB = (Side == 1); maskW = (Side == -1)
        if maskB.any(): self.base_black = (1 - baseline_alpha)*self.base_black + baseline_alpha*float(Ret[maskB].mean())
        if maskW.any(): self.base_white = (1 - baseline_alpha)*self.base_white + baseline_alpha*float(Ret[maskW].mean())
        baselines = np.where(Side == 1, self.base_black, self.base_white).astype(np.float32)
        Adv = Ret - baselines

        # forward
        H1 = self._relu(X @ self.W1 + self.b1[None, :])
        H2 = self._relu(H1 @ self.W2 + self.b2[None, :])
        H3 = self._relu(H2 @ self.W3 + self.b3[None, :])
        Z  = H3 @ self.W4 + self.b4[None, :]
        P  = masked_softmax(Z, Msk)

        # losses
        logp_a = np.log(P[np.arange(N), A] + 1e-12)
        pg_loss = -np.mean(Adv * logp_a)
        entropy = float(-(P * np.log(P + 1e-12)).sum(axis=1).mean())
        loss = pg_loss - entropy_coef * entropy

        # grad wrt logits
        Gz = P.copy()
        Gz[np.arange(N), A] -= 1.0
        Gz *= (Adv / max(1, N))[:, None]  # policy grad term
        # entropy grad adds: -entropy_coef * dH/dZ; for softmax, dH/dZ = -(log P + 1) * P + ...
        # We'll implement a simple approx: subtract entropy_coef * ( - (logP + 1) * P ) = + entropy_coef * (logP + 1) * P
        if entropy_coef != 0.0:
            ent_term = (np.log(P + 1e-12) + 1.0) * P
            Gz -= entropy_coef * ent_term / max(1, N)

        # backprop
        gW4 = H3.T @ Gz
        gb4 = Gz.sum(axis=0)
        Gh3 = Gz @ self.W4.T
        Gh3 *= (H3 > 0).astype(np.float32)

        gW3 = H2.T @ Gh3
        gb3 = Gh3.sum(axis=0)
        Gh2 = Gh3 @ self.W3.T
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
            gW4 += weight_decay * self.W4

        # clip
        total_norm = math.sqrt(
            (gW1**2).sum()+(gb1**2).sum()+(gW2**2).sum()+(gb2**2).sum()+
            (gW3**2).sum()+(gb3**2).sum()+(gW4**2).sum()+(gb4**2).sum()
        )
        if total_norm > clip:
            scale = clip / (total_norm + 1e-8)
            for G in (gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4): G *= scale

        self._adam_step(gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4, lr)
        return {
            "loss": float(loss),
            "entropy": float(entropy),
            "grad_norm": float(total_norm),
            "baseline_black": float(self.base_black),
            "baseline_white": float(self.base_white),
        }

    def _adam_step(self, gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4, lr):
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
        upd(self.W4, gW4, st.mW4, st.vW4); upd(self.b4, gb4, st.mb4, st.vb4)

    # ----- save/load -----
    def save(self, path: str, step: int | None = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4,
                 mW1=self.opt.mW1, vW1=self.opt.vW1, mb1=self.opt.mb1, vb1=self.opt.vb1,
                 mW2=self.opt.mW2, vW2=self.opt.vW2, mb2=self.opt.mb2, vb2=self.opt.vb2,
                 mW3=self.opt.mW3, vW3=self.opt.vW3, mb3=self.opt.mb3, vb3=self.opt.vb3,
                 mW4=self.opt.mW4, vW4=self.opt.vW4, mb4=self.opt.mb4, vb4=self.opt.vb4,
                 t=self.opt.t,
                 base_black=self.base_black, base_white=self.base_white,
                 step=0 if step is None else step)

    @classmethod
    def load(cls, path: str) -> "PolicyMLP9Big":
        d = np.load(path, allow_pickle=False)
        files = set(d.files)
        # Native big format
        if {"W1","b1","W2","b2","W3","b3","W4","b4"} <= files:
            obj = cls(seed=0)
            obj.W1, obj.b1 = d["W1"], d["b1"]
            obj.W2, obj.b2 = d["W2"], d["b2"]
            obj.W3, obj.b3 = d["W3"], d["b3"]
            obj.W4, obj.b4 = d["W4"], d["b4"]
            obj.opt.t = int(d.get("t", 0))
            for name in ["mW1","vW1","mb1","vb1","mW2","vW2","mb2","vb2","mW3","vW3","mb3","vb3","mW4","vW4","mb4","vb4"]:
                if name in d.files: setattr(obj.opt, name, d[name])
            obj.base_black = float(d.get("base_black", 0.0))
            obj.base_white = float(d.get("base_white", 0.0))
            return obj

        # Migrate from smaller 9x9 (256→256) checkpoints: W1,b1,W2,b2,W3,b3 only
        if {"W1","b1","W2","b2","W3","b3"} <= files:
            old_W1, old_b1 = d["W1"], d["b1"]      # [243,256], [256]
            old_W2, old_b2 = d["W2"], d["b2"]      # [256,256], [256]
            old_W3, old_b3 = d["W3"], d["b3"]      # [256,81],  [81]
            obj = cls(seed=0)
            # First layer: copy into first 256 cols
            obj.W1[:, :256] = old_W1; obj.b1[:256] = old_b1
            # Second layer: identity on top-left 256×256
            obj.W2[:256, :256] = np.eye(256, dtype=np.float32)
            # Third layer: identity on top-left 256×256
            obj.W3[:256, :256] = np.eye(256, dtype=np.float32)
            # Output: copy old output weights into first 256 rows
            obj.W4[:256, :] = old_W3; obj.b4[:] = old_b3
            # fresh optimizer/baselines
            obj.base_black = float(d.get("base_black", 0.0))
            obj.base_white = float(d.get("base_white", 0.0))
            obj.opt.t = 0
            return obj

        raise KeyError(f"Unsupported checkpoint format in {path}: keys={sorted(files)}")
