"""Fully connected autoencoder in PyTorch. Score = mean squared reconstruction error."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from finanomaly.models.base import AnomalyModel

MAX_EPOCHS = 200
PATIENCE = 15
BATCH = 128


def _build(d: int, hidden: list[int], bottleneck: int, dropout: float) -> nn.Module:
    enc: list[nn.Module] = []
    prev = d
    for h in hidden:
        enc += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    enc += [nn.Linear(prev, bottleneck)]
    dec: list[nn.Module] = []
    prev = bottleneck
    for h in reversed(hidden):
        dec += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    dec += [nn.Linear(prev, d)]
    return nn.Sequential(*enc, *dec)


class AutoencoderModel(AnomalyModel):
    family = "autoencoder"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.net: nn.Module | None = None
        self.d = 0
        self.epochs_run = 0
        self.best_val_loss = float("nan")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        seed: int = 0,
        X_val: np.ndarray | None = None,
        **kw: Any,
    ) -> AutoencoderModel:
        """Train on X; early stopping on X_val (a later, non-event period). If X_val is None the
        last 20 percent of X (time ordered) is held out."""
        self.seed = seed
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        rng = np.random.default_rng(seed)
        if X_val is None:
            cut = int(len(X) * 0.8)
            X, X_val = X[:cut], X[cut:]
        self.d = X.shape[1]
        p = self.params
        self.net = _build(self.d, list(p["hidden"]), int(p["bottleneck"]), float(p["dropout"]))
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(MAX_EPOCHS):
            self.net.train()
            perm = rng.permutation(len(Xt))
            for i in range(0, len(perm), BATCH):
                b = Xt[perm[i : i + BATCH]]
                opt.zero_grad()
                loss = loss_fn(self.net(b), b)
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(loss_fn(self.net(Xv), Xv))
            self.epochs_run = epoch + 1
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss = best
        self.net.eval()
        return self

    def _errors(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            return ((self.net(Xt) - Xt) ** 2).numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._errors(X).mean(axis=1)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Share of the squared reconstruction error carried by each feature (rows sum to 1)."""
        e = self._errors(X)
        tot = e.sum(axis=1, keepdims=True)
        return e / np.where(tot > 0, tot, 1.0)

    def save(self, path: Path) -> None:
        assert self.net is not None
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        meta = {"params": self.params, "d": self.d, "seed": self.seed}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> AutoencoderModel:
        meta = json.loads((path / "meta.json").read_text())
        m = cls(**meta["params"])
        m.d, m.seed = meta["d"], meta["seed"]
        p = m.params
        m.net = _build(m.d, list(p["hidden"]), int(p["bottleneck"]), float(p["dropout"]))
        m.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        m.net.eval()
        return m

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters()) if self.net is not None else 0
