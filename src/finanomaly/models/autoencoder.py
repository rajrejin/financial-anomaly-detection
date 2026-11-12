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


def _build(n_features: int, hidden: list[int], bottleneck: int, dropout: float) -> nn.Module:
    """Encoder squeezes the input down to `bottleneck` numbers, decoder mirrors it back up."""
    encoder: list[nn.Module] = []
    width = n_features
    for size in hidden:
        encoder += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
        width = size
    encoder += [nn.Linear(width, bottleneck)]

    decoder: list[nn.Module] = []
    width = bottleneck
    for size in reversed(hidden):
        decoder += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
        width = size
    decoder += [nn.Linear(width, n_features)]
    return nn.Sequential(*encoder, *decoder)


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
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
        )
        loss_fn = nn.MSELoss()
        train = torch.tensor(X, dtype=torch.float32)
        val = torch.tensor(X_val, dtype=torch.float32)

        best_loss, best_weights, epochs_without_improvement = float("inf"), None, 0
        for epoch in range(MAX_EPOCHS):
            # One pass over the training rows in shuffled mini-batches. The target is the
            # input itself: the network is rewarded for reproducing normal days.
            self.net.train()
            order = rng.permutation(len(train))
            for start in range(0, len(order), BATCH):
                batch = train[order[start : start + BATCH]]
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                optimizer.step()

            # Early stopping: keep the weights with the lowest validation loss and stop once
            # it has not improved for PATIENCE epochs.
            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(val), val))
            self.epochs_run = epoch + 1
            if val_loss < best_loss - 1e-6:
                best_loss, epochs_without_improvement = val_loss, 0
                best_weights = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    break
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
        self.best_val_loss = best_loss
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
        errors = self._errors(X)
        total = errors.sum(axis=1, keepdims=True)
        return errors / np.where(total > 0, total, 1.0)

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
