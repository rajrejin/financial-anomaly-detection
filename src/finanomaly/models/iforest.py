"""Isolation Forest (scikit-learn)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from finanomaly.models.base import AnomalyModel


class IsolationForestModel(AnomalyModel):
    family = "isolation-forest"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model: IsolationForest | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, seed: int = 0, **kw: Any
    ) -> IsolationForestModel:
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=int(self.params["n_estimators"]),
            max_samples=min(int(self.params["max_samples"]), len(X)),
            max_features=float(self.params["max_features"]),
            random_state=seed,
            n_jobs=1,
        ).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return -self.model.score_samples(X)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Occlusion contribution: score(x) minus score(x with feature j set to 0, the z-score
        centre). Positive means the feature raises the anomaly score."""
        base = self.score(X)
        out = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            Xj = X.copy()
            Xj[:, j] = 0.0
            out[:, j] = base - self.score(Xj)
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({"model": self.model, "params": self.params, "seed": self.seed}, f)

    @classmethod
    def load(cls, path: Path) -> IsolationForestModel:
        with open(path / "model.pkl", "rb") as f:
            d = pickle.load(f)  # noqa: S301 - artifact produced by this project
        m = cls(**d["params"])
        m.model, m.seed = d["model"], d["seed"]
        return m

    def n_parameters(self) -> int:
        return int(self.params["n_estimators"]) * int(self.params["max_samples"])
