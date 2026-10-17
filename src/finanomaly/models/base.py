"""Common model interface: fit, score, explain (plus save and load for packaging)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class AnomalyModel(ABC):
    """Higher score means more anomalous. Inputs are trailing z-scored feature matrices."""

    family: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.seed = 0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None, seed: int = 0, **kw: Any) -> Any:
        """Fit on rows of X (and labels y for supervised families)."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per row."""

    @abstractmethod
    def explain(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contribution matrix of shape (n, d) for the rows of X."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AnomalyModel: ...

    @abstractmethod
    def n_parameters(self) -> int:
        """Rough model size used for the 'simplicity' tie-break."""
