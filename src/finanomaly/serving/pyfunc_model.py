"""MLflow pyfunc wrapper: feature DataFrame in, score and alarm flag out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from finanomaly.models.base import AnomalyModel


def model_class(family: str) -> type[AnomalyModel]:
    """Resolve the model class lazily so serving only imports what the model needs."""
    if family == "isolation_forest":
        from finanomaly.models.iforest import IsolationForestModel

        return IsolationForestModel
    if family == "autoencoder":
        from finanomaly.models.autoencoder import AutoencoderModel

        return AutoencoderModel
    from finanomaly.models.trees import LightGBMModel, XGBoostModel

    return {"lightgbm": LightGBMModel, "xgboost": XGBoostModel}[family]


def save_model_dir(model: AnomalyModel, path: Path, meta: dict[str, Any]) -> None:
    model.save(path / "model")
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


class AnomalyPyfunc(PythonModel):
    """Input columns are the trailing z-scored features (names in meta.json)."""

    def load_context(self, context: Any) -> None:
        d = Path(context.artifacts["model_dir"])
        self.meta = json.loads((d / "meta.json").read_text())
        self.model = model_class(self.meta["family"]).load(d / "model")
        self.threshold = float(self.meta["threshold"])
        self.features: list[str] = list(self.meta["features"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        X = model_input[self.features].to_numpy(dtype=float)
        score = self.model.score(X)
        return pd.DataFrame({"score": score, "alarm": (score > self.threshold).astype(int)})

    def explain(self, X: np.ndarray) -> np.ndarray:
        return self.model.explain(X)
