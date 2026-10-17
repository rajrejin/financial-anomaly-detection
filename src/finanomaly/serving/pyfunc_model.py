"""MLflow pyfunc wrapper: feature DataFrame in, score and alarm flag out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from finanomaly.models.autoencoder import AutoencoderModel
from finanomaly.models.base import AnomalyModel
from finanomaly.models.iforest import IsolationForestModel
from finanomaly.models.trees import LightGBMModel, XGBoostModel

CLASSES: dict[str, type[AnomalyModel]] = {
    "isolation_forest": IsolationForestModel,
    "autoencoder": AutoencoderModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
}


def save_model_dir(model: AnomalyModel, path: Path, meta: dict[str, Any]) -> None:
    model.save(path / "model")
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


class AnomalyPyfunc(PythonModel):
    """Input columns are the trailing z-scored features (names in meta.json)."""

    def load_context(self, context: Any) -> None:
        d = Path(context.artifacts["model_dir"])
        self.meta = json.loads((d / "meta.json").read_text())
        self.model = CLASSES[self.meta["family"]].load(d / "model")
        self.threshold = float(self.meta["threshold"])
        self.features: list[str] = list(self.meta["features"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        X = model_input[self.features].to_numpy(dtype=float)
        score = self.model.score(X)
        return pd.DataFrame({"score": score, "alarm": (score > self.threshold).astype(int)})

    def explain(self, X: np.ndarray) -> np.ndarray:
        return self.model.explain(X)
