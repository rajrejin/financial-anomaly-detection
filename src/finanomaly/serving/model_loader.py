"""Load an exported model directory. Refuses a missing or inconsistent manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from finanomaly.config import Config, load_config
from finanomaly.features import build_features
from finanomaly.serving.export import file_checksums


class ModelLoadError(RuntimeError):
    pass


@dataclass
class Bundle:
    manifest: dict[str, Any]
    feature_config: dict[str, Any]
    pyfunc: Any  # the unwrapped AnomalyPyfunc
    cfg: Config
    table: pd.DataFrame
    z: pd.DataFrame
    valid: np.ndarray
    status: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"])

    @property
    def features(self) -> list[str]:
        return list(self.feature_config["features"])


def verify_manifest(root: Path) -> dict[str, Any]:
    mf = root / "manifest.json"
    if not mf.exists():
        raise ModelLoadError("manifest.json missing")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    for key in ("model_name", "model_version", "run_id", "threshold", "files", "data_hash"):
        if key not in manifest:
            raise ModelLoadError(f"manifest field missing: {key}")
    if not (root / "feature_config.json").exists() or not (root / "mlflow_model").exists():
        raise ModelLoadError("model files missing")
    actual = file_checksums(root)
    if actual != manifest["files"]:
        raise ModelLoadError("model files do not match the manifest checksums")
    return manifest


def load_table(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    hist = root / "history"
    table = pd.read_parquet(hist / "daily.parquet")
    status = {}
    if (hist / "status.json").exists():
        status = json.loads((hist / "status.json").read_text())
    return table, status


def load_bundle(
    root: Path, cfg: Config | None = None, mlflow_model_dir: Path | None = None
) -> Bundle:
    cfg = cfg or load_config()
    root = Path(root)
    manifest = verify_manifest(root)
    feature_config = json.loads((root / "feature_config.json").read_text())
    pyfunc = mlflow.pyfunc.load_model(str(mlflow_model_dir or root / "mlflow_model"))
    inner = pyfunc.unwrap_python_model()
    table, status = load_table(root)
    _, z = build_features(table, cfg)
    valid = z.notna().all(axis=1).to_numpy()
    if list(z.columns) != list(feature_config["features"]):
        raise ModelLoadError("feature set does not match the model")
    return Bundle(manifest, feature_config, inner, cfg, table, z, valid, status)
