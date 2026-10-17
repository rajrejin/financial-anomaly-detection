"""Configuration loading: one YAML file, with environment overrides."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

ROOT = Path(os.environ.get("FINANOMALY_ROOT", Path(__file__).resolve().parents[2]))


class Windows(BaseModel):
    label_days: int
    detect_before: int
    detect_after: int
    buffer_days: int


class Splits(BaseModel):
    fit_start: str
    val_start: str
    val_end: str
    test_start: str


class Alarm(BaseModel):
    budget_per_year: float
    merge_gap_days: int


class Features(BaseModel):
    vol_windows: list[int]
    z_window: int
    corr_window: int
    drawdown_window: int


class DataCfg(BaseModel):
    start: str
    max_ffill_days: int
    bundesbank_base: str
    ecb_base: str


class Paths(BaseModel):
    raw: str
    processed: str
    results: str
    events: str


class MlflowCfg(BaseModel):
    tracking_uri: str
    artifact_root: str


class Serving(BaseModel):
    model_dir: str


class Config(BaseModel):
    seed: int
    paths: Paths
    data: DataCfg
    windows: Windows
    splits: Splits
    alarm: Alarm
    features: Features
    mlflow: MlflowCfg
    serving: Serving

    def path(self, name: str) -> Path:
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else ROOT / p


def load_config(path: Path | None = None) -> Config:
    path = path or Path(os.environ.get("FINANOMALY_CONFIG", ROOT / "configs" / "default.yaml"))
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "FINANOMALY_SEED" in os.environ:
        raw["seed"] = int(os.environ["FINANOMALY_SEED"])
    if "FINANOMALY_MLFLOW_URI" in os.environ:
        raw["mlflow"]["tracking_uri"] = os.environ["FINANOMALY_MLFLOW_URI"]
    return Config(**raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
