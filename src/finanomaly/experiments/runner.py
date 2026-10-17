"""Fit and evaluate one configuration (no MLflow here; see tracking.py)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss

from finanomaly.config import ROOT
from finanomaly.context import Context
from finanomaly.evaluation.report import calibrate_and_eval
from finanomaly.events import mask_from
from finanomaly.models.autoencoder import AutoencoderModel
from finanomaly.models.base import AnomalyModel
from finanomaly.models.iforest import IsolationForestModel
from finanomaly.models.trees import LightGBMModel, XGBoostModel

FAMILIES: dict[str, type[AnomalyModel]] = {
    "isolation_forest": IsolationForestModel,
    "autoencoder": AutoencoderModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
}
SUPERVISED = {"lightgbm", "xgboost"}
REGISTRY_NAMES = {
    "isolation_forest": "anomaly-isolation-forest",
    "autoencoder": "anomaly-autoencoder",
    "lightgbm": "anomaly-lightgbm",
}
ES_YEARS = 5


def load_experiments(path: Path | None = None) -> dict[str, Any]:
    env = os.environ.get("FINANOMALY_EXPERIMENTS")
    p = path or (Path(env) if env else ROOT / "configs" / "experiments.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


@dataclass
class StageResult:
    family: str
    config_idx: int
    params: dict[str, Any]
    seed: int
    cutoff: str
    model: AnomalyModel
    scores: np.ndarray
    threshold: float
    val_metrics: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


def _training_rows(ctx: Context, cutoff: pd.Timestamp) -> np.ndarray:
    idx = ctx.index.tz_localize(None)
    return np.asarray(idx < cutoff) & ctx.valid


def fit_model(
    ctx: Context,
    family: str,
    params: dict[str, Any],
    seed: int,
    cutoff: pd.Timestamp,
    train_start: pd.Timestamp | None = None,
) -> tuple[AnomalyModel, np.ndarray]:
    """Fit on data strictly before `cutoff`. Returns (model, full-length scores, NaN if invalid)."""
    X = ctx.z.to_numpy(dtype=float)
    rows = _training_rows(ctx, cutoff)
    if train_start is not None:
        rows = rows & np.asarray(ctx.index.tz_localize(None) >= train_start)
    model = FAMILIES[family](**params)
    usable = [w for w in ctx.wins if w.event.onset < cutoff]
    if family in SUPERVISED:
        y = np.zeros(ctx.n, dtype=int)
        for w in usable:
            y[w.label[0] : w.label[1]] = 1
        idx = ctx.index.tz_localize(None)
        es_start = cutoff - pd.DateOffset(years=ES_YEARS)
        if train_start is not None and es_start <= train_start:
            es_start = train_start + (cutoff - train_start) / 2  # shorter training period
        tr = rows & np.asarray(idx < es_start)
        es = rows & np.asarray(idx >= es_start)
        model.fit(X[tr], y[tr], seed=seed, X_es=X[es], y_es=y[es])
    else:
        event_days = mask_from(usable, ctx.n, "detect") | mask_from(usable, ctx.n, "buffer")
        model.fit(X[rows & ~event_days], None, seed=seed)
    scores = np.full(ctx.n, np.nan)
    scores[ctx.valid] = model.score(X[ctx.valid])
    return model, scores


def calibration_metrics(ctx: Context, scores: np.ndarray, seg: tuple[int, int]) -> dict:
    m = np.zeros(ctx.n, dtype=bool)
    m[seg[0] : seg[1]] = True
    m &= ~np.isnan(scores)
    p = np.clip(scores[m], 1e-6, 1 - 1e-6)
    y = ctx.y[m]
    bins = np.linspace(0, 1, 11)
    ece, table = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
            table.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(sel.sum()),
                    "mean_p": float(p[sel].mean()),
                    "event_rate": float(y[sel].mean()),
                }
            )
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "reliability": table,
    }


def run_val_stage(
    ctx: Context,
    family: str,
    config_idx: int,
    params: dict[str, Any],
    seed: int,
    train_start: pd.Timestamp | None = None,
) -> StageResult:
    cutoff = pd.Timestamp(ctx.cfg.splits.val_start)
    model, scores = fit_model(ctx, family, params, seed, cutoff, train_start)
    thr, res = calibrate_and_eval(ctx, scores, ctx.val_seg)
    extra: dict[str, Any] = {}
    if family in SUPERVISED:
        extra["calibration"] = calibration_metrics(ctx, scores, ctx.val_seg)
    return StageResult(
        family, config_idx, params, seed, str(cutoff.date()), model, scores, thr, res, extra
    )


def run_final_stage(
    ctx: Context,
    family: str,
    config_idx: int,
    params: dict[str, Any],
    seed: int,
    val_threshold: float | None,
    train_start: pd.Timestamp | None = None,
) -> StageResult:
    """Refit on all pre-2019 data. Unsupervised thresholds are recalibrated on the 2015 to 2018
    non-event days; supervised models reuse the validation stage probability threshold."""
    cutoff = pd.Timestamp(ctx.cfg.splits.test_start)
    model, scores = fit_model(ctx, family, params, seed, cutoff, train_start)
    if family in SUPERVISED:
        assert val_threshold is not None
        thr = val_threshold
        _, res = calibrate_and_eval(ctx, scores, ctx.val_seg)  # in-sample, informative only
    else:
        thr, res = calibrate_and_eval(ctx, scores, ctx.val_seg)
    return StageResult(
        family, config_idx, params, seed, str(cutoff.date()), model, scores, thr, res
    )
