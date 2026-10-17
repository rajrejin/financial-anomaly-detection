"""Scoring service: score dates, ranges and ad-hoc feature vectors with the loaded bundle."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from finanomaly.evaluation.episodes import alarm_episodes
from finanomaly.serving.model_loader import Bundle

TOP_K = 5
FEATURE_ABS_LIMIT = 20.0  # model inputs are z-scores clipped at 10


class ScoringError(ValueError):
    pass


class ScoringService:
    def __init__(self, bundle: Bundle, stale_days: int = 5) -> None:
        self.b = bundle
        self.stale_days = stale_days
        idx = bundle.z.index.tz_localize(None) if bundle.z.index.tz is not None else bundle.z.index
        self.dates = idx
        X = bundle.z.to_numpy(dtype=float)
        self.scores = np.full(len(X), np.nan)
        v = bundle.valid
        self.scores[v] = bundle.pyfunc.model.score(X[v])
        self.X = X

    # -- meta -------------------------------------------------------------------------------
    @property
    def first_date(self) -> pd.Timestamp:
        return self.dates[self.b.valid][0]

    @property
    def last_date(self) -> pd.Timestamp:
        return self.dates[self.b.valid][-1]

    def data_age_days(self, now: pd.Timestamp | None = None) -> int:
        now = now or pd.Timestamp(time.time(), unit="s").normalize()
        return int((now - self.last_date).days)

    def is_stale(self, now: pd.Timestamp | None = None) -> bool:
        return self.data_age_days(now) > self.stale_days or not self.b.status.get(
            "quality_ok", True
        )

    # -- scoring ----------------------------------------------------------------------------
    def _contrib(self, row: np.ndarray) -> list[dict[str, float | str]]:
        c = self.b.pyfunc.model.explain(row.reshape(1, -1))[0]
        order = np.argsort(-np.abs(c))[:TOP_K]
        return [{"feature": self.b.features[i], "contribution": float(c[i])} for i in order]

    def position(self, date: pd.Timestamp) -> int:
        d = (
            pd.Timestamp(date).tz_localize(None)
            if pd.Timestamp(date).tzinfo
            else pd.Timestamp(date)
        )
        if d < self.first_date or d > self.last_date:
            raise ScoringError(
                f"date outside the available range {self.first_date.date()} to "
                f"{self.last_date.date()}"
            )
        pos = int(self.dates.searchsorted(d))
        if pos >= len(self.dates) or self.dates[pos] != d or not self.b.valid[pos]:
            raise ScoringError(f"{d.date()} is not a trading day in the dataset")
        return pos

    def score_date(self, date: pd.Timestamp | None = None) -> dict[str, Any]:
        pos = int(np.flatnonzero(self.b.valid)[-1]) if date is None else self.position(date)
        s = float(self.scores[pos])
        return {
            "date": str(self.dates[pos].date()),
            "score": s,
            "threshold": self.b.threshold,
            "alarm": bool(s > self.b.threshold),
            "contributions": self._contrib(self.X[pos]),
            "contribution_kind": self.contribution_kind(),
            "stale": self.is_stale(),
        }

    def contribution_kind(self) -> str:
        return {
            "isolation_forest": "occlusion (score change when the feature is set to its centre)",
            "autoencoder": "share of squared reconstruction error",
            "lightgbm": "SHAP value in log-odds",
            "xgboost": "SHAP value in log-odds",
        }[self.b.manifest["family"]]

    def anomalies(self, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
        a = self.position_ge(start)
        e = self.position_le(end)
        if e < a:
            return []
        alarm = np.zeros(len(self.scores), dtype=bool)
        alarm[a : e + 1] = np.nan_to_num(self.scores[a : e + 1], nan=-np.inf) > self.b.threshold
        out = []
        for s, t in alarm_episodes(alarm, int(self.b.feature_config["merge_gap_days"])):
            seg = self.scores[s : t + 1]
            k = int(np.nanargmax(seg))
            out.append(
                {
                    "start": str(self.dates[s].date()),
                    "end": str(self.dates[t].date()),
                    "days": t - s + 1,
                    "peak_date": str(self.dates[s + k].date()),
                    "peak_score": float(seg[k]),
                }
            )
        return out

    def position_ge(self, d: pd.Timestamp) -> int:
        d = pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d)
        return int(self.dates.searchsorted(max(d, self.first_date)))

    def position_le(self, d: pd.Timestamp) -> int:
        d = pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d)
        return int(self.dates.searchsorted(min(d, self.last_date), side="right")) - 1

    def score_vector(self, features: dict[str, float]) -> dict[str, Any]:
        names = self.b.features
        missing = [n for n in names if n not in features]
        extra = [k for k in features if k not in names]
        if missing or extra:
            raise ScoringError(f"features must match the schema; missing={missing} extra={extra}")
        for k, v in features.items():
            if not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v):
                raise ScoringError(f"feature {k} must be a finite number")
            if abs(v) > FEATURE_ABS_LIMIT:
                raise ScoringError(f"feature {k} outside +-{FEATURE_ABS_LIMIT}")
        row = np.array([float(features[n]) for n in names])
        s = float(self.b.pyfunc.model.score(row.reshape(1, -1))[0])
        return {
            "score": s,
            "threshold": self.b.threshold,
            "alarm": bool(s > self.b.threshold),
            "contributions": self._contrib(row),
            "contribution_kind": self.contribution_kind(),
        }
