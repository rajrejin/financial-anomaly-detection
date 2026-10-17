"""Rule-based baselines (no learning). Both produce a score; the alarm threshold is calibrated to
the false alarm budget with the same code as for the learned models."""

from __future__ import annotations

import numpy as np

from finanomaly.context import Context

HISTORY = 756
MIN_HISTORY = 252


def rule_return_score(ctx: Context) -> np.ndarray:
    """Absolute DAX return in trailing robust standard deviations."""
    return np.abs(ctx.z["r_dax"].to_numpy(dtype=float))


def rule_vol_score(ctx: Context) -> np.ndarray:
    """Percentile rank of the 20 day realised volatility within its own trailing history."""
    v = ctx.base["vol20"].to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        past = v[max(0, i - HISTORY) : i]
        past = past[~np.isnan(past)]
        if len(past) >= MIN_HISTORY:
            out[i] = float(np.mean(past <= v[i]))
    return out


RULES = {"rule_return": rule_return_score, "rule_vol": rule_vol_score}
