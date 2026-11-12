"""Leakage-safe features. Every value at day t uses data up to and including day t only.

Model inputs are trailing median/MAD z-scores of the base features. The statistics for day t come
from the `window` days strictly before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from finanomaly.config import Config

BASE_FEATURES = [
    "r_dax",
    "r_usd",
    "r_chf",
    "r_gbp",
    "r_jpy",
    "d_y2",
    "d_y10",
    "d_spread",
    "policy_rate",
    "d_policy",
    "vol5",
    "vol20",
    "vol60",
    "vol_ratio",
    "drawdown",
    "corr60",
]

GROUPS: dict[str, list[str]] = {
    "equity": ["r_dax", "vol5", "vol20", "vol60", "vol_ratio", "drawdown"],
    "fx": ["r_usd", "r_chf", "r_gbp", "r_jpy", "corr60"],
    "rates": ["d_y2", "d_y10", "d_spread", "policy_rate", "d_policy"],
}

# Minimum scale used when the window MAD and standard deviation are both zero.
SCALE_FLOOR = {"policy_rate": 0.05, "d_policy": 0.05}
DEFAULT_FLOOR = 1e-6
Z_CLIP = 10.0


def base_features(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Raw daily features from the aligned price table (one row per DAX trading day)."""

    def log_return(series: pd.Series) -> pd.Series:
        return np.log(series).diff()

    feats = pd.DataFrame(index=table.index)

    # Daily log returns of the DAX and of the euro against four currencies.
    feats["r_dax"] = log_return(table["dax_close"])
    feats["r_usd"] = log_return(table["usd_per_eur"])
    feats["r_chf"] = log_return(table["chf_per_eur"])
    feats["r_gbp"] = log_return(table["gbp_per_eur"])
    feats["r_jpy"] = log_return(table["jpy_per_eur"] / table["usd_per_eur"])  # USD/JPY via EUR

    # Interest rates: day-over-day changes of German yields, their spread, and the ECB deposit rate.
    feats["d_y2"] = table["de_yield_2y"].diff()
    feats["d_y10"] = table["de_yield_10y"].diff()
    feats["d_spread"] = (table["de_yield_10y"] - table["de_yield_2y"]).diff()
    feats["policy_rate"] = table["ecb_dfr"]
    feats["d_policy"] = table["ecb_dfr"].diff()

    # Realised volatility of the DAX over short, medium and long windows, annualised.
    short, medium, long = cfg.features.vol_windows
    annualise = np.sqrt(252.0)
    feats["vol5"] = feats["r_dax"].rolling(short).std() * annualise
    feats["vol20"] = feats["r_dax"].rolling(medium).std() * annualise
    feats["vol60"] = feats["r_dax"].rolling(long).std() * annualise
    feats["vol_ratio"] = feats["vol5"] / feats["vol60"]

    # Drawdown: how far the DAX is below its highest close of the last year.
    window = cfg.features.drawdown_window
    rolling_max = table["dax_close"].rolling(window, min_periods=window).max()
    feats["drawdown"] = table["dax_close"] / rolling_max - 1.0

    feats["corr60"] = feats["r_dax"].rolling(cfg.features.corr_window).corr(feats["r_usd"])
    return feats[BASE_FEATURES]


def _trailing_z_column(x: np.ndarray, window: int, floor: float) -> np.ndarray:
    """Robust z-score of each value against the `window` values strictly before it.

    z = (value - median) / (1.4826 * MAD). The factor 1.4826 makes the MAD comparable to a
    standard deviation for normal data. The first `window` entries stay NaN.
    """
    n = len(x)
    z = np.full(n, np.nan)
    if n <= window:
        return z
    # past[i] = x[i : i + window] is the history used for day i + window, so day t never
    # sees its own value or anything later.
    past = sliding_window_view(x, window)[: n - window]
    with np.errstate(all="ignore"):
        median = np.nanmedian(past, axis=1)
        mad = np.nanmedian(np.abs(past - median[:, None]), axis=1) * 1.4826
        std = np.nanstd(past, axis=1)
        history_complete = np.sum(~np.isnan(past), axis=1) == window
    # Fall back from MAD to the standard deviation to a fixed floor when a series is flat.
    scale = np.where(mad > 0, mad, np.where(std > 0, std, floor))
    scale = np.maximum(scale, floor)
    today = x[window:]
    scores = np.clip((today - median) / scale, -Z_CLIP, Z_CLIP)
    scores[~history_complete] = np.nan
    z[window:] = scores
    return z


def trailing_z(feats: pd.DataFrame, window: int) -> pd.DataFrame:
    columns = {}
    for name in feats.columns:
        floor = SCALE_FLOOR.get(name, DEFAULT_FLOOR)
        columns[name] = _trailing_z_column(feats[name].to_numpy(dtype=float), window, floor)
    return pd.DataFrame(columns, index=feats.index)


def build_features(table: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (base features, trailing z-scored model inputs)."""
    base = base_features(table, cfg)
    return base, trailing_z(base, cfg.features.z_window)


def valid_rows(z: pd.DataFrame) -> pd.Series:
    """True for days where every z-scored feature is available."""
    return z.notna().all(axis=1)
