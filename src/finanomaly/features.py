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


def base_features(t: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = pd.DataFrame(index=t.index)
    lr = lambda s: np.log(s).diff()  # noqa: E731
    f["r_dax"] = lr(t["dax_close"])
    f["r_usd"] = lr(t["usd_per_eur"])
    f["r_chf"] = lr(t["chf_per_eur"])
    f["r_gbp"] = lr(t["gbp_per_eur"])
    f["r_jpy"] = lr(t["jpy_per_eur"] / t["usd_per_eur"])  # USD/JPY via EUR
    f["d_y2"] = t["de_yield_2y"].diff()
    f["d_y10"] = t["de_yield_10y"].diff()
    f["d_spread"] = (t["de_yield_10y"] - t["de_yield_2y"]).diff()
    f["policy_rate"] = t["ecb_dfr"]
    f["d_policy"] = t["ecb_dfr"].diff()
    w1, w2, w3 = cfg.features.vol_windows
    ann = np.sqrt(252.0)
    f["vol5"] = f["r_dax"].rolling(w1).std() * ann
    f["vol20"] = f["r_dax"].rolling(w2).std() * ann
    f["vol60"] = f["r_dax"].rolling(w3).std() * ann
    f["vol_ratio"] = f["vol5"] / f["vol60"]
    dw = cfg.features.drawdown_window
    f["drawdown"] = t["dax_close"] / t["dax_close"].rolling(dw, min_periods=dw).max() - 1.0
    f["corr60"] = f["r_dax"].rolling(cfg.features.corr_window).corr(f["r_usd"])
    return f[BASE_FEATURES]


def _trailing_z_column(x: np.ndarray, window: int, floor: float) -> np.ndarray:
    n = len(x)
    z = np.full(n, np.nan)
    if n <= window:
        return z
    # windows[i] = x[i : i + window], used for day i + window (strictly past data)
    win = sliding_window_view(x, window)[: n - window]
    with np.errstate(all="ignore"):
        med = np.nanmedian(win, axis=1)
        mad = np.nanmedian(np.abs(win - med[:, None]), axis=1) * 1.4826
        sd = np.nanstd(win, axis=1)
        valid = np.sum(~np.isnan(win), axis=1) == window
    scale = np.where(mad > 0, mad, np.where(sd > 0, sd, floor))
    scale = np.maximum(scale, floor)
    cur = x[window:]
    out = np.clip((cur - med) / scale, -Z_CLIP, Z_CLIP)
    out[~valid] = np.nan
    z[window:] = out
    return z


def trailing_z(f: pd.DataFrame, window: int) -> pd.DataFrame:
    out = {}
    for c in f.columns:
        floor = SCALE_FLOOR.get(c, DEFAULT_FLOOR)
        out[c] = _trailing_z_column(f[c].to_numpy(dtype=float), window, floor)
    return pd.DataFrame(out, index=f.index)


def build_features(t: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (base features, trailing z-scored model inputs)."""
    base = base_features(t, cfg)
    z = trailing_z(base, cfg.features.z_window)
    return base, z


def valid_rows(z: pd.DataFrame) -> pd.Series:
    return z.notna().all(axis=1)
