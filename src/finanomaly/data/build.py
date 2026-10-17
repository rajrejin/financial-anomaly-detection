"""Fetch all sources and build the tidy daily table aligned on DAX trading days."""

from __future__ import annotations

import logging

import pandas as pd

from finanomaly.config import Config
from finanomaly.data import bundesbank, dax, ecb

log = logging.getLogger(__name__)

# Policy rates are step functions (a level that persists until changed), so they are carried
# forward without a day limit. Observation-type series (FX, yields, DAX) use the 3 day limit.
LEVEL_SERIES = {"ecb_mro", "ecb_dfr", "ecb_mlf"}


def fetch_all(cfg: Config) -> pd.DataFrame:
    """Download (or read cache) every series; write data/series_catalog.csv. Returns catalog."""
    rows = []
    for name in bundesbank.SERIES:
        flow, key = bundesbank.SERIES[name]
        df = bundesbank.fetch_series(cfg, name)
        rows.append(
            _catalog_row(name, "Bundesbank", flow, key, df, df.attrs if df is not None else {})
        )
    for name in ecb.SERIES:
        flow, key = ecb.SERIES[name]
        df = ecb.fetch_series(cfg, name)
        rows.append(_catalog_row(name, "ECB", flow, key, df, {"unit": "percent"}))
    d = dax.fetch_dax(cfg)
    rows.append(
        {
            "name": "dax_close",
            "source": "Yahoo Finance via yfinance",
            "flow": "n/a",
            "key": "^GDAXI",
            "title": "DAX performance index, daily close",
            "unit": "index points",
            "frequency": "daily",
            "first_date": str(d.index.min().date()),
            "last_date": str(d.index.max().date()),
            "n_obs": len(d),
            "status": "ok",
        }
    )
    cat = pd.DataFrame(rows)
    out = cfg.path("raw").parent / "series_catalog.csv"
    cat.to_csv(out, index=False)
    return cat


def _catalog_row(name: str, source: str, flow: str, key: str, df, meta: dict) -> dict:
    if df is None or df.empty:
        return {"name": name, "source": source, "flow": flow, "key": key, "status": "missing"}
    return {
        "name": name,
        "source": source,
        "flow": flow,
        "key": key,
        "title": meta.get("title", ""),
        "unit": meta.get("unit", ""),
        "frequency": "daily",
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "n_obs": len(df),
        "status": "ok",
    }


def align_series(
    s: pd.Series, index: pd.DatetimeIndex, max_gap_days: int | None
) -> tuple[pd.Series, pd.Series]:
    """Reindex onto `index`. Fill from the last observation if it is at most `max_gap_days`
    calendar days old (None means no limit). Returns (values, is_forward_filled)."""
    s = s.dropna()
    obs_dates = pd.Series(s.index, index=s.index)
    full = s.reindex(s.index.union(index)).ffill()
    last_obs = obs_dates.reindex(s.index.union(index)).ffill()
    full = full.reindex(index)
    last_obs = last_obs.reindex(index)
    age = (index.to_series() - last_obs).dt.days
    filled = (age > 0) & age.notna()
    if max_gap_days is not None:
        too_old = age > max_gap_days
        full = full.mask(too_old)
        filled = filled & ~too_old
    return full, filled.fillna(False).astype(bool)


def build_dataset(cfg: Config) -> pd.DataFrame:
    d = dax.fetch_dax(cfg)
    index = d.index
    table = pd.DataFrame(index=index)
    table["dax_close"] = d["Close"]
    table["dax_high"] = d["High"]
    table["dax_low"] = d["Low"]
    sources: dict[str, pd.DataFrame | None] = {}
    for name in bundesbank.SERIES:
        sources[name] = bundesbank.fetch_series(cfg, name)
    for name in ecb.SERIES:
        sources[name] = ecb.fetch_series(cfg, name)
    for name, df in sources.items():
        if df is None:
            log.warning("series %s missing", name)
            continue
        limit = None if name in LEVEL_SERIES else cfg.data.max_ffill_days
        vals, filled = align_series(df[name], index, limit)
        table[name] = vals
        table[f"{name}__ffill"] = filled
    table.index = pd.DatetimeIndex(table.index, tz="UTC", name="date")
    out = cfg.path("processed")
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "daily.parquet")
    return table


def load_dataset(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("processed") / "daily.parquet")
