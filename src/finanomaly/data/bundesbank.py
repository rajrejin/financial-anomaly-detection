"""Bundesbank SDMX web service client (dataflows BBEX3 and BBSIS)."""

from __future__ import annotations

import io

import pandas as pd

from finanomaly.config import Config
from finanomaly.data.http import fetch_cached

# name -> (flow, key). Keys discovered via the dataflow metadata and verified at run time.
SERIES: dict[str, tuple[str, str]] = {
    "usd_per_eur": ("BBEX3", "D.USD.EUR.BB.AC.000"),
    "chf_per_eur": ("BBEX3", "D.CHF.EUR.BB.AC.000"),
    "gbp_per_eur": ("BBEX3", "D.GBP.EUR.BB.AC.000"),
    "jpy_per_eur": ("BBEX3", "D.JPY.EUR.BB.AC.000"),
    "de_yield_2y": ("BBSIS", "D.I.ZST.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A"),
    "de_yield_10y": ("BBSIS", "D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A"),
}


def fetch_series(cfg: Config, name: str) -> pd.DataFrame | None:
    flow, key = SERIES[name]
    cache = cfg.path("raw") / "bundesbank" / f"{flow}_{key}_from{cfg.data.start[:4]}.csv.gz"
    body = fetch_cached(
        f"{cfg.data.bundesbank_base}/data/{flow}/{key}",
        {"startPeriod": cfg.data.start, "format": "sdmx_csv"},
        cache,
    )
    if body is None:
        return None
    df = pd.read_csv(io.BytesIO(body), sep=";", encoding="latin-1", na_values=["."])
    df.columns = [c.lstrip("﻿ï»¿") for c in df.columns]
    df["date"] = pd.to_datetime(df["TIME_PERIOD"])
    out = df[["date", "OBS_VALUE"]].dropna().rename(columns={"OBS_VALUE": name})
    out[name] = out[name].astype(float)
    meta = {
        "title": str(df["BBK_TITLE"].iloc[0]) if "BBK_TITLE" in df else "",
        "unit": str(df["BBK_UNIT"].iloc[0]) if "BBK_UNIT" in df else "",
    }
    out.attrs.update(meta)
    return out.set_index("date")
