"""ECB Data Portal client (dataflow FM key interest rates)."""

from __future__ import annotations

import io

import pandas as pd

from finanomaly.config import Config
from finanomaly.data.http import fetch_cached

SERIES: dict[str, tuple[str, str]] = {
    "ecb_mro": ("FM", "D.U2.EUR.4F.KR.MRR_FR.LEV"),
    "ecb_dfr": ("FM", "D.U2.EUR.4F.KR.DFR.LEV"),
    "ecb_mlf": ("FM", "D.U2.EUR.4F.KR.MLFR.LEV"),
}


def fetch_series(cfg: Config, name: str, force: bool = False) -> pd.DataFrame | None:
    flow, key = SERIES[name]
    cache = cfg.path("raw") / "ecb" / f"{flow}_{key}_from{cfg.data.start[:4]}.csv.gz"
    body = fetch_cached(
        f"{cfg.data.ecb_base}/{flow}/{key}",
        {"format": "csvdata", "startPeriod": cfg.data.start},
        cache,
        force=force,
    )
    if body is None:
        return None
    df = pd.read_csv(io.BytesIO(body))
    df["date"] = pd.to_datetime(df["TIME_PERIOD"])
    out = df[["date", "OBS_VALUE"]].dropna().rename(columns={"OBS_VALUE": name})
    out[name] = out[name].astype(float)
    return out.drop_duplicates("date").set_index("date")
