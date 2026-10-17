"""DAX index data: yfinance primary, Stooq fallback. Both cached."""

from __future__ import annotations

import gzip
import io
import logging

import pandas as pd

from finanomaly.config import Config

log = logging.getLogger(__name__)


def fetch_dax(cfg: Config) -> pd.DataFrame:
    cache = cfg.path("raw") / "dax" / "gdaxi_yfinance.csv.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return _read(gzip.decompress(cache.read_bytes()))
    try:
        import yfinance as yf

        d = yf.download(
            "^GDAXI",
            start=cfg.data.start,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
        if d is None or d.empty:
            raise RuntimeError("yfinance returned no data")
        d = d.reset_index()
        d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
        # Drop the current day if it may be incomplete: keep days strictly before today (UTC).
        d = d[d["Date"] < pd.Timestamp.utcnow().tz_localize(None).normalize()]
        buf = io.StringIO()
        d.to_csv(buf, index=False)
        cache.write_bytes(gzip.compress(buf.getvalue().encode()))
        return _read(buf.getvalue().encode())
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("yfinance failed (%s); trying Stooq", exc)
        return _stooq(cfg)


def _stooq(cfg: Config) -> pd.DataFrame:  # pragma: no cover - network dependent
    import httpx

    r = httpx.get("https://stooq.com/q/d/l/?s=%5Edax&i=d", timeout=60)
    r.raise_for_status()
    cache = cfg.path("raw") / "dax" / "dax_stooq.csv.gz"
    cache.write_bytes(gzip.compress(r.content))
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"Data": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")[["Open", "High", "Low", "Close"]].sort_index()


def _read(body: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(body), parse_dates=["Date"])
    df = df.drop_duplicates("Date").sort_values("Date").set_index("Date")
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume"]]
