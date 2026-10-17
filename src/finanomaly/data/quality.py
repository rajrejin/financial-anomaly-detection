"""Data quality report: coverage, gaps, duplicates, outliers (kept), plausibility."""

from __future__ import annotations

import json

import numpy as np

from finanomaly.config import Config
from finanomaly.data.build import load_dataset

PLAUSIBLE = {
    "usd_per_eur": (0.5, 2.0),
    "chf_per_eur": (0.8, 2.0),
    "gbp_per_eur": (0.4, 1.2),
    "jpy_per_eur": (80.0, 250.0),
    "de_yield_2y": (-1.5, 8.0),
    "de_yield_10y": (-1.5, 8.0),
    "ecb_mro": (0.0, 6.0),
    "ecb_dfr": (-1.0, 6.0),
    "ecb_mlf": (0.0, 7.0),
    "dax_close": (1000.0, 60000.0),
}


def run_quality(cfg: Config) -> dict:
    t = load_dataset(cfg)
    report: dict = {
        "rows": len(t),
        "date_min": str(t.index.min().date()),
        "date_max": str(t.index.max().date()),
        "duplicate_dates": int(t.index.duplicated().sum()),
        "series": {},
    }
    value_cols = [
        c for c in t.columns if not c.endswith("__ffill") and c not in ("dax_high", "dax_low")
    ]
    for c in value_cols:
        s = t[c]
        lo, hi = PLAUSIBLE[c]
        first = s.first_valid_index()
        info: dict = {
            "first_valid": str(first.date()) if first is not None else None,
            "missing": int(s.isna().sum()),
            "missing_share": round(float(s.isna().mean()), 5),
            "forward_filled_days": int(t[f"{c}__ffill"].sum()) if f"{c}__ffill" in t else 0,
            "min": float(s.min()),
            "max": float(s.max()),
            "plausible_range": [lo, hi],
            "implausible_values": int(((s < lo) | (s > hi)).sum()),
        }
        if f"{c}__ffill" in t:
            info["forward_fill_share"] = round(float(t[f"{c}__ffill"].mean()), 5)
        # gaps of consecutive missing values longer than 3 trading days
        isna = s.isna().astype(int)
        runs = isna.groupby((isna.diff() != 0).cumsum()).sum()
        info["longest_missing_run_days"] = int(runs.max()) if len(runs) else 0
        # outliers: change series beyond 8 robust SD (reported, not removed)
        d = (
            np.log(s).diff()
            if c in ("dax_close", "usd_per_eur", "chf_per_eur", "gbp_per_eur", "jpy_per_eur")
            else s.diff()
        )
        mad = float((d - d.median()).abs().median()) * 1.4826
        if mad > 0:
            z = (d - d.median()) / mad
            info["outliers_8_robust_sd"] = int((z.abs() > 8).sum())
            info["largest_abs_robust_z"] = round(float(z.abs().max()), 2)
        report["series"][c] = info
    out = cfg.path("results") / "data_quality.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def quality_ok(report: dict) -> tuple[bool, list[str]]:
    problems = []
    if report["duplicate_dates"]:
        problems.append("duplicate dates")
    for name, info in report["series"].items():
        if info["implausible_values"]:
            problems.append(f"{name}: implausible values")
    return not problems, problems
