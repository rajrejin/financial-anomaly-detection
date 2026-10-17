"""Write results/environment.json."""

from __future__ import annotations

import json
import platform
from importlib import metadata

import psutil

from finanomaly.config import ROOT

PKGS = [
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "torch",
    "shap",
    "mlflow",
    "fastapi",
    "pydantic",
    "yfinance",
]


def write_environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "packages": {p: metadata.version(p) for p in PKGS},
    }
    out = ROOT / "results" / "environment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(env, indent=2))
    return env
