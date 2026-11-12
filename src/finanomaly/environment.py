"""Environment facts: package versions (results/environment.json) and the current git commit."""

from __future__ import annotations

import json
import platform
import subprocess
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


def git_commit() -> str:
    """Commit hash of the working tree, or "unknown" outside a git checkout (e.g. in Docker)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
