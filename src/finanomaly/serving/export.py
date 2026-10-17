"""Export the production model from the registry into a self-contained directory.

Layout of the export directory:

    mlflow_model/        the registered MLflow pyfunc model
    feature_config.json  feature names and parameters
    history/daily.parquet  raw daily table (mutable; see history/status.json)
    history/status.json
    manifest.json        model name, version, alias, run id, hashes, git commit, file checksums
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import mlflow.artifacts
from mlflow.tracking import MlflowClient

from finanomaly.config import Config, sha256_file
from finanomaly.experiments import tracking
from finanomaly.experiments.runner import REGISTRY_NAMES
from finanomaly.progress import git_commit

IMMUTABLE_DIRS = ("mlflow_model",)
IMMUTABLE_FILES = ("feature_config.json",)


def file_checksums(root: Path) -> dict[str, str]:
    out = {}
    files = [root / f for f in IMMUTABLE_FILES]
    for d in IMMUTABLE_DIRS:
        files += [p for p in sorted((root / d).rglob("*")) if p.is_file()]
    for p in files:
        out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def find_production(client: MlflowClient) -> tuple[str, Any]:
    for name in REGISTRY_NAMES.values():
        try:
            return name, client.get_model_version_by_alias(name, "production")
        except Exception:
            continue
    raise RuntimeError("no model has the production alias; run `finanomaly promote` first")


def export_model(cfg: Config, out_dir: Path | None = None, alias: str = "production") -> dict:
    client = tracking.setup(cfg)
    name, mv = find_production(client)
    root = out_dir or Path(cfg.serving.model_dir)
    root = root if root.is_absolute() else (Path(cfg.root) / root)
    if root.exists() and any(root.iterdir()) and not (root / "manifest.json").exists():
        raise RuntimeError(f"{root} is not empty and is not an export directory")
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}@{alias}", dst_path=str(root / "_dl")
    )
    shutil.move(local, str(root / "mlflow_model"))
    shutil.rmtree(root / "_dl", ignore_errors=True)
    meta = next(
        m
        for m in (
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / "mlflow_model").rglob("meta.json"))
        )
        if "threshold" in m
    )
    feature_config = {
        "features": meta["features"],
        "z_window": cfg.features.z_window,
        "vol_windows": cfg.features.vol_windows,
        "corr_window": cfg.features.corr_window,
        "drawdown_window": cfg.features.drawdown_window,
        "merge_gap_days": cfg.alarm.merge_gap_days,
    }
    (root / "feature_config.json").write_text(json.dumps(feature_config, indent=2))
    hist = root / "history"
    hist.mkdir(exist_ok=True)
    shutil.copy(cfg.path("processed") / "daily.parquet", hist / "daily.parquet")
    status = {
        "quality_ok": True,
        "published": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "daily_sha256": sha256_file(hist / "daily.parquet"),
    }
    (hist / "status.json").write_text(json.dumps(status, indent=2))
    manifest = {
        "model_name": name,
        "model_version": int(mv.version),
        "alias": alias,
        "run_id": mv.run_id,
        "family": meta["family"],
        "threshold": meta["threshold"],
        "data_hash": meta["data_hash"],
        "events_hash": sha256_file(cfg.path("events")),
        "git_commit": git_commit(),
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": file_checksums(root),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
