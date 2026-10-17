"""Refresh job: fetch the latest days, rebuild, check quality, publish only if the checks pass."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from finanomaly.config import Config, sha256_file
from finanomaly.data.build import build_dataset, load_dataset
from finanomaly.data.quality import quality_ok, run_quality

log = logging.getLogger(__name__)


def refresh(cfg: Config, export_dir: Path | None = None) -> dict[str, Any]:
    """Returns a status dict. On failed checks the previous dataset stays in place and the
    status file (read by the API) is marked with quality_ok = false."""
    export_dir = export_dir or Path(cfg.serving.model_dir)
    export_dir = export_dir if export_dir.is_absolute() else Path(cfg.root) / export_dir
    status: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        new = build_dataset(cfg, force=True, write=False)
    except Exception as exc:  # network failure: keep the last good data
        status.update({"quality_ok": False, "problems": [f"fetch failed: {exc}"]})
        _write_status(export_dir, status)
        return status
    report = run_quality(cfg, table=new, write=False)
    ok, problems = quality_ok(report)
    path = cfg.path("processed") / "daily.parquet"
    if path.exists():
        old = load_dataset(cfg)
        if new.index.max() < old.index.max():
            ok, problems = False, [*problems, "new data ends before the existing data"]
        if len(new) < len(old):
            ok, problems = False, [*problems, "new data has fewer rows than the existing data"]
    status.update(
        {"quality_ok": ok, "problems": problems, "last_date": str(new.index.max().date())}
    )
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(path)
        hist = export_dir / "history"
        if hist.exists():
            shutil.copy(path, hist / "daily.parquet")
            status["daily_sha256"] = sha256_file(hist / "daily.parquet")
        status["published"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (cfg.path("results") / "data_quality.json").write_text(json.dumps(report, indent=2))
    _write_status(export_dir, status)
    return status


def _write_status(export_dir: Path, status: dict[str, Any]) -> None:
    hist = export_dir / "history"
    if hist.exists():
        prev = {}
        sp = hist / "status.json"
        if sp.exists():
            prev = json.loads(sp.read_text())
        merged = {**prev, **status}
        sp.write_text(json.dumps(merged, indent=2))
