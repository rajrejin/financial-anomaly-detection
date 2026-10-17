"""Manifest helpers shared by export and loading (no heavy imports)."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
