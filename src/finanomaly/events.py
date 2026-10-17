"""Event handling: loading, windows, labels and the freeze file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from finanomaly.config import Config, sha256_file, sha256_obj


@dataclass(frozen=True)
class Event:
    id: str
    name: str
    category: str
    onset: pd.Timestamp
    sources: tuple[str, ...]
    justification: str


def load_events(cfg: Config) -> list[Event]:
    raw = yaml.safe_load(Path(cfg.path("events")).read_text(encoding="utf-8"))
    return [
        Event(
            id=e["id"],
            name=e["name"],
            category=e["category"],
            onset=pd.Timestamp(e["onset"]),
            sources=tuple(e["sources"]),
            justification=e["justification"],
        )
        for e in raw["events"]
    ]


def window_config(cfg: Config) -> dict:
    return {"windows": cfg.windows.model_dump(), "splits": cfg.splits.model_dump()}


def freeze(cfg: Config) -> dict:
    data = {
        "events_sha256": sha256_file(cfg.path("events")),
        "window_config_sha256": sha256_obj(window_config(cfg)),
        "n_events": len(load_events(cfg)),
    }
    out = cfg.path("results") / "events_frozen.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return data


def check_frozen(cfg: Config) -> bool:
    frozen = json.loads((cfg.path("results") / "events_frozen.json").read_text())
    return frozen["events_sha256"] == sha256_file(cfg.path("events")) and frozen[
        "window_config_sha256"
    ] == sha256_obj(window_config(cfg))


def _naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(None) if idx.tz is not None else idx


def onset_position(index: pd.DatetimeIndex, onset: pd.Timestamp) -> int:
    """Position of the first trading day on or after the onset date."""
    return int(_naive(index).searchsorted(onset))


@dataclass(frozen=True)
class EventWindows:
    """Positional windows (inclusive start, exclusive end) for one event."""

    event: Event
    onset_pos: int
    label: tuple[int, int]
    detect: tuple[int, int]
    buffer: tuple[int, int]


def event_windows(cfg: Config, index: pd.DatetimeIndex, events: list[Event]) -> list[EventWindows]:
    w = cfg.windows
    n = len(index)
    out = []
    for e in events:
        p = onset_position(index, e.onset)
        if p >= n:
            continue
        out.append(
            EventWindows(
                event=e,
                onset_pos=p,
                label=(p, min(p + w.label_days, n)),
                detect=(max(p - w.detect_before, 0), min(p + w.detect_after + 1, n)),
                buffer=(
                    min(p + w.detect_after + 1, n),
                    min(p + w.detect_after + 1 + w.buffer_days, n),
                ),
            )
        )
    return out


def label_series(wins: list[EventWindows], n: int) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    for w in wins:
        y[w.label[0] : w.label[1]] = 1
    return y


def mask_from(wins: list[EventWindows], n: int, kind: str) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for w in wins:
        a, b = getattr(w, kind)
        m[a:b] = True
    return m
