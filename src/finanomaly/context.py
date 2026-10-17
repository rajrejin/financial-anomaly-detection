"""Shared experiment context: dataset, features, events, windows and splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finanomaly.config import Config, sha256_file
from finanomaly.data.build import load_dataset
from finanomaly.events import Event, EventWindows, event_windows, label_series, load_events
from finanomaly.features import build_features, valid_rows
from finanomaly.splits import Split, make_split


@dataclass
class Context:
    cfg: Config
    table: pd.DataFrame
    base: pd.DataFrame
    z: pd.DataFrame
    valid: np.ndarray
    events: list[Event]
    wins: list[EventWindows]
    y: np.ndarray
    split: Split
    data_hash: str

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.table.index

    @property
    def n(self) -> int:
        return len(self.table)

    def seg(self, mask: np.ndarray) -> tuple[int, int]:
        pos = np.flatnonzero(mask)
        return int(pos[0]), int(pos[-1]) + 1

    @property
    def val_seg(self) -> tuple[int, int]:
        return self.seg(self.split.val)

    @property
    def test_seg(self) -> tuple[int, int]:
        return self.seg(self.split.test)

    @property
    def fit_seg(self) -> tuple[int, int]:
        return self.seg(self.split.fit)


def load_context(cfg: Config, table: pd.DataFrame | None = None) -> Context:
    t = table if table is not None else load_dataset(cfg)
    base, z = build_features(t, cfg)
    events = load_events(cfg)
    wins = event_windows(cfg, t.index, events)
    y = label_series(wins, len(t))
    data_hash = sha256_file(cfg.path("processed") / "daily.parquet")
    return Context(
        cfg=cfg,
        table=t,
        base=base,
        z=z,
        valid=valid_rows(z).to_numpy(),
        events=events,
        wins=wins,
        y=y,
        split=make_split(cfg, t.index),
        data_hash=data_hash,
    )
