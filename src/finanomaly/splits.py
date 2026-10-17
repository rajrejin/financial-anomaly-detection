"""Temporal splits and training labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finanomaly.config import Config
from finanomaly.events import EventWindows, label_series


@dataclass(frozen=True)
class Split:
    fit: np.ndarray  # boolean masks over the dataset index
    val: np.ndarray
    pre2019: np.ndarray
    test: np.ndarray


def make_split(cfg: Config, index: pd.DatetimeIndex) -> Split:
    idx = index.tz_localize(None) if index.tz is not None else index
    val_start = pd.Timestamp(cfg.splits.val_start)
    val_end = pd.Timestamp(cfg.splits.val_end)
    test_start = pd.Timestamp(cfg.splits.test_start)
    fit_start = pd.Timestamp(cfg.splits.fit_start)
    return Split(
        fit=np.asarray((idx >= fit_start) & (idx < val_start)),
        val=np.asarray((idx >= val_start) & (idx <= val_end)),
        pre2019=np.asarray((idx >= fit_start) & (idx < test_start)),
        test=np.asarray(idx >= test_start),
    )


def training_labels(
    wins: list[EventWindows], n: int, index: pd.DatetimeIndex, cutoff: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray]:
    """Labels for rows before `cutoff`, using only events with onset before `cutoff`.

    Returns (y, rows) where `rows` is a boolean mask of usable training rows.
    """
    idx = index.tz_localize(None) if index.tz is not None else index
    usable = [w for w in wins if w.event.onset < cutoff]
    y = label_series(usable, n)
    rows = np.asarray(idx < cutoff)
    # A label window that crosses the cutoff is cut at the cutoff by `rows`.
    return y, rows
