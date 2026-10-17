"""Alarm episodes, event detection and false alarm counting."""

from __future__ import annotations

import numpy as np

from finanomaly.events import EventWindows, mask_from

TRADING_DAYS_PER_YEAR = 252.0


def alarm_episodes(alarm: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Maximal runs of alarm days (inclusive start and end positions).

    Runs separated by fewer than `merge_gap` non-alarm days are merged.
    """
    pos = np.flatnonzero(alarm)
    if len(pos) == 0:
        return []
    episodes: list[list[int]] = [[int(pos[0]), int(pos[0])]]
    for p in pos[1:]:
        p = int(p)
        gap_days = p - episodes[-1][1] - 1
        if gap_days < merge_gap:
            episodes[-1][1] = p
        else:
            episodes.append([p, p])
    return [(a, b) for a, b in episodes]


def segment_mask(n: int, seg: tuple[int, int]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[seg[0] : seg[1]] = True
    return m


def count_false_alarms(
    alarm: np.ndarray,
    seg: tuple[int, int],
    wins: list[EventWindows],
    merge_gap: int,
) -> tuple[int, float, list[tuple[int, int]]]:
    """False alarm episodes inside `seg`, years of eligible time, and the episodes themselves.

    Eligible time excludes every detection window and buffer (all events).
    """
    n = len(alarm)
    in_seg = segment_mask(n, seg)
    excluded = mask_from(wins, n, "detect") | mask_from(wins, n, "buffer")
    a = alarm & in_seg
    eps = alarm_episodes(a, merge_gap)
    false_eps = [(s, e) for s, e in eps if not excluded[s : e + 1].any()]
    years = float((in_seg & ~excluded).sum()) / TRADING_DAYS_PER_YEAR
    return len(false_eps), years, false_eps


def detect_events(alarm: np.ndarray, seg: tuple[int, int], wins: list[EventWindows]) -> list[dict]:
    """Per event: detected flag and delay (trading days from onset to first alarm in window)."""
    n = len(alarm)
    in_seg = segment_mask(n, seg)
    out = []
    for w in wins:
        if not (seg[0] <= w.onset_pos < seg[1]):
            continue
        a, b = w.detect
        hits = np.flatnonzero(alarm[a:b] & in_seg[a:b]) + a
        detected = len(hits) > 0
        out.append(
            {
                "event": w.event.id,
                "onset": str(w.event.onset.date()),
                "detected": bool(detected),
                "delay_days": int(hits[0] - w.onset_pos) if detected else None,
            }
        )
    return out


def calibrate_threshold(
    scores: np.ndarray,
    seg: tuple[int, int],
    wins: list[EventWindows],
    budget_per_year: float,
    merge_gap: int,
    n_grid: int = 400,
    max_day_share: float = 0.05,
) -> float:
    """Lowest threshold whose false alarm episode rate on `seg` is within the budget."""
    s = scores[seg[0] : seg[1]]
    s = s[~np.isnan(s)]
    grid = np.unique(np.quantile(s, np.linspace(0.5, 1.0, n_grid)))
    for thr in grid:
        alarm = np.nan_to_num(scores, nan=-np.inf) > thr
        k, years, _ = count_false_alarms(alarm, seg, wins, merge_gap)
        in_seg = segment_mask(len(alarm), seg)
        excl = mask_from(wins, len(alarm), "detect") | mask_from(wins, len(alarm), "buffer")
        elig = in_seg & ~excl
        share = float((alarm & elig).sum()) / max(int(elig.sum()), 1)
        if years > 0 and k / years <= budget_per_year and share <= max_day_share:
            return float(thr)
    return float(grid[-1])
