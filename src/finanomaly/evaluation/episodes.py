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
    episodes = alarm_episodes(alarm & in_seg, merge_gap)
    # An episode that touches any event window is not counted as a false alarm.
    false_episodes = [
        (start, end) for start, end in episodes if not excluded[start : end + 1].any()
    ]
    years = float((in_seg & ~excluded).sum()) / TRADING_DAYS_PER_YEAR
    return len(false_episodes), years, false_episodes


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
    n = len(scores)
    seg_scores = scores[seg[0] : seg[1]]
    seg_scores = seg_scores[~np.isnan(seg_scores)]
    # Candidate thresholds: the upper half of the score distribution, lowest first.
    candidates = np.unique(np.quantile(seg_scores, np.linspace(0.5, 1.0, n_grid)))
    excluded = mask_from(wins, n, "detect") | mask_from(wins, n, "buffer")
    eligible = segment_mask(n, seg) & ~excluded
    for threshold in candidates:
        alarm = np.nan_to_num(scores, nan=-np.inf) > threshold
        n_false, years, _ = count_false_alarms(alarm, seg, wins, merge_gap)
        alarm_day_share = float((alarm & eligible).sum()) / max(int(eligible.sum()), 1)
        within_budget = years > 0 and n_false / years <= budget_per_year
        if within_budget and alarm_day_share <= max_day_share:
            return float(threshold)
    return float(candidates[-1])
