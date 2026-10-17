"""Evaluation metrics for a score series on a segment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from finanomaly.evaluation.episodes import count_false_alarms, detect_events
from finanomaly.events import EventWindows, label_series, mask_from


def score_separation(scores: np.ndarray, label_mask: np.ndarray, normal_mask: np.ndarray) -> float:
    """Standardised mean difference between event-labelled days and normal days."""
    a = scores[label_mask & ~np.isnan(scores)]
    b = scores[normal_mask & ~np.isnan(scores)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def evaluate_segment(
    scores: np.ndarray,
    threshold: float,
    seg: tuple[int, int],
    wins: list[EventWindows],
    merge_gap: int,
) -> dict:
    n = len(scores)
    alarm = np.nan_to_num(scores, nan=-np.inf) > threshold
    events = detect_events(alarm, seg, wins)
    n_fa, years, fa_eps = count_false_alarms(alarm, seg, wins, merge_gap)
    y = label_series(wins, n)
    in_seg = np.zeros(n, dtype=bool)
    in_seg[seg[0] : seg[1]] = True
    valid = in_seg & ~np.isnan(scores)
    out: dict = {
        "n_events": len(events),
        "n_detected": int(sum(e["detected"] for e in events)),
        "event_recall": (sum(e["detected"] for e in events) / len(events)) if events else None,
        "false_alarm_episodes": n_fa,
        "eligible_years": years,
        "false_alarm_rate": (n_fa / years) if years > 0 else None,
        "events": events,
        "false_alarm_episode_list": [(int(a), int(b)) for a, b in fa_eps],
    }
    if y[valid].min() != y[valid].max():
        out["average_precision"] = float(average_precision_score(y[valid], scores[valid]))
        out["roc_auc"] = float(roc_auc_score(y[valid], scores[valid]))
    else:
        out["average_precision"] = None
        out["roc_auc"] = None
    normal = in_seg & ~mask_from(wins, n, "detect") & ~mask_from(wins, n, "buffer")
    out["score_separation"] = score_separation(scores, in_seg & (y == 1), normal)
    return out
