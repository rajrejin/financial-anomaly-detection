import numpy as np
import pandas as pd

from finanomaly.config import load_config
from finanomaly.evaluation.episodes import (
    alarm_episodes,
    calibrate_threshold,
    count_false_alarms,
    detect_events,
)
from finanomaly.evaluation.metrics import evaluate_segment, score_separation
from finanomaly.events import Event, event_windows

CFG = load_config()


def _event(onset):
    return Event("e", "e", "equity", pd.Timestamp(onset), ("u",), "j")


def _wins(n, onset_pos):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return event_windows(CFG, idx, [_event(idx[onset_pos])])


def test_alarm_episodes_merge_rule():
    a = np.zeros(20, dtype=bool)
    a[[2, 3, 6, 12]] = True
    # gap between 3 and 6 is 2 non-alarm days (<3): merged. 6 to 12 is 5 days: separate.
    assert alarm_episodes(a, 3) == [(2, 6), (12, 12)]
    assert alarm_episodes(a, 2) == [(2, 3), (6, 6), (12, 12)]
    assert alarm_episodes(np.zeros(5, dtype=bool), 3) == []


def test_detection_and_delay():
    wins = _wins(100, 40)  # detect window [38, 61)
    alarm = np.zeros(100, dtype=bool)
    alarm[43] = True
    res = detect_events(alarm, (0, 100), wins)
    assert res[0]["detected"] and res[0]["delay_days"] == 3
    alarm[:] = False
    alarm[39] = True
    assert detect_events(alarm, (0, 100), wins)[0]["delay_days"] == -1
    alarm[:] = False
    alarm[70] = True
    assert not detect_events(alarm, (0, 100), wins)[0]["detected"]


def test_false_alarm_counting():
    wins = _wins(504, 40)  # detect [38, 61), buffer [61, 71)
    alarm = np.zeros(504, dtype=bool)
    alarm[[45, 65, 100, 101, 300]] = True  # 45 in detect, 65 in buffer, 100-101 and 300 false
    n, years, eps = count_false_alarms(alarm, (0, 504), wins, 3)
    assert n == 2 and eps == [(100, 101), (300, 300)]
    assert abs(years - (504 - 33) / 252.0) < 1e-12


def test_metrics_hand_computed():
    wins = _wins(504, 40)
    scores = np.zeros(504)
    scores[45] = 5.0
    scores[100] = 5.0
    scores[300] = 5.0
    res = evaluate_segment(scores, 1.0, (0, 504), wins, 3)
    assert res["n_events"] == 1 and res["event_recall"] == 1.0
    assert res["false_alarm_episodes"] == 2
    assert abs(res["false_alarm_rate"] - 2 / ((504 - 33) / 252.0)) < 1e-12
    # label window is [40, 50): one positive day (45) with score 5 of 10 positives, others 0
    assert res["average_precision"] is not None and 0 < res["average_precision"] < 1


def test_score_separation_arithmetic():
    s = np.array([1.0, 1.0, 3.0, 3.0, 0.0, 2.0, 0.0, 2.0])
    lab = np.array([True, True, True, True, False, False, False, False])
    norm = ~lab
    # means 2 and 1; sample variances 4/3 each so pooled sd = sqrt(4/3)
    assert abs(score_separation(s, lab, norm) - 1 / np.sqrt(4 / 3)) < 1e-12


def test_calibration_respects_budget():
    n = 504 * 2
    wins = _wins(n, 40)
    rng_scores = np.zeros(n)
    # five isolated spikes of distinct size far from the event, one year each ~ 2 years
    for k, p in enumerate([200, 400, 600, 800, 900]):
        rng_scores[p] = 10 + k
    thr = calibrate_threshold(rng_scores, (0, n), wins, budget_per_year=1.0, merge_gap=3)
    alarm = rng_scores > thr
    n_fa, years, _ = count_false_alarms(alarm, (0, n), wins, 3)
    assert n_fa / years <= 1.0
    assert n_fa >= 1
