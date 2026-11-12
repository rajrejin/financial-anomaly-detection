"""Audit: how much of the test-period event recall could arise by chance?

Event recall counts an event as detected if any alarm falls in a 23-trading-day detection window.
A model that alarms often will hit such windows by chance. For each entry this script measures, on
test-period days outside every event window and buffer, the probability that a window of the same
length contains at least one alarm, and compares the observed detections with a binomial chance
level. It also repeats the comparison for fast detection (an alarm within 2 trading days of onset).

A second, labelled oracle analysis sets every entry's threshold on the test scores so that all
entries alarm on the same share of test days, and counts detections. This uses test information and
is for interpretation only; it changes no decision.

Writes results/audit_chance_level.json. Run: python scripts/audit_chance_level.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binom

from finanomaly.config import get_config
from finanomaly.context import load_context
from finanomaly.events import mask_from

FAST_BEFORE, FAST_AFTER = 2, 2
SHARES = [0.025, 0.05, 0.083]


def main() -> None:
    cfg = get_config()
    ctx = load_context(cfg)
    res = cfg.path("results")
    sc = pd.read_csv(res / "test_scores.csv", index_col=0)
    a, b = ctx.test_seg
    assert len(sc) == b - a, "test_scores.csv does not match the test segment"
    base = json.loads((res / "baselines_validation.json").read_text())
    fa = pd.read_csv(res / "test_false_alarms.csv")
    thr = {k: v["threshold"] for k, v in base.items()}
    thr.update(fa.groupby("model")["threshold"].first().to_dict())

    n = ctx.n
    excl = (mask_from(ctx.wins, n, "detect") | mask_from(ctx.wins, n, "buffer"))[a:b]
    test_wins = [w for w in ctx.wins if a <= w.onset_pos < b]
    w = ctx.cfg.windows
    det_len = w.detect_before + w.detect_after + 1
    fast_len = FAST_BEFORE + FAST_AFTER + 1

    def chance(alarm: np.ndarray, length: int) -> float:
        hits, total = 0, 0
        for s in range(0, len(alarm) - length + 1):
            if excl[s : s + length].any():
                continue
            total += 1
            hits += bool(alarm[s : s + length].any())
        return hits / total if total else float("nan")

    def detections(alarm: np.ndarray) -> tuple[int, int]:
        full, fast = 0, 0
        for win in test_wins:
            p = win.onset_pos - a
            d0, d1 = win.detect[0] - a, win.detect[1] - a
            full += bool(alarm[d0:d1].any())
            fast += bool(alarm[max(p - FAST_BEFORE, 0) : p + FAST_AFTER + 1].any())
        return full, fast

    k_events = len(test_wins)
    entries = []
    for name in sc.columns:
        s = sc[name].to_numpy()
        alarm = np.nan_to_num(s, nan=-np.inf) > thr[name]
        full, fast = detections(alarm)
        p_full, p_fast = chance(alarm, det_len), chance(alarm, fast_len)
        entries.append(
            {
                "entry": name,
                "alarm_day_share": float(alarm.mean()),
                "detected": full,
                "chance_window_hit_prob": p_full,
                "expected_by_chance": p_full * k_events,
                "p_value_at_least_detected": float(binom.sf(full - 1, k_events, p_full)),
                "fast_detected": fast,
                "chance_fast_hit_prob": p_fast,
                "expected_fast_by_chance": p_fast * k_events,
                "p_value_at_least_fast": float(binom.sf(fast - 1, k_events, p_fast)),
            }
        )

    oracle = []
    for share in SHARES:
        for name in sc.columns:
            s = sc[name].to_numpy()
            t = np.nanquantile(s, 1 - share)
            alarm = np.nan_to_num(s, nan=-np.inf) > t
            full, fast = detections(alarm)
            oracle.append(
                {"alarm_day_share": share, "entry": name, "detected": full, "fast_detected": fast}
            )

    out = {
        "n_test_events": k_events,
        "detection_window_days": det_len,
        "fast_window_days": fast_len,
        "note": (
            "Chance probabilities are estimated on test days outside all event windows and "
            "buffers. p-values are binomial tails assuming independent events, which is only "
            "approximately true. The oracle table thresholds every entry on the test scores to a "
            "common alarm-day share; it uses test information and is for interpretation only."
        ),
        "entries": entries,
        "oracle_matched_share": oracle,
    }
    (res / "audit_chance_level.json").write_text(json.dumps(out, indent=2))
    for e in entries:
        print(
            f"{e['entry']:28s} share={e['alarm_day_share']:.3f} det={e['detected']}/{k_events} "
            f"chance={e['expected_by_chance']:.2f} p={e['p_value_at_least_detected']:.4f} | "
            f"fast={e['fast_detected']} chance={e['expected_fast_by_chance']:.2f} "
            f"p={e['p_value_at_least_fast']:.4f}"
        )
    print(
        pd.DataFrame(oracle)
        .pivot(index="entry", columns="alarm_day_share", values=["detected", "fast_detected"])
        .to_string()
    )


if __name__ == "__main__":
    main()
