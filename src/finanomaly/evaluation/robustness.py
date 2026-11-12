"""Phases 9 to 11: leave-one-event-out, sensitivity, cross-checks, ablation and walk-forward.

These are labelled supplementary analyses. They are never used to retune or to change the
promotion decision. Results go to results/<name>.json in the table format used by the README.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from finanomaly.config import Config
from finanomaly.context import Context, load_context
from finanomaly.evaluation.episodes import alarm_episodes, calibrate_threshold, detect_events
from finanomaly.evaluation.metrics import evaluate_segment
from finanomaly.evaluation.report import write_json
from finanomaly.evaluation.test_eval import load_registered
from finanomaly.events import mask_from
from finanomaly.experiments.runner import (
    FAMILIES,
    fit_model,
    load_experiments,
    run_final_stage,
    run_val_stage,
)
from finanomaly.features import GROUPS
from finanomaly.models.rules import RULES

FAMILY_LABEL = {
    "isolation_forest": "isolation forest",
    "autoencoder": "autoencoder",
    "lightgbm": "lightgbm",
}


def _registered(ctx: Context) -> dict[str, tuple[int, dict[str, Any]]]:
    reg = json.loads((ctx.cfg.path("results") / "registration.json").read_text())
    ex = load_experiments()
    return {f: (info["config_idx"], ex[f][info["config_idx"]]) for f, info in reg.items()}


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _save(cfg: Config, name: str, columns: list[str], rows: list[dict], note: str) -> None:
    write_json(cfg.path("results") / name, {"columns": columns, "rows": rows, "note": note})


# ---------------------------------------------------------------------------------------------
# leave-one-event-out
# ---------------------------------------------------------------------------------------------
def _oof_scores(
    ctx: Context, params: dict[str, Any], y: np.ndarray, train_mask: np.ndarray, n_best: int
) -> np.ndarray:
    """Out-of-fold LightGBM probabilities over 5 contiguous time blocks."""
    from finanomaly.models.trees import LightGBMModel

    X = ctx.z.to_numpy(dtype=float)
    rows = np.flatnonzero(ctx.valid)
    blocks = np.array_split(rows, 5)
    oof = np.full(ctx.n, np.nan)
    for blk in blocks:
        tr = np.setdiff1d(rows, blk)
        tr = tr[train_mask[tr]]
        m = LightGBMModel(**params)
        m.best_iteration = n_best
        m.fit(X[tr], y[tr], seed=0)
        oof[blk] = m.raw_score(X[blk])
    return oof


def leave_one_event_out(ctx: Context) -> None:
    cfg = ctx.cfg
    reg = _registered(ctx)
    n = ctx.n
    gap = cfg.alarm.merge_gap_days
    start = int(np.flatnonzero(ctx.valid)[0])
    seg = (start, n)
    X = ctx.z.to_numpy(dtype=float)
    best_iter = int(load_registered(ctx)["lightgbm"]["model"].best_iteration)
    rows = []
    for held in ctx.wins:
        others = [w for w in ctx.wins if w is not held]
        eid = held.event.id
        other_days = mask_from(others, n, "detect") | mask_from(others, n, "buffer")
        held_days = mask_from([held], n, "detect") | mask_from([held], n, "buffer")
        results: dict[str, list[dict]] = {}
        # rule baselines: thresholds recalibrated without the held-out event in the list
        for name, fn in RULES.items():
            s = np.where(ctx.valid, fn(ctx), np.nan)
            thr = calibrate_threshold(
                s,
                seg,
                others,
                cfg.alarm.budget_per_year,
                gap,
                max_day_share=cfg.alarm.max_alarm_day_share,
            )
            alarm = np.nan_to_num(s, nan=-np.inf) > thr
            results[name] = detect_events(alarm, seg, [held])
        # unsupervised: trained without excluding the held-out days (the event is unknown)
        for fam in ("isolation_forest", "autoencoder"):
            idx, params = reg[fam]
            m = FAMILIES[fam](**params).fit(X[ctx.valid & ~other_days], None, seed=0)
            s = np.full(n, np.nan)
            s[ctx.valid] = m.score(X[ctx.valid])
            thr = calibrate_threshold(
                s,
                seg,
                others,
                cfg.alarm.budget_per_year,
                gap,
                max_day_share=cfg.alarm.max_alarm_day_share,
            )
            alarm = np.nan_to_num(s, nan=-np.inf) > thr
            results[fam] = detect_events(alarm, seg, [held])
        # supervised: labels of the other events only; the held-out rows are left out of training
        idx, params = reg["lightgbm"]
        y = np.zeros(n, dtype=int)
        for w in others:
            y[w.label[0] : w.label[1]] = 1
        train_mask = ctx.valid & ~held_days
        oof = _oof_scores(ctx, params, y, train_mask, best_iter)
        thr = calibrate_threshold(
            oof,
            seg,
            others,
            cfg.alarm.budget_per_year,
            gap,
            max_day_share=cfg.alarm.max_alarm_day_share,
        )
        from finanomaly.models.trees import LightGBMModel

        final = LightGBMModel(**params)
        final.best_iteration = best_iter
        final.fit(X[train_mask], y[train_mask], seed=0)
        s = np.full(n, np.nan)
        s[ctx.valid] = final.raw_score(X[ctx.valid])
        alarm = np.nan_to_num(s, nan=-np.inf) > thr
        results["lightgbm"] = detect_events(alarm, seg, [held])
        for model, res in results.items():
            r = res[0]
            rows.append(
                {
                    "event": eid,
                    "model": model,
                    "detected": "yes" if r["detected"] else "no",
                    "delay": "n/a" if r["delay_days"] is None else f"{r['delay_days']:+d}",
                }
            )
    # summary rows
    df = pd.DataFrame(rows)
    note = (
        "Each event is held out in turn: unsupervised models and rule thresholds are fit and "
        "calibrated with the event treated as ordinary data (budget 3 per year over the whole "
        "sample); LightGBM is trained on the other events' labels without the held-out rows, with "
        "its threshold from out-of-fold probabilities (5 time blocks). Not chronological, so "
        "later events inform earlier ones. Detected events per model: "
        + ", ".join(
            f"{m} {int((g.detected == 'yes').sum())} of {len(g)}" for m, g in df.groupby("model")
        )
        + "."
    )
    _save(cfg, "loeo.json", ["event", "model", "detected", "delay"], rows, note)


# ---------------------------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------------------------
def _stage_pair(
    ctx: Context, fam: str, idx: int, params: dict, train_start: pd.Timestamp | None = None
) -> dict[str, Any]:
    v = run_val_stage(ctx, fam, idx, params, 0, train_start)
    f = run_final_stage(ctx, fam, idx, params, 0, v.threshold, train_start)
    r = evaluate_segment(
        f.scores, f.threshold, ctx.test_seg, ctx.wins, ctx.cfg.alarm.merge_gap_days
    )
    return {
        "val_recall": v.val_metrics["event_recall"],
        "test_recall": r["event_recall"],
        "test_fa": r["false_alarm_rate"],
        "test_ap": r["average_precision"],
        "test_share": r["alarm_day_share"],
    }


def sensitivity(ctx: Context) -> None:
    cfg = ctx.cfg
    reg = _registered(ctx)
    settings: list[tuple[str, Config, pd.Timestamp | None]] = []

    def with_windows(**kw: int) -> Config:
        return cfg.model_copy(update={"windows": cfg.windows.model_copy(update=kw)})

    def with_budget(b: float) -> Config:
        return cfg.model_copy(update={"alarm": cfg.alarm.model_copy(update={"budget_per_year": b})})

    settings.append(("base", cfg, None))
    for d in (5, 20):
        settings.append((f"label window {d} days", with_windows(label_days=d), None))
    for d in (10, 40):
        settings.append(
            (f"detection window after onset {d} days", with_windows(detect_after=d), None)
        )
    for b in (1.0, 6.0):
        settings.append((f"false alarm budget {b:g} per year", with_budget(b), None))
    settings.append(("training start 2010", cfg, pd.Timestamp("2010-01-01")))
    rows = []
    for label, c, ts in settings:
        cx = load_context(c)
        for fam in ("isolation_forest", "lightgbm"):
            idx, params = reg[fam]
            r = _stage_pair(cx, fam, idx, params, ts)
            rows.append(
                {
                    "setting": label,
                    "model": FAMILY_LABEL[fam],
                    "val_recall": _fmt(r["val_recall"]),
                    "test_recall": _fmt(r["test_recall"]),
                    "test_fa_per_year": _fmt(r["test_fa"], 2),
                    "test_ap": _fmt(r["test_ap"]),
                    "test_alarm_days": _fmt(r["test_share"] * 100, 1) + "%",
                }
            )
    _save(
        cfg,
        "sensitivity.json",
        [
            "setting",
            "model",
            "val_recall",
            "test_recall",
            "test_fa_per_year",
            "test_ap",
            "test_alarm_days",
        ],
        rows,
        "Labelled sensitivity analysis on the registered configurations (seed 0); nothing here was "
        "used to retune. Window changes alter labels, detection windows and buffers together. "
        "Test AP is on daily labels under the changed label window, so it is not comparable "
        "across label window settings. Only 3 validation and 6 test events.",
    )


# ---------------------------------------------------------------------------------------------
# cross-checks and ablation
# ---------------------------------------------------------------------------------------------
def crosscheck(ctx: Context) -> None:
    cfg = ctx.cfg
    sc = pd.read_csv(cfg.path("results") / "test_scores.csv", index_col=0)
    reg = load_registered(ctx)
    thr = {info["name"]: info["threshold"] for info in reg.values()}
    base = json.loads((cfg.path("results") / "baselines_validation.json").read_text())
    thr.update({k: v["threshold"] for k, v in base.items()})
    names = list(sc.columns)
    gap = cfg.alarm.merge_gap_days
    eps = {k: alarm_episodes((sc[k] > thr[k]).to_numpy(), gap) for k in names}

    def overlap(a: str, b: str) -> int:
        return sum(1 for s, e in eps[a] if any(not (e < s2 or e2 < s) for s2, e2 in eps[b]))

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ok = sc[[a, b]].dropna()
            rho = spearmanr(ok[a], ok[b]).statistic
            rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "spearman_test_days": _fmt(float(rho), 2),
                    "episodes_a": len(eps[a]),
                    "episodes_b": len(eps[b]),
                    "a_overlapping_b": overlap(a, b),
                    "b_overlapping_a": overlap(b, a),
                }
            )
    _save(
        cfg,
        "crosscheck.json",
        [
            "pair",
            "spearman_test_days",
            "episodes_a",
            "episodes_b",
            "a_overlapping_b",
            "b_overlapping_a",
        ],
        rows,
        "Rank correlation of daily scores in the test period and overlap of alarm episodes "
        "(episodes of A that overlap at least one episode of B), all alarms including those inside "
        "event windows.",
    )
    share_rows = []
    for k in names:
        al = (sc[k] > thr[k]).to_numpy()
        lens = [e - s + 1 for s, e in eps[k]]
        share_rows.append(
            {
                "entry": k,
                "alarm_days": int(al.sum()),
                "share_of_test_days": _fmt(float(al.mean()) * 100, 1) + "%",
                "episodes": len(eps[k]),
                "mean_episode_days": _fmt(float(np.mean(lens)) if lens else None, 1),
            }
        )
    _save(
        cfg,
        "alarm_share.json",
        ["entry", "alarm_days", "share_of_test_days", "episodes", "mean_episode_days"],
        share_rows,
        "Alarm days in the test period at each entry's fixed threshold (read from the stored "
        "scores, no refit). The false alarm episode metric ignores episodes that overlap an "
        "event window, so the share of alarm days is reported alongside it.",
    )


def ablation(ctx: Context) -> None:
    cfg = ctx.cfg
    reg = _registered(ctx)
    combos = {
        "equity only": GROUPS["equity"],
        "equity + fx": GROUPS["equity"] + GROUPS["fx"],
        "equity + rates": GROUPS["equity"] + GROUPS["rates"],
        "fx + rates (no equity)": GROUPS["fx"] + GROUPS["rates"],
        "all features": list(ctx.z.columns),
    }
    rows = []
    for label, cols in combos.items():
        cols = [c for c in ctx.z.columns if c in cols]
        z = ctx.z[cols]
        cx = replace(ctx, z=z, valid=z.notna().all(axis=1).to_numpy())
        for fam in ("isolation_forest", "autoencoder", "lightgbm"):
            idx, params = reg[fam]
            v = run_val_stage(cx, fam, idx, params, 0)
            f = run_final_stage(cx, fam, idx, params, 0, v.threshold)
            r = evaluate_segment(
                f.scores, f.threshold, cx.test_seg, cx.wins, cfg.alarm.merge_gap_days
            )
            rows.append(
                {
                    "features": label,
                    "n_features": len(cols),
                    "model": FAMILY_LABEL[fam],
                    "val_recall": _fmt(v.val_metrics["event_recall"]),
                    "val_ap": _fmt(v.val_metrics["average_precision"]),
                    "test_recall": _fmt(r["event_recall"]),
                    "test_fa_per_year": _fmt(r["false_alarm_rate"], 2),
                    "test_ap": _fmt(r["average_precision"]),
                    "test_alarm_days": _fmt(r["alarm_day_share"] * 100, 1) + "%",
                }
            )
    _save(
        cfg,
        "ablation.json",
        [
            "features",
            "n_features",
            "model",
            "val_recall",
            "val_ap",
            "test_recall",
            "test_fa_per_year",
            "test_ap",
            "test_alarm_days",
        ],
        rows,
        "Feature group ablation with the registered configurations (seed 0), same protocol as the "
        "main pipeline. Differences are small relative to the seed spread and the 3 and 6 event "
        "counts and are not claimed as effects. The share of alarm days matters: a model that "
        "alarms on most days has a low false alarm episode count because one long episode "
        "overlaps an event window.",
    )


# ---------------------------------------------------------------------------------------------
# walk-forward refit
# ---------------------------------------------------------------------------------------------
def walkforward(ctx: Context) -> None:
    cfg = ctx.cfg
    reg = _registered(ctx)
    idx_dates = ctx.index.tz_localize(None)
    a, b = ctx.test_seg
    gap = cfg.alarm.merge_gap_days
    test_summary = json.loads((cfg.path("results") / "test_summary.json").read_text())
    rows = []
    years = sorted({d.year for d in idx_dates[a:b]})
    for fam in ("isolation_forest", "autoencoder"):
        idx, params = reg[fam]
        adj = np.full(ctx.n, np.nan)
        for yr in years:
            cutoff = pd.Timestamp(f"{yr}-01-01")
            model, scores = fit_model(ctx, fam, params, 0, cutoff)
            calib = np.asarray(
                (idx_dates >= cutoff - pd.DateOffset(years=4)) & (idx_dates < cutoff)
            )
            cs = ctx_seg(calib)
            thr = calibrate_threshold(
                scores,
                cs,
                ctx.wins,
                cfg.alarm.budget_per_year,
                gap,
                max_day_share=cfg.alarm.max_alarm_day_share,
            )
            year_mask = np.asarray(idx_dates.year == yr) & np.asarray(
                idx_dates >= pd.Timestamp(cfg.splits.test_start)
            )
            adj[year_mask] = scores[year_mask] - thr
        r = evaluate_segment(adj, 0.0, (a, b), ctx.wins, gap)
        fixed = next(
            e
            for e in test_summary["entries"]
            if e["label"].startswith(f"anomaly-{fam.replace('_', '-')}")
        )
        for sched, res in (("fixed (fit before 2019)", fixed), ("walk-forward, annual refit", r)):
            rows.append(
                {
                    "model": FAMILY_LABEL[fam],
                    "schedule": sched,
                    "detected": f"{res['n_detected']} of {res['n_events']}",
                    "test_fa_per_year": _fmt(res["false_alarm_rate"], 2),
                    "test_ap": _fmt(res["average_precision"]),
                }
            )
    _save(
        cfg,
        "walkforward.json",
        ["model", "schedule", "detected", "test_fa_per_year", "test_ap"],
        rows,
        "Walk-forward: before each test year the model is refit on all earlier data and its "
        "threshold recalibrated on the previous four years (non-event days, same budget). "
        "The fixed schedule used for promotion is shown for comparison; it was not changed.",
    )


def ctx_seg(mask: np.ndarray) -> tuple[int, int]:
    pos = np.flatnonzero(mask)
    return int(pos[0]), int(pos[-1]) + 1


def run_all(ctx: Context, which: list[str] | None = None) -> list[str]:
    steps = {
        "loeo": leave_one_event_out,
        "sensitivity": sensitivity,
        "crosscheck": crosscheck,
        "ablation": ablation,
        "walkforward": walkforward,
    }
    done = []
    for name, fn in steps.items():
        if which is None or name in which:
            fn(ctx)
            done.append(name)
    return done
