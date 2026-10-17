"""Test-period evaluation. Run once, after the promotion decision (see docs/PROMOTION_DECISION.md).

This module is the only place besides robustness analyses that touches the test segment.
"""

from __future__ import annotations

import json
import time
from typing import Any

import matplotlib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from finanomaly.context import Context  # noqa: E402
from finanomaly.evaluation.metrics import evaluate_segment  # noqa: E402
from finanomaly.evaluation.report import write_json  # noqa: E402
from finanomaly.experiments import tracking  # noqa: E402
from finanomaly.experiments.runner import (  # noqa: E402
    REGISTRY_NAMES,
    load_experiments,
    run_final_stage,
    run_val_stage,
)
from finanomaly.models.rules import RULES  # noqa: E402


def _score_all(ctx: Context, model: Any) -> np.ndarray:
    s = np.full(ctx.n, np.nan)
    X = ctx.z.to_numpy(dtype=float)
    s[ctx.valid] = model.score(X[ctx.valid])
    return s


def load_registered(ctx: Context, alias: str = "staging") -> dict[str, dict[str, Any]]:
    tracking.setup(ctx.cfg)
    out = {}
    for family, name in REGISTRY_NAMES.items():
        py = mlflow.pyfunc.load_model(f"models:/{name}@{alias}").unwrap_python_model()
        out[family] = {"name": name, "model": py.model, "threshold": py.threshold, "meta": py.meta}
    return out


def _entry(label: str, res: dict) -> dict:
    return {
        "label": label,
        "n_events": res["n_events"],
        "n_detected": res["n_detected"],
        "event_recall": res["event_recall"],
        "false_alarm_episodes": res["false_alarm_episodes"],
        "false_alarm_rate": res["false_alarm_rate"],
        "average_precision": res["average_precision"],
        "roc_auc": res["roc_auc"],
        "score_separation": res["score_separation"],
        "events": res["events"],
    }


def evaluate_test(ctx: Context) -> dict[str, Any]:
    t0 = time.time()
    cfg = ctx.cfg
    res_dir = cfg.path("results")
    gap = cfg.alarm.merge_gap_days
    seg = ctx.test_seg
    dec = json.loads((res_dir / "promotion_decision.json").read_text())
    champion = dec["champion"]["name"] if dec["champion"] else None
    base = json.loads((res_dir / "baselines_validation.json").read_text())

    entries: list[dict] = []
    all_scores: dict[str, np.ndarray] = {}
    fa_rows = []
    for name, fn in RULES.items():
        s = np.where(ctx.valid, fn(ctx), np.nan)
        r = evaluate_segment(s, base[name]["threshold"], seg, ctx.wins, gap)
        entries.append(_entry(name, r))
        all_scores[name] = s
    reg = load_registered(ctx, "staging")
    champion_events: list[dict] = []
    for info in reg.values():
        s = _score_all(ctx, info["model"])
        r = evaluate_segment(s, info["threshold"], seg, ctx.wins, gap)
        label = info["name"] + (" (champion)" if info["name"] == champion else "")
        entries.append(_entry(label, r))
        all_scores[info["name"]] = s
        if info["name"] == champion:
            champion_events = r["events"]
        for a, b in r["false_alarm_episode_list"]:
            seg_scores = s[a : b + 1]
            fa_rows.append(
                {
                    "model": info["name"],
                    "start": str(ctx.index[a].date()),
                    "end": str(ctx.index[b].date()),
                    "days": b - a + 1,
                    "peak_score": float(np.nanmax(seg_scores)),
                    "peak_date": str(ctx.index[a + int(np.nanargmax(seg_scores))].date()),
                    "threshold": info["threshold"],
                }
            )
    fa = pd.DataFrame(fa_rows)
    fa.to_csv(res_dir / "test_false_alarms.csv", index=False)
    ev_rows = []
    for e in entries:
        for ev in e["events"]:
            ev_rows.append({"entry": e["label"], **ev})
    pd.DataFrame(ev_rows).to_csv(res_dir / "test_events.csv", index=False)
    sc = pd.DataFrame(
        {k: v[seg[0] : seg[1]] for k, v in all_scores.items()},
        index=ctx.index[seg[0] : seg[1]],
    )
    sc.to_csv(res_dir / "test_scores.csv")
    _plot_scores(ctx, sc, res_dir / "figures" / "test_scores.png")
    summary = {
        "entries": [{k: v for k, v in e.items() if k != "events"} for e in entries],
        "champion": champion,
        "champion_events": champion_events,
        "note": (
            "Thresholds were fixed on pre-2019 data. Six test events give very little statistical "
            "power, so the per-event rows matter more than the aggregates."
        ),
    }
    write_json(res_dir / "test_summary.json", summary)
    write_json(
        res_dir / "test_audit.json",
        {
            "promotion_decision_mtime": (res_dir / "promotion_decision.json").stat().st_mtime,
            "test_evaluation_started": t0,
            "test_evaluation_after_decision": (res_dir / "promotion_decision.json").stat().st_mtime
            < t0,
        },
    )
    return summary


def _plot_scores(ctx: Context, sc: pd.DataFrame, path: Any) -> None:
    fig, axes = plt.subplots(len(sc.columns), 1, figsize=(11, 2.2 * len(sc.columns)), sharex=True)
    idx = sc.index.tz_localize(None)
    for ax, col in zip(np.atleast_1d(axes), sc.columns, strict=True):
        ax.plot(idx, sc[col].to_numpy(), lw=0.6)
        ax.set_ylabel(col, fontsize=7)
        for w in ctx.wins:
            if w.event.onset >= idx[0]:
                ax.axvspan(
                    ctx.index[w.detect[0]].tz_localize(None),
                    ctx.index[min(w.detect[1], ctx.n - 1)].tz_localize(None),
                    color="orange",
                    alpha=0.25,
                    lw=0,
                )
    fig.suptitle("Test period scores (orange: detection windows of curated events)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def seed_stability(ctx: Context) -> dict[str, Any]:
    """Five seeds for the registered configuration of each family; validation and test."""
    reg = json.loads((ctx.cfg.path("results") / "registration.json").read_text())
    seeds = list(load_experiments()["seeds_final"])
    gap = ctx.cfg.alarm.merge_gap_days
    out: dict[str, Any] = {}
    for info in reg.values():
        idx, params = info["config_idx"], load_experiments()[family][info["config_idx"]]
        rec, far, ap = [], [], []
        for s in seeds:
            v = run_val_stage(ctx, family, idx, params, s)
            f = run_final_stage(ctx, family, idx, params, s, v.threshold)
            r = evaluate_segment(f.scores, f.threshold, ctx.test_seg, ctx.wins, gap)
            rec.append(r["event_recall"])
            far.append(r["false_alarm_rate"])
            ap.append(r["average_precision"])
        out[family] = {
            "seeds": seeds,
            "test_recall": rec,
            "test_recall_mean": float(np.mean(rec)),
            "test_recall_std": float(np.std(rec)),
            "test_fa_rate": far,
            "test_fa_mean": float(np.mean(far)),
            "test_fa_std": float(np.std(far)),
            "test_ap_mean": float(np.mean(ap)),
            "test_ap_std": float(np.std(ap)),
            "validation": info["seed_study"],
        }
    write_json(ctx.cfg.path("results") / "seed_stability.json", out)
    return out


def rank_stability(ctx: Context) -> dict[str, Any]:
    """Do configuration rankings carry over from validation to test? (all 7 configs per family)"""
    gap = ctx.cfg.alarm.merge_gap_days
    ex = load_experiments()
    rows = []
    for family in REGISTRY_NAMES:
        for idx, params in enumerate(ex[family]):
            v = run_val_stage(ctx, family, idx, params, 0)
            f = run_final_stage(ctx, family, idx, params, 0, v.threshold)
            r = evaluate_segment(f.scores, f.threshold, ctx.test_seg, ctx.wins, gap)
            rows.append(
                {
                    "family": family,
                    "config_idx": idx,
                    "val_recall": v.val_metrics["event_recall"],
                    "val_ap": v.val_metrics["average_precision"],
                    "val_fa": v.val_metrics["false_alarm_rate"],
                    "test_recall": r["event_recall"],
                    "test_ap": r["average_precision"],
                    "test_fa": r["false_alarm_rate"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ctx.cfg.path("results") / "config_rank_stability.csv", index=False)
    out: dict[str, Any] = {}
    for family, g in df.groupby("family"):
        rho_ap = spearmanr(g["val_ap"], g["test_ap"])
        rho_rec = spearmanr(g["val_recall"], g["test_recall"])
        out[str(family)] = {
            "spearman_val_ap_vs_test_ap": float(rho_ap.statistic),
            "spearman_val_recall_vs_test_recall": (
                None if np.isnan(rho_rec.statistic) else float(rho_rec.statistic)
            ),
            "n_configs": len(g),
        }
    write_json(ctx.cfg.path("results") / "rank_stability.json", out)
    return out
