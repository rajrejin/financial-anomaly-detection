"""Phase 6: SHAP for tree models, reconstruction shares for the autoencoder, permutation
sensitivity for the isolation forest. Outputs: results/explain and results/explainability.json.
Explanations describe model behaviour, not the causes of market moves."""

from __future__ import annotations

import json
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

from finanomaly.context import Context  # noqa: E402
from finanomaly.evaluation.test_eval import load_registered  # noqa: E402
from finanomaly.experiments.promotion import Row, select_best  # noqa: E402
from finanomaly.experiments.runner import load_experiments, run_final_stage, run_val_stage  # noqa: E402
from finanomaly.explain.checks import shap_additivity  # noqa: E402

HEADLINE = ["covid_crash_2020", "ukraine_invasion_2022", "ecb_hike_2022"]


def _alarm_day(ctx: Context, scores: np.ndarray, thr: float, event_id: str) -> int:
    w = next(w for w in ctx.wins if w.event.id == event_id)
    a, b = w.detect
    hits = np.flatnonzero(np.nan_to_num(scores[a:b], nan=-np.inf) > thr)
    return int(hits[0] + a) if len(hits) else w.onset_pos


def _scores(ctx: Context, model: Any) -> np.ndarray:
    s = np.full(ctx.n, np.nan)
    X = ctx.z.to_numpy(dtype=float)
    s[ctx.valid] = model.score(X[ctx.valid])
    return s


def tree_explanations(ctx: Context, model: Any, thr: float, label: str, out_dir: Any) -> dict:
    X = ctx.z.to_numpy(dtype=float)
    names = list(ctx.z.columns)
    explainer = shap.TreeExplainer(model.booster())
    sv_all = explainer.shap_values(X[ctx.valid])
    if isinstance(sv_all, list):
        sv_all = sv_all[1]
    base = float(np.asarray(explainer.expected_value).ravel()[-1])
    add = shap_additivity(model, X[ctx.valid])
    mean_abs = np.abs(sv_all).mean(axis=0)
    # gain based importance
    booster = model.booster()
    if model.family == "lightgbm":
        gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    else:
        g = booster.get_score(importance_type="gain")
        gain = np.array([g.get(f"f{i}", 0.0) for i in range(len(names))])
    rho = spearmanr(mean_abs, gain)
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs, "gain": gain})
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(
        out_dir / f"{label}_importance.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(mean_abs)
    ax.barh(np.array(names)[order], mean_abs[order])
    ax.set_title(f"{label}: mean |SHAP| (margin space)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_global_importance.png", dpi=100)
    plt.close(fig)
    scores = _scores(ctx, model)
    days = {}
    for ev in HEADLINE:
        d = _alarm_day(ctx, scores, thr, ev)
        row = ctx.z.iloc[[d]].to_numpy(dtype=float)
        sv = explainer.shap_values(row)
        if isinstance(sv, list):
            sv = sv[1]
        expl = shap.Explanation(
            values=np.asarray(sv)[0], base_values=base, data=row[0], feature_names=names
        )
        plt.figure()
        shap.plots.waterfall(expl, show=False, max_display=10)
        plt.title(f"{label}: {ev}, alarm day {ctx.index[d].date()}", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"{label}_waterfall_{ev}.png", dpi=100)
        plt.close("all")
        top = np.argsort(-np.abs(np.asarray(sv)[0]))[:3]
        days[ev] = {
            "alarm_day": str(ctx.index[d].date()),
            "top_features": [(names[i], float(np.asarray(sv)[0][i])) for i in top],
        }
    return {
        "max_additivity_gap": add["max_additivity_gap"],
        "spearman_shap_vs_gain": float(rho.statistic),
        "top_by_shap": imp.sort_values("mean_abs_shap", ascending=False)["feature"].tolist()[:5],
        "top_by_gain": imp.sort_values("gain", ascending=False)["feature"].tolist()[:5],
        "headline_days": days,
    }


def ae_explanations(ctx: Context, model: Any, thr: float, out_dir: Any) -> dict:
    names = list(ctx.z.columns)
    scores = _scores(ctx, model)
    X = ctx.z.to_numpy(dtype=float)
    out = {}
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(13, 4), sharey=True)
    for ax, ev in zip(axes, HEADLINE, strict=True):
        d = _alarm_day(ctx, scores, thr, ev)
        share = model.explain(X[[d]])[0]
        order = np.argsort(share)
        ax.barh(np.array(names)[order], share[order])
        ax.set_title(f"{ev}\n{ctx.index[d].date()}", fontsize=8)
        top = np.argsort(-share)[:3]
        out[ev] = {
            "alarm_day": str(ctx.index[d].date()),
            "top_shares": [(names[i], float(share[i])) for i in top],
            "shares_sum": float(share.sum()),
        }
    fig.suptitle("Autoencoder: share of squared reconstruction error by feature")
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_contributions.png", dpi=100)
    plt.close(fig)
    return out


def if_permutation(ctx: Context, model: Any, n_repeats: int = 5, seed: int = 0) -> dict:
    """Shuffle one feature within the test period and measure the drop in average precision."""
    names = list(ctx.z.columns)
    a, b = ctx.test_seg
    rows = np.flatnonzero(ctx.valid[a:b]) + a
    X = ctx.z.to_numpy(dtype=float)[rows]
    y = ctx.y[rows]
    base_ap = float(average_precision_score(y, model.score(X)))
    rng = np.random.default_rng(seed)
    res = {}
    for j, n in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base_ap - float(average_precision_score(y, model.score(Xp))))
        res[n] = {"ap_drop_mean": float(np.mean(drops)), "ap_drop_std": float(np.std(drops))}
    return {"base_test_ap": base_ap, "features": res}


def best_xgboost(ctx: Context) -> tuple[Any, float, int]:
    df = pd.read_csv(ctx.cfg.path("results") / "runs_summary.csv")
    g = df[(df.family == "xgboost") & (df.seed == 0)]
    rows = [
        Row(
            int(r.config_idx),
            r.val_event_recall,
            r.val_false_alarm_rate,
            r.val_average_precision,
            r.n_parameters,
        )
        for r in g.itertuples()
    ]
    best = select_best(rows)
    params = load_experiments()["xgboost"][best.config_idx]
    v = run_val_stage(ctx, "xgboost", best.config_idx, params, 0)
    f = run_final_stage(ctx, "xgboost", best.config_idx, params, 0, v.threshold)
    return f.model, f.threshold, best.config_idx


def run_explainability(ctx: Context) -> dict[str, Any]:
    out_dir = ctx.cfg.path("results") / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = load_registered(ctx, "staging")
    report: dict[str, Any] = {}
    lg = reg["lightgbm"]
    report["lightgbm"] = tree_explanations(ctx, lg["model"], lg["threshold"], "lightgbm", out_dir)
    xmodel, xthr, xidx = best_xgboost(ctx)
    report["xgboost"] = tree_explanations(ctx, xmodel, xthr, "xgboost", out_dir)
    report["xgboost"]["config_idx"] = xidx
    ae = reg["autoencoder"]
    report["autoencoder"] = ae_explanations(ctx, ae["model"], ae["threshold"], out_dir)
    report["isolation_forest"] = if_permutation(ctx, reg["isolation_forest"]["model"])
    report["note"] = (
        "Explanations describe model behaviour, not the causes of market moves. SHAP values are "
        "in the margin (log-odds) space of the tree model before Platt calibration."
    )
    (ctx.cfg.path("results") / "explainability.json").write_text(json.dumps(report, indent=2))
    return report
