"""Experiment orchestration: run all configurations of a family and export summaries."""

from __future__ import annotations

import logging
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from finanomaly.config import Config  # noqa: E402
from finanomaly.context import Context  # noqa: E402
from finanomaly.experiments import tracking  # noqa: E402
from finanomaly.experiments.runner import (  # noqa: E402
    FAMILIES,
    load_experiments,
    run_val_stage,
)

log = logging.getLogger(__name__)


def val_run_name(family: str, idx: int, seed: int) -> str:
    return f"{family}-cfg{idx}-seed{seed}-val"


def run_family(ctx: Context, family: str, seed: int = 0) -> dict[str, Any]:
    """Run all configurations of one family on the validation stage (resumable)."""
    cfg = ctx.cfg
    client = tracking.setup(cfg)
    exp_id = tracking.get_experiment(client, cfg, f"anomaly/{family}")
    configs = load_experiments()[family]
    done, ran = 0, 0
    t0 = time.time()
    for idx, params in enumerate(configs):
        name = val_run_name(family, idx, seed)
        if tracking.finished_run(client, exp_id, name) is not None:
            log.info("skip finished run %s", name)
            done += 1
            continue
        res = run_val_stage(ctx, family, idx, params, seed)
        tracking.log_stage_run(client, ctx, exp_id, name, res, "validation")
        ran += 1
        log.info("finished %s", name)
    return {"family": family, "skipped": done, "ran": ran, "seconds": round(time.time() - t0, 1)}


def runs_dataframe(cfg: Config) -> pd.DataFrame:
    client = tracking.setup(cfg)
    exps = [e for e in client.search_experiments() if e.name.startswith("anomaly/")]
    df = mlflow.search_runs(experiment_ids=[e.experiment_id for e in exps], output_format="pandas")
    return df


def export_summary(cfg: Config) -> pd.DataFrame:
    df = runs_dataframe(cfg)
    if df.empty:
        return df
    df = df[df["tags.stage"] == "validation"]
    cols = ["tags.family", "tags.config_idx", "tags.seed", "run_id", "status"]
    cols += [c for c in df.columns if c.startswith("metrics.val_") and "delay_" not in c]
    cols += ["metrics.threshold", "metrics.n_parameters"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.columns = [c.replace("tags.", "").replace("metrics.", "") for c in out.columns]
    out["config_idx"] = out["config_idx"].astype(int)
    out = out.sort_values(["family", "config_idx", "seed"])
    path = cfg.path("results") / "runs_summary.csv"
    out.to_csv(path, index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam, g in out.groupby("family"):
        ax.scatter(g["val_false_alarm_rate"], g["val_event_recall"], label=fam, s=40, alpha=0.7)
    ax.set_xlabel("validation false alarm episodes per year")
    ax.set_ylabel("validation event recall")
    ax.set_title("All configurations (validation, 2015 to 2018)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.path("results") / "figures" / "runs_comparison.png", dpi=110)
    plt.close(fig)
    return out


__all__ = ["FAMILIES", "run_family", "export_summary"]
