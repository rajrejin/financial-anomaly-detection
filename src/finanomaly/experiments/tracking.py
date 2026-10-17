"""MLflow tracking: one run per configuration with a deterministic run name (resumable)."""

from __future__ import annotations

import logging
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from finanomaly.config import ROOT, Config, sha256_file  # noqa: E402
from finanomaly.context import Context  # noqa: E402
from finanomaly.experiments.runner import StageResult  # noqa: E402
from finanomaly.progress import git_commit  # noqa: E402
from finanomaly.serving.pyfunc_model import AnomalyPyfunc, save_model_dir  # noqa: E402

log = logging.getLogger(__name__)


def tracking_uri(cfg: Config) -> str:
    uri = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        return "sqlite:///" + (ROOT / uri[len("sqlite:///") :]).as_posix()
    return uri


def setup(cfg: Config) -> MlflowClient:
    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(uri)


def get_experiment(client: MlflowClient, cfg: Config, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    art = (ROOT / cfg.mlflow.artifact_root / name.replace("/", "_")).resolve().as_uri()
    return client.create_experiment(name, artifact_location=art)


def finished_run(client: MlflowClient, exp_id: str, run_name: str) -> Any | None:
    runs = client.search_runs(
        [exp_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        max_results=1,
    )
    return runs[0] if runs else None


def _pip_requirements() -> list[str]:
    pk = ["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "torch", "mlflow"]
    return [f"{p}=={metadata.version(p).split('+')[0]}" for p in pk]


def score_plot(ctx: Context, res: StageResult, path: Path) -> None:
    idx = ctx.index.tz_localize(None)
    a, b = ctx.val_seg
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(idx[a:b], res.scores[a:b], lw=0.7)
    ax.axhline(res.threshold, color="tab:red", lw=0.8, ls="--", label="threshold")
    for w in ctx.wins:
        if a <= w.onset_pos < b:
            ax.axvspan(
                idx[w.detect[0]], idx[min(w.detect[1], len(idx) - 1)], color="orange", alpha=0.25
            )
    ax.set_title(f"{res.family} config {res.config_idx} seed {res.seed}: validation scores")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def flat_metrics(res: StageResult, prefix: str = "val_") -> dict[str, float]:
    m = res.val_metrics
    out: dict[str, float] = {
        f"{prefix}event_recall": float(m["event_recall"]),
        f"{prefix}false_alarm_rate": float(m["false_alarm_rate"]),
        f"{prefix}false_alarm_episodes": float(m["false_alarm_episodes"]),
        f"{prefix}n_events": float(m["n_events"]),
        f"{prefix}n_detected": float(m["n_detected"]),
        "threshold": float(res.threshold),
    }
    for k in ("average_precision", "roc_auc", "score_separation"):
        if m.get(k) is not None and not np.isnan(m[k]):
            out[f"{prefix}{k}"] = float(m[k])
    for e in m["events"]:
        out[f"{prefix}delay_{e['event']}"] = float(e["delay_days"]) if e["detected"] else -999.0
    cal = res.extra.get("calibration")
    if cal:
        out[f"{prefix}log_loss"] = cal["log_loss"]
        out[f"{prefix}brier"] = cal["brier"]
        out[f"{prefix}ece"] = cal["ece"]
    out["n_parameters"] = float(res.model.n_parameters())
    return out


def log_stage_run(
    client: MlflowClient,
    ctx: Context,
    exp_id: str,
    run_name: str,
    res: StageResult,
    stage: str,
    extra_tags: dict[str, str] | None = None,
    log_model: bool = True,
) -> str:
    cfg = ctx.cfg
    tags = {
        "family": res.family,
        "seed": str(res.seed),
        "stage": stage,
        "data_hash": ctx.data_hash,
        "events_hash": sha256_file(cfg.path("events")),
        "dataset_period": f"{cfg.splits.fit_start}..{res.cutoff}",
        "git_commit": git_commit(),
        "config_idx": str(res.config_idx),
    }
    tags.update(extra_tags or {})
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as run:
        mlflow.log_params({k: str(v) for k, v in res.params.items()})
        mlflow.log_param("fit_cutoff", res.cutoff)
        mlflow.log_metrics(flat_metrics(res))
        mlflow.log_artifact(str(ROOT / "configs" / "experiments.yaml"), "config")
        mlflow.log_artifact(str(ROOT / "configs" / "default.yaml"), "config")
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            score_plot(ctx, res, tdp / "val_scores.png")
            mlflow.log_artifact(str(tdp / "val_scores.png"), "plots")
            import json

            (tdp / "val_metrics.json").write_text(
                json.dumps(res.val_metrics, indent=2, default=str)
            )
            mlflow.log_artifact(str(tdp / "val_metrics.json"), "metrics")
            if log_model:
                log_pyfunc(ctx, res, tdp / "pack")
        return run.info.run_id


def log_pyfunc(ctx: Context, res: StageResult, pack_dir: Path) -> None:
    feats = list(ctx.z.columns)
    meta = {
        "family": res.family,
        "threshold": res.threshold,
        "features": feats,
        "params": res.params,
        "seed": res.seed,
        "fit_cutoff": res.cutoff,
        "data_hash": ctx.data_hash,
    }
    save_model_dir(res.model, pack_dir, meta)
    example = ctx.z.loc[ctx.valid].iloc[-3:].reset_index(drop=True)
    sig_out = pd.DataFrame({"score": [0.0], "alarm": [0]}).astype({"alarm": "int64"})
    signature = infer_signature(example, sig_out)
    mlflow.pyfunc.log_model(
        name="model",
        python_model=AnomalyPyfunc(),
        artifacts={"model_dir": str(pack_dir)},
        signature=signature,
        input_example=example,
        pip_requirements=_pip_requirements(),
    )
