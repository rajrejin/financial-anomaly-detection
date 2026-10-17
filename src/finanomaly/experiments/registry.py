"""Model Registry: register the best model of each family, set aliases and promote.

MLflow 3 replaced registry stages with aliases, so `staging` and `production` are aliases here
(see docs/DECISIONS.md). A `lifecycle` tag mirrors the alias for readability.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from mlflow.tracking import MlflowClient

from finanomaly.config import Config
from finanomaly.context import Context
from finanomaly.experiments import tracking
from finanomaly.experiments.promotion import Candidate, GateConfig, Row, apply_gate, select_best
from finanomaly.experiments.runner import (
    REGISTRY_NAMES,
    load_experiments,
    run_final_stage,
    run_val_stage,
)
from finanomaly.experiments.summary import val_run_name
from finanomaly.explain.checks import check_explanations
from finanomaly.serving.contract import serving_contract

log = logging.getLogger(__name__)
FAMILY_BY_NAME = {v: k for k, v in REGISTRY_NAMES.items()}


def best_config(client: MlflowClient, exp_id: str) -> tuple[Row, str]:
    runs = client.search_runs(
        [exp_id],
        filter_string="tags.stage = 'validation' and tags.seed = '0' "
        "and attributes.status = 'FINISHED'",
        max_results=100,
    )
    rows, ids = [], {}
    for r in runs:
        m = r.data.metrics
        idx = int(r.data.tags["config_idx"])
        rows.append(
            Row(
                idx,
                m["val_event_recall"],
                m["val_false_alarm_rate"],
                m.get("val_average_precision"),
                m.get("n_parameters", 0.0),
            )
        )
        ids[idx] = r.info.run_id
    best = select_best(rows)
    return best, ids[best.config_idx]


def seed_study(
    ctx: Context, client: MlflowClient, family: str, exp_id: str, idx: int, seeds: list[int]
) -> dict[str, Any]:
    """Run the best configuration over several seeds on the validation stage (resumable)."""
    params = load_experiments()[family][idx]
    recalls, fas, aps = [], [], []
    for s in seeds:
        name = val_run_name(family, idx, s)
        run = tracking.finished_run(client, exp_id, name)
        if run is None:
            res = run_val_stage(ctx, family, idx, params, s)
            tracking.log_stage_run(client, ctx, exp_id, name, res, "validation", log_model=False)
            m = res.val_metrics
            recalls.append(m["event_recall"])
            fas.append(m["false_alarm_rate"])
            aps.append(m["average_precision"] or 0.0)
        else:
            m = run.data.metrics
            recalls.append(m["val_event_recall"])
            fas.append(m["val_false_alarm_rate"])
            aps.append(m.get("val_average_precision", 0.0))
    return {
        "seeds": seeds,
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls, ddof=0)),
        "fa_mean": float(np.mean(fas)),
        "fa_std": float(np.std(fas, ddof=0)),
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps, ddof=0)),
        "recalls": recalls,
        "false_alarm_rates": fas,
    }


def register_all(ctx: Context, families: list[str] | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    client = tracking.setup(cfg)
    seeds = list(load_experiments()["seeds_final"])
    out: dict[str, Any] = {}
    for family, reg_name in REGISTRY_NAMES.items():
        if families and family not in families:
            continue
        exp_id = tracking.get_experiment(client, cfg, f"anomaly/{family}")
        best, val_run_id = best_config(client, exp_id)
        spread = seed_study(ctx, client, family, exp_id, best.config_idx, seeds)
        val_run = client.get_run(val_run_id)
        val_thr = val_run.data.metrics["threshold"]
        params = load_experiments()[family][best.config_idx]
        final_name = f"{family}-cfg{best.config_idx}-final-seed0"
        existing = tracking.finished_run(client, exp_id, final_name)
        if existing is None:
            res = run_final_stage(ctx, family, best.config_idx, params, 0, val_thr)
            run_id = tracking.log_stage_run(client, ctx, exp_id, final_name, res, "final")
            for k, v in spread.items():
                if isinstance(v, float):
                    client.log_metric(run_id, f"seedstudy_{k}", v)
            client.set_tag(run_id, "val_stage_run_id", val_run_id)
        else:
            run_id = existing.info.run_id
        mv = _register_version(client, reg_name, run_id, ctx, best, spread, val_run_id)
        client.set_registered_model_alias(reg_name, "staging", mv.version)
        client.set_model_version_tag(reg_name, mv.version, "lifecycle", "staging")
        out[family] = {
            "name": reg_name,
            "version": int(mv.version),
            "config_idx": best.config_idx,
            "run_id": run_id,
            "val_stage_run_id": val_run_id,
            "seed_study": spread,
        }
    path = cfg.path("results") / "registration.json"
    path.write_text(json.dumps(out, indent=2))
    return out


def _register_version(
    client: MlflowClient,
    name: str,
    run_id: str,
    ctx: Context,
    best: Row,
    spread: dict,
    val_run_id: str,
) -> Any:
    import mlflow

    existing = [v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id]
    if existing:
        return existing[0]
    try:
        client.create_registered_model(name, description=f"Anomaly detector ({name}).")
    except Exception:  # already exists
        pass
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    desc = (
        f"Config {best.config_idx}, refit on data before {ctx.cfg.splits.test_start}. "
        f"Validation (2015 to 2018, fit before 2015) event recall {best.recall:.3f}, false alarm "
        f"episodes per year {best.fa_rate:.2f}. Data hash {ctx.data_hash[:12]}."
    )
    client.update_model_version(name, mv.version, description=desc)
    for k, v in {
        "data_hash": ctx.data_hash,
        "dataset_period": f"{ctx.cfg.splits.fit_start}..{ctx.cfg.splits.test_start}",
        "val_event_recall": f"{best.recall:.4f}",
        "val_false_alarm_rate": f"{best.fa_rate:.4f}",
        "val_stage_run_id": val_run_id,
        "seed_recall_std": f"{spread['recall_std']:.4f}",
    }.items():
        client.set_model_version_tag(name, mv.version, k, v)
    return mv


def _candidate_from_alias(
    client: MlflowClient, ctx: Context, name: str, alias: str, check: bool
) -> Candidate | None:
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except Exception:
        return None
    tags = mv.tags
    family = FAMILY_BY_NAME[name]
    explanations_ok = contract_ok = True
    if check:
        model_uri = f"models:/{name}@{alias}"
        contract_ok = bool(serving_contract(ctx, model_uri)["ok"])
        import mlflow.pyfunc

        py = mlflow.pyfunc.load_model(model_uri).unwrap_python_model()
        X = ctx.z.loc[ctx.valid].to_numpy(dtype=float)[-200:]
        explanations_ok = bool(check_explanations(py.model, X)["available"])
    run = client.get_run(tags["val_stage_run_id"])
    ap = run.data.metrics.get("val_average_precision", 0.0)
    return Candidate(
        name=name,
        family=family,
        version=int(mv.version),
        recall=float(tags["val_event_recall"]),
        fa_rate=float(tags["val_false_alarm_rate"]),
        ap=float(ap),
        recall_std=float(tags["seed_recall_std"]),
        explanations_ok=explanations_ok,
        contract_ok=contract_ok,
    )


def run_promotion(ctx: Context) -> dict[str, Any]:
    cfg: Config = ctx.cfg
    client = tracking.setup(cfg)
    base = json.loads((cfg.path("results") / "baselines_validation.json").read_text())
    baseline_recall = max(v["validation"]["event_recall"] for v in base.values())
    cands, production = [], None
    for name in REGISTRY_NAMES.values():
        prod = _candidate_from_alias(client, ctx, name, "production", check=False)
        if prod is not None:
            production = prod
        c = _candidate_from_alias(client, ctx, name, "staging", check=True)
        if c is not None:
            cands.append(c)
    gate = GateConfig(
        budget_per_year=cfg.alarm.budget_per_year, fa_tolerance=0.5, max_recall_std=0.2
    )
    decision = apply_gate(cands, baseline_recall, production, gate)
    champ = decision["champion"]
    if champ is not None:
        name, version = champ["name"], champ["version"]
        for other in REGISTRY_NAMES.values():
            try:
                cur = client.get_model_version_by_alias(other, "production")
                client.delete_registered_model_alias(other, "production")
                client.set_model_version_tag(other, cur.version, "lifecycle", "archived")
            except Exception:
                pass
        client.set_registered_model_alias(name, "production", version)
        client.set_model_version_tag(name, version, "lifecycle", "production")
    path = cfg.path("results") / "promotion_decision.json"
    path.write_text(json.dumps(decision, indent=2))
    return decision
