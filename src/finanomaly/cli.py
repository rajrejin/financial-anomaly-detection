"""finanomaly command line interface."""

from __future__ import annotations

import typer

from finanomaly.logging_utils import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Financial anomaly detection")


@app.callback()
def _main() -> None:
    setup_logging()


@app.command()
def env() -> None:
    """Write results/environment.json."""
    from finanomaly.environment import write_environment

    typer.echo(write_environment())


@app.command()
def fetch() -> None:
    """Download all raw series (cached) and write data/series_catalog.csv."""
    from finanomaly.config import get_config
    from finanomaly.data.build import fetch_all

    typer.echo(fetch_all(get_config()).to_string())


@app.command("build-dataset")
def build_dataset_cmd() -> None:
    """Build data/processed/daily.parquet."""
    from finanomaly.config import get_config
    from finanomaly.data.build import build_dataset

    t = build_dataset(get_config())
    typer.echo(f"{t.shape} {t.index.min().date()} {t.index.max().date()}")


@app.command()
def quality() -> None:
    """Write results/data_quality.json."""
    from finanomaly.config import get_config
    from finanomaly.data.quality import quality_ok, run_quality

    rep = run_quality(get_config())
    typer.echo(quality_ok(rep))


@app.command()
def eda() -> None:
    """Write exploratory figures to results/figures/eda."""
    from finanomaly.config import get_config
    from finanomaly.data.eda import run_eda

    for f in run_eda(get_config()):
        typer.echo(f)


@app.command("events-freeze")
def events_freeze() -> None:
    """Write results/events_frozen.json (hashes of events and window config)."""
    from finanomaly.config import get_config
    from finanomaly.events import freeze

    typer.echo(freeze(get_config()))


@app.command()
def baselines() -> None:
    """Evaluate the rule baselines on the validation period."""
    from finanomaly.config import get_config
    from finanomaly.context import load_context
    from finanomaly.evaluation.report import evaluate_baselines_validation

    ctx = load_context(get_config())
    out = evaluate_baselines_validation(ctx)
    for k, v in out.items():
        r = v["validation"]
        typer.echo(
            f"{k}: thr={v['threshold']:.3f} recall={r['event_recall']} "
            f"fa_rate={r['false_alarm_rate']:.2f} ap={r['average_precision']}"
        )


@app.command()
def experiment(
    family: str = typer.Option(..., help="isolation_forest|autoencoder|lightgbm|xgboost|all"),
    seed: int = typer.Option(0),
) -> None:
    """Run all 7 configurations of a family (resumable) and refresh results/runs_summary.csv."""
    from finanomaly.config import get_config
    from finanomaly.context import load_context
    from finanomaly.experiments.runner import FAMILIES
    from finanomaly.experiments.summary import export_summary, run_family

    cfg = get_config()
    ctx = load_context(cfg)
    fams = list(FAMILIES) if family == "all" else [family]
    for f in fams:
        typer.echo(run_family(ctx, f, seed))
    typer.echo(f"{len(export_summary(cfg))} validation runs in results/runs_summary.csv")


@app.command()
def register() -> None:
    """Register the best configuration of each family and set the staging alias."""
    from finanomaly.config import get_config
    from finanomaly.context import load_context
    from finanomaly.experiments.registry import register_all

    ctx = load_context(get_config())
    for fam, info in register_all(ctx).items():
        typer.echo(f"{fam}: {info['name']} v{info['version']} cfg{info['config_idx']}")


@app.command()
def promote() -> None:
    """Run the promotion gate, set the production alias and write the decision document."""
    from finanomaly.config import get_config
    from finanomaly.context import load_context
    from finanomaly.experiments.decision_doc import write_decision_doc
    from finanomaly.experiments.registry import run_promotion

    cfg = get_config()
    decision = run_promotion(load_context(cfg))
    write_decision_doc(cfg)
    champ = decision["champion"]
    typer.echo(f"champion: {champ['name'] if champ else None}")


@app.command()
def evaluate(
    stability: bool = typer.Option(True, help="also run seed and rank stability"),
) -> None:
    """Test-period evaluation (run once, after the promotion decision)."""
    from finanomaly.config import get_config
    from finanomaly.context import load_context
    from finanomaly.evaluation.test_eval import evaluate_test, rank_stability, seed_stability
    from finanomaly.experiments.decision_doc import write_decision_doc

    cfg = get_config()
    ctx = load_context(cfg)
    summary = evaluate_test(ctx)
    for e in summary["entries"]:
        typer.echo(
            f"{e['label']}: {e['n_detected']}/{e['n_events']} fa={e['false_alarm_rate']:.2f} "
            f"ap={e['average_precision']}"
        )
    if stability:
        seed_stability(ctx)
        rank_stability(ctx)
    write_decision_doc(cfg)


@app.command()
def review() -> None:
    """Write docs/FALSE_ALARM_REVIEW.md from results/test_false_alarms.csv."""
    from finanomaly.config import get_config
    from finanomaly.evaluation.review import write_review

    typer.echo(write_review(get_config())[:600])


@app.command()
def explain() -> None:
    """SHAP, reconstruction shares and permutation sensitivity (results/explain)."""
    from finanomaly.config import get_config
    from finanomaly.context import load_context
    from finanomaly.explain.report import run_explainability

    rep = run_explainability(load_context(get_config()))
    for k in ("lightgbm", "xgboost"):
        typer.echo(
            f"{k}: additivity gap {rep[k]['max_additivity_gap']:.2e}, "
            f"rho(shap,gain)={rep[k]['spearman_shap_vs_gain']:.2f}"
        )


@app.command("export-model")
def export_model_cmd(out: str = typer.Option("", help="output directory")) -> None:
    """Export the production-alias model to a self-contained directory."""
    from pathlib import Path

    from finanomaly.config import get_config
    from finanomaly.serving.export import export_model

    m = export_model(get_config(), Path(out) if out else None)
    typer.echo({k: v for k, v in m.items() if k != "files"})


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI service (needs an exported model directory)."""
    import uvicorn

    from finanomaly.config import ROOT

    uvicorn.run("api.main:app", host=host, port=port, app_dir=str(ROOT))


@app.command()
def refresh() -> None:
    """Fetch the latest days, rebuild, run quality checks and publish only if they pass."""
    from finanomaly.config import get_config
    from finanomaly.data.refresh import refresh as do_refresh

    typer.echo(do_refresh(get_config()))


@app.command()
def robustness(
    only: str = typer.Option(
        "", help="comma separated: loeo,sensitivity,crosscheck,ablation,walkforward"
    ),
) -> None:
    """Supplementary analyses (leave-one-event-out, sensitivity, cross-checks, walk-forward)."""
    from finanomaly.config import get_config
    from finanomaly.context import load_context
    from finanomaly.evaluation.robustness import run_all

    which = [s for s in only.split(",") if s] or None
    typer.echo(run_all(load_context(get_config()), which))
