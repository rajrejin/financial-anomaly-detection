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
