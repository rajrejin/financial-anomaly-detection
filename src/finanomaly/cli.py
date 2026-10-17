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
