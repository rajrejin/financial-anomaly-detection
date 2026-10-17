"""Shared fixtures. `project` runs the whole pipeline through the CLI on real cached data in a
temporary project root, with small experiment grids (2 configurations per family, 2 seeds and
short autoencoder training) so it takes about a minute. Skips when the raw data cache is absent."""

import shutil
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from finanomaly.cli import app
from finanomaly.config import ROOT, load_config

RAW = ROOT / "data" / "raw"


def run_cli(args):
    res = CliRunner().invoke(app, args, catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res.output


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    if not RAW.exists() or not (ROOT / "data" / "series_catalog.csv").exists():
        pytest.skip("raw data cache absent (run `finanomaly fetch` first)")
    root = tmp_path_factory.mktemp("proj")
    shutil.copytree(RAW, root / "data" / "raw")
    shutil.copy(ROOT / "data" / "events.yaml", root / "data" / "events.yaml")
    (root / "results" / "figures").mkdir(parents=True)
    (root / "docs").mkdir()
    ex = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
    small = {"seeds_final": [0, 1]}
    for fam in ("isolation_forest", "autoencoder", "lightgbm", "xgboost"):
        small[fam] = ex[fam][:2]
    (root / "experiments.yaml").write_text(yaml.safe_dump(small))
    cfg = load_config().model_copy(update={"root": str(root)})
    mp = pytest.MonkeyPatch()
    mp.setattr("finanomaly.config.get_config", lambda: cfg)
    mp.setenv("FINANOMALY_EXPERIMENTS", str(root / "experiments.yaml"))
    mp.setattr("finanomaly.models.autoencoder.MAX_EPOCHS", 6)
    log = {}
    run_cli(["env"])
    run_cli(["fetch"])  # served from the copied cache, no network
    run_cli(["build-dataset"])
    log["quality"] = run_cli(["quality"])
    run_cli(["events-freeze"])
    run_cli(["eda"])
    run_cli(["baselines"])
    run_cli(["experiment", "--family", "all"])
    run_cli(["experiment", "--family", "isolation_forest"])  # resumable: nothing is rerun
    log["register"] = run_cli(["register"])
    log["promote"] = run_cli(["promote"])
    log["doc_before_test"] = (root / "docs" / "PROMOTION_DECISION.md").read_text()
    run_cli(["evaluate"])
    run_cli(["review"])
    run_cli(["explain"])
    run_cli(["robustness", "--only", "crosscheck,walkforward"])
    export = root / "export"
    run_cli(["export-model", "--out", str(export)])
    yield SimpleNamespace(root=root, cfg=cfg, export=export, log=log)
    mp.undo()
