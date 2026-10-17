"""Whole pipeline through the CLI on real cached data, in a temporary project root.

Uses small experiment grids (2 configurations per family, 2 seeds, short autoencoder training)
so the run takes about a minute. Skips when the raw data cache is absent.
"""

import json
import shutil

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from finanomaly.cli import app
from finanomaly.config import ROOT, load_config

RAW = ROOT / "data" / "raw"
pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    if not RAW.exists() or not (ROOT / "data" / "series_catalog.csv").exists():
        pytest.skip("raw data cache absent")
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
    yield root, cfg
    mp.undo()


def run(args):
    res = CliRunner().invoke(app, args, catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res.output


def test_pipeline_end_to_end(sandbox):
    root, cfg = sandbox
    run(["env"])
    run(["fetch"])
    assert (root / "data" / "series_catalog.csv").exists()
    run(["build-dataset"])
    out = run(["quality"])
    assert "True" in out
    run(["events-freeze"])
    assert json.loads((root / "results" / "events_frozen.json").read_text())["n_events"] == 15
    run(["eda"])
    assert (root / "results" / "figures" / "eda" / "series_events.png").exists()
    run(["baselines"])
    run(["experiment", "--family", "all"])
    summary = pd.read_csv(root / "results" / "runs_summary.csv")
    assert len(summary) == 8 and set(summary.family) == {
        "isolation_forest",
        "autoencoder",
        "lightgbm",
        "xgboost",
    }
    run(["experiment", "--family", "isolation_forest"])  # resumable: nothing is rerun
    out = run(["register"])
    assert "anomaly-lightgbm" in out
    out = run(["promote"])
    assert "champion:" in out
    doc = (root / "docs" / "PROMOTION_DECISION.md").read_text()
    assert "Promotion decision" in doc and "Not yet evaluated" in doc
    run(["evaluate"])
    doc = (root / "docs" / "PROMOTION_DECISION.md").read_text()
    assert "Per-event outcome of the champion" in doc
    audit = json.loads((root / "results" / "test_audit.json").read_text())
    assert audit["test_evaluation_after_decision"] is True
    run(["review"])
    assert (root / "docs" / "FALSE_ALARM_REVIEW.md").exists()
    run(["explain"])
    rep = json.loads((root / "results" / "explainability.json").read_text())
    assert rep["lightgbm"]["max_additivity_gap"] < 1e-6
    assert abs(sum(v["shares_sum"] for v in rep["autoencoder"].values()) - 3) < 1e-6
    out_dir = root / "export"
    run(["export-model", "--out", str(out_dir)])
    assert (out_dir / "manifest.json").exists()
    # a second export into the same directory replaces it; an unrelated directory is refused
    run(["export-model", "--out", str(out_dir)])
    other = root / "unrelated"
    other.mkdir()
    (other / "keep.txt").write_text("x")
    with pytest.raises(RuntimeError):
        CliRunner().invoke(app, ["export-model", "--out", str(other)], catch_exceptions=False)
    assert (other / "keep.txt").exists()


def test_registered_production_alias_and_audit(sandbox):
    root, cfg = sandbox
    dec = json.loads((root / "results" / "promotion_decision.json").read_text())
    reg = json.loads((root / "results" / "registration.json").read_text())
    assert set(reg) == {"isolation_forest", "autoencoder", "lightgbm"}
    if dec["champion"]:
        from finanomaly.experiments import tracking

        client = tracking.setup(cfg)
        mv = client.get_model_version_by_alias(dec["champion"]["name"], "production")
        assert int(mv.version) == dec["champion"]["version"]


def test_refresh_publishes_only_when_quality_passes(sandbox, monkeypatch):
    root, cfg = sandbox
    from finanomaly.data import refresh as refresh_mod
    from finanomaly.data.build import load_dataset

    table = load_dataset(cfg)
    export = root / "export"
    monkeypatch.setattr(refresh_mod, "build_dataset", lambda c, force, write: table)
    ok = refresh_mod.refresh(cfg, export)
    assert ok["quality_ok"] is True and "published" in ok
    bad_table = table.iloc[:-20]  # older than the stored data and with fewer rows
    monkeypatch.setattr(refresh_mod, "build_dataset", lambda c, force, write: bad_table)
    bad = refresh_mod.refresh(cfg, export)
    assert bad["quality_ok"] is False and bad["problems"]
    status = json.loads((export / "history" / "status.json").read_text())
    assert status["quality_ok"] is False
    assert len(load_dataset(cfg)) == len(table)  # previous good data kept

    def boom(c, force, write):
        raise RuntimeError("network down")

    monkeypatch.setattr(refresh_mod, "build_dataset", boom)
    failed = refresh_mod.refresh(cfg, export)
    assert failed["quality_ok"] is False and "fetch failed" in failed["problems"][0]
