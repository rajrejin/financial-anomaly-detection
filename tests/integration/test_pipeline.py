"""Assertions on the outputs of the full CLI pipeline run by the `project` fixture."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from finanomaly.cli import app

pytestmark = pytest.mark.needs_data


def test_data_and_freeze(project):
    root = project.root
    assert (root / "data" / "series_catalog.csv").exists()
    assert "True" in project.log["quality"]
    frozen = json.loads((root / "results" / "events_frozen.json").read_text())
    assert frozen["n_events"] == 15
    assert (root / "results" / "figures" / "eda" / "series_events.png").exists()


def test_runs_logged_and_resumable(project):
    summary = pd.read_csv(project.root / "results" / "runs_summary.csv")
    assert len(summary) == 8
    assert set(summary.family) == {"isolation_forest", "autoencoder", "lightgbm", "xgboost"}


def test_promotion_then_test_evaluation_order(project):
    root = project.root
    assert "anomaly-lightgbm" in project.log["register"]
    assert "champion:" in project.log["promote"]
    assert "Not yet evaluated" in project.log["doc_before_test"]
    doc = (root / "docs" / "PROMOTION_DECISION.md").read_text()
    assert "Per-event outcome of the champion" in doc
    audit = json.loads((root / "results" / "test_audit.json").read_text())
    assert audit["test_evaluation_after_decision"] is True
    assert (root / "docs" / "FALSE_ALARM_REVIEW.md").exists()


def test_production_alias_matches_decision(project):
    root, cfg = project.root, project.cfg
    dec = json.loads((root / "results" / "promotion_decision.json").read_text())
    reg = json.loads((root / "results" / "registration.json").read_text())
    assert set(reg) == {"isolation_forest", "autoencoder", "lightgbm"}
    if dec["champion"]:
        from finanomaly.experiments import tracking

        mv = tracking.setup(cfg).get_model_version_by_alias(dec["champion"]["name"], "production")
        assert int(mv.version) == dec["champion"]["version"]


def test_explanations_checks(project):
    rep = json.loads((project.root / "results" / "explainability.json").read_text())
    assert rep["lightgbm"]["max_additivity_gap"] < 1e-6
    assert abs(sum(v["shares_sum"] for v in rep["autoencoder"].values()) - 3) < 1e-6


def test_export_replaces_only_export_dirs(project):
    out = project.export
    assert (out / "manifest.json").exists()
    res = CliRunner().invoke(app, ["export-model", "--out", str(out)], catch_exceptions=False)
    assert res.exit_code == 0
    other = project.root / "unrelated"
    other.mkdir()
    (other / "keep.txt").write_text("x")
    with pytest.raises(RuntimeError):
        CliRunner().invoke(app, ["export-model", "--out", str(other)], catch_exceptions=False)
    assert (other / "keep.txt").exists()


def test_refresh_publishes_only_when_quality_passes(project, monkeypatch):
    cfg, export = project.cfg, project.export
    from finanomaly.data import refresh as refresh_mod
    from finanomaly.data.build import load_dataset

    table = load_dataset(cfg)
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
    # restore a published status so later tests see a healthy export
    monkeypatch.setattr(refresh_mod, "build_dataset", lambda c, force, write: table)
    assert refresh_mod.refresh(cfg, export)["quality_ok"] is True
