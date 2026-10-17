import numpy as np
import pytest

from finanomaly.config import load_config
from finanomaly.context import load_context
from finanomaly.experiments.runner import load_experiments, run_val_stage
from finanomaly.explain.checks import check_explanations

CFG = load_config()


@pytest.fixture(scope="module")
def ctx():
    if not (CFG.path("processed") / "daily.parquet").exists():
        pytest.skip("data cache absent")
    return load_context(CFG)


@pytest.mark.needs_data
@pytest.mark.parametrize("family", ["isolation_forest", "autoencoder", "lightgbm", "xgboost"])
def test_family_runs_and_explains(ctx, family, tmp_path):
    params = load_experiments()[family][0]
    res = run_val_stage(ctx, family, 0, params, seed=0)
    assert np.isfinite(res.scores[ctx.valid]).all()
    assert 0.0 <= res.val_metrics["event_recall"] <= 1.0
    X = ctx.z.loc[ctx.valid].to_numpy(dtype=float)[-50:]
    chk = check_explanations(res.model, X)
    assert chk["available"], chk
    # save / load round trip gives identical scores
    res.model.save(tmp_path)
    again = type(res.model).load(tmp_path)
    np.testing.assert_allclose(again.score(X), res.model.score(X), rtol=1e-5, atol=1e-6)


@pytest.mark.needs_data
def test_reproducible_rerun_matches(ctx):
    params = load_experiments()["isolation_forest"][0]
    a = run_val_stage(ctx, "isolation_forest", 0, params, seed=0)
    b = run_val_stage(ctx, "isolation_forest", 0, params, seed=0)
    assert a.threshold == b.threshold
    assert a.val_metrics["event_recall"] == b.val_metrics["event_recall"]
    np.testing.assert_allclose(a.scores[ctx.valid], b.scores[ctx.valid])


@pytest.mark.needs_data
def test_supervised_training_excludes_test_events(ctx):
    # the label vector used for the validation stage must not contain test-event windows
    cutoff = np.datetime64(CFG.splits.val_start)
    idx = ctx.index.tz_localize(None).to_numpy()
    for w in ctx.wins:
        if w.event.onset >= __import__("pandas").Timestamp(cutoff):
            a, b = w.label
            assert (idx[a:b] >= cutoff).all()
