import numpy as np
import pandas as pd
import pytest

from finanomaly.config import load_config
from finanomaly.events import load_events
from finanomaly.features import BASE_FEATURES, build_features, trailing_z
from finanomaly.splits import make_split, training_labels

CFG = load_config()


def _table():
    path = CFG.path("processed") / "daily.parquet"
    if not path.exists():
        pytest.skip("data cache absent")
    return pd.read_parquet(path)


def test_trailing_z_uses_only_past_hand_computed():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0, 5.0]})
    z = trailing_z(x, 4)
    # day 4: window [1,2,3,4], median 2.5, MAD = median(|1.5,.5,.5,1.5|)=1.0 -> scale 1.4826
    assert abs(z["a"].iloc[4] - (100 - 2.5) / 1.4826) < 1e-9 or z["a"].iloc[4] == 10.0
    assert np.isnan(z["a"].iloc[:4]).all()
    # day 5 window [2,3,4,100]: the current value 5 has no influence on its own statistics
    assert abs(z["a"].iloc[5] - (5 - 3.5) / (1.4826 * 1.0)) < 1e-9


@pytest.mark.needs_data
def test_no_future_leakage_truncation_equivalence():
    t = _table()
    base_full, z_full = build_features(t, CFG)
    for cut in (1500, 3500, 5200, 6900):
        t_cut = t.iloc[: cut + 1]
        base_cut, z_cut = build_features(t_cut, CFG)
        pd.testing.assert_series_equal(
            base_full.iloc[cut], base_cut.iloc[cut], check_names=False, rtol=1e-9, atol=1e-12
        )
        pd.testing.assert_series_equal(
            z_full.iloc[cut], z_cut.iloc[cut], check_names=False, rtol=1e-9, atol=1e-12
        )
    assert list(base_full.columns) == BASE_FEATURES


@pytest.mark.needs_data
def test_perturbing_the_future_does_not_change_the_past():
    t = _table()
    _, z_a = build_features(t, CFG)
    t2 = t.copy()
    cut = 4000
    t2.iloc[cut + 1 :, t2.columns.get_loc("dax_close")] *= 1.5
    _, z_b = build_features(t2, CFG)
    pd.testing.assert_frame_equal(z_a.iloc[: cut + 1], z_b.iloc[: cut + 1])


@pytest.mark.needs_data
def test_no_test_label_window_in_training():
    from finanomaly.events import event_windows

    t = _table()
    wins = event_windows(CFG, t.index, load_events(CFG))
    cutoff = pd.Timestamp(CFG.splits.test_start)
    y, rows = training_labels(wins, len(t), t.index, cutoff)
    split = make_split(CFG, t.index)
    assert not (rows & split.test).any()
    for w in wins:
        if w.event.onset >= cutoff:
            a, b = w.label
            assert not (y[a:b][rows[a:b]]).any()
    # splits are disjoint and ordered
    assert not (split.fit & split.val).any() and not (split.val & split.test).any()
