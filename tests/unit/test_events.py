import pandas as pd
import pytest

from finanomaly.config import load_config
from finanomaly.events import check_frozen, event_windows, label_series, load_events

CFG = load_config()


def test_events_frozen():
    assert check_frozen(CFG), "events.yaml or window config changed after the freeze"


def test_event_list_meets_minimum():
    ev = load_events(CFG)
    assert len(ev) >= 10
    test_ev = [e for e in ev if e.onset >= pd.Timestamp(CFG.splits.test_start)]
    assert len(test_ev) >= 4
    assert all(len(e.sources) >= 1 and e.justification for e in ev)


def test_windows_hand_computed():
    # 30 consecutive business days; onset at position 10
    idx = pd.bdate_range("2020-01-01", periods=30, tz="UTC")
    ev = load_events(CFG)[0]
    ev = type(ev)(
        ev.id, ev.name, ev.category, idx[10].tz_localize(None), ev.sources, ev.justification
    )
    (w,) = event_windows(CFG, idx, [ev])
    assert w.label == (10, 20)  # onset plus next 9 days
    assert w.detect == (8, 30)  # 2 before, 20 after (clipped to the end)
    assert w.buffer == (30, 30)
    y = label_series([w], 30)
    assert y.sum() == 10 and y[10] == 1 and y[19] == 1 and y[20] == 0


@pytest.mark.needs_data
def test_events_in_data_range():
    from finanomaly.data.build import load_dataset

    path = CFG.path("processed") / "daily.parquet"
    if not path.exists():
        pytest.skip("data cache absent")
    t = load_dataset(CFG)
    wins = event_windows(CFG, t.index, load_events(CFG))
    assert len(wins) == len(load_events(CFG))
