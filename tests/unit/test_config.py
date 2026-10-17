from finanomaly.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.windows.label_days == 10
    assert cfg.splits.test_start == "2019-01-01"
