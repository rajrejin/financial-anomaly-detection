import pytest

from finanomaly.config import load_config
from finanomaly.experiments import tracking


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory):
    cfg = load_config()
    if not (cfg.path("processed") / "daily.parquet").exists():
        pytest.skip("data cache absent")
    try:
        client = tracking.setup(cfg)
        from finanomaly.serving.export import find_production

        find_production(client)
    except Exception as exc:  # no registry in this checkout
        pytest.skip(f"no registered production model: {exc}")
    from finanomaly.serving.export import export_model

    d = tmp_path_factory.mktemp("export")
    export_model(cfg, d)
    return d
