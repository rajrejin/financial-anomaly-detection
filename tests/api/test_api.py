import json
import shutil

import pytest
from api.main import create_app
from api.settings import Settings
from fastapi.testclient import TestClient


def client_for(path, **kw):
    return TestClient(create_app(Settings(model_dir=str(path), json_logs=False, **kw)))


@pytest.fixture(scope="module")
def client(export_dir):
    with client_for(export_dir) as c:
        yield c


def test_health_ready_model(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    r = client.get("/v1/ready")
    assert r.status_code == 200 and r.json()["model_loaded"]
    m = client.get("/v1/model").json()
    assert m["alias"] == "production" and len(m["features"]) == 16
    assert "files" not in m["manifest"] and m["version"] >= 1


def test_score_latest_and_alarm_consistency(client):
    latest = client.get("/v1/score").json()
    assert latest["date"] and len(latest["contributions"]) == 5
    eps = client.get("/v1/anomalies", params={"start": "2019-01-01", "end": "2026-12-31"}).json()
    assert eps["episodes"], "the model raised no alarm episode in the test period"
    peak = eps["episodes"][0]["peak_date"]
    day = client.get("/v1/score", params={"date": peak}).json()
    assert day["alarm"] is True and day["score"] > day["threshold"]


def test_score_error_paths(client):
    r = client.get("/v1/score", params={"date": "2020-02-23"})  # a Sunday
    assert r.status_code == 422 and r.json()["error"]["code"] == "http_422"
    assert "Traceback" not in r.text
    assert client.get("/v1/score", params={"date": "1980-01-01"}).status_code == 422
    assert client.get("/v1/score", params={"date": "2999-01-01"}).status_code == 422
    assert client.get("/v1/score", params={"date": "not-a-date"}).status_code == 422
    assert client.get("/nope").status_code == 404


def test_anomalies_range(client):
    r = client.get("/v1/anomalies", params={"start": "2020-02-01", "end": "2020-04-30"})
    assert r.status_code == 200 and "episodes" in r.json()
    bad = client.get("/v1/anomalies", params={"start": "2020-05-01", "end": "2020-04-01"})
    assert bad.status_code == 422
    long = client.get("/v1/anomalies", params={"start": "1900-01-01", "end": "2999-01-01"})
    assert long.status_code == 422
    assert client.get("/v1/anomalies").status_code == 422


def test_post_score_validation(client):
    feats = client.get("/v1/model").json()["features"]
    ok = client.post("/v1/score", json={"features": {f: 0.0 for f in feats}})
    assert ok.status_code == 200 and ok.json()["alarm"] in (True, False)
    missing = {f: 0.0 for f in feats[1:]}
    assert client.post("/v1/score", json={"features": missing}).status_code == 422
    extra = {**{f: 0.0 for f in feats}, "bogus": 1.0}
    assert client.post("/v1/score", json={"features": extra}).status_code == 422
    big = {f: 0.0 for f in feats}
    big[feats[0]] = 1e6
    assert client.post("/v1/score", json={"features": big}).status_code == 422
    wrong_type = {f: 0.0 for f in feats}
    wrong_type[feats[0]] = "abc"
    assert client.post("/v1/score", json={"features": wrong_type}).status_code == 422
    assert client.post("/v1/score", content=b"{bad json").status_code == 422


def test_metrics_endpoint(client):
    client.get("/v1/health")
    text = client.get("/metrics").text
    assert "anomaly_requests_total" in text and "anomaly_model_loaded 1.0" in text


def test_api_key_enforced(export_dir):
    with client_for(export_dir, api_key="s3cret") as c:
        assert c.get("/v1/health").status_code == 200
        assert c.get("/v1/score").status_code == 401
        assert c.get("/v1/score", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_refuses_inconsistent_manifest(export_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(export_dir, bad)
    mf = json.loads((bad / "manifest.json").read_text())
    first = next(iter(mf["files"]))
    mf["files"][first] = "0" * 64
    (bad / "manifest.json").write_text(json.dumps(mf))
    with client_for(bad) as c:
        assert c.get("/v1/ready").status_code == 503
        assert c.get("/v1/score").status_code == 503
    (bad / "manifest.json").unlink()
    with client_for(bad) as c:
        assert c.get("/v1/ready").status_code == 503
        assert c.get("/v1/health").status_code == 200


def test_staleness_flag_on_failed_quality(export_dir, tmp_path):
    d = tmp_path / "stale"
    shutil.copytree(export_dir, d)
    st = d / "history" / "status.json"
    s = json.loads(st.read_text())
    s["quality_ok"] = False
    st.write_text(json.dumps(s))
    with client_for(d) as c:
        r = c.get("/v1/ready").json()
        assert r["stale"] is True and r["quality_ok"] is False
        assert c.get("/v1/score").json()["stale"] is True
