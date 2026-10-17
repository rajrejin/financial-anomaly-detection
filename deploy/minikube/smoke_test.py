"""Smoke test against a running API (local, compose or port-forwarded Minikube service).

Usage: python deploy/minikube/smoke_test.py http://127.0.0.1:8000
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import sys

import httpx


def main(base: str) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
        if not ok:
            failures.append(name)

    c = httpx.Client(base_url=base, timeout=30)
    r = c.get("/v1/health")
    check("health", r.status_code == 200 and r.json() == {"status": "ok"})
    r = c.get("/v1/ready")
    check("ready", r.status_code == 200 and r.json().get("model_loaded") is True, r.text[:120])
    r = c.get("/v1/model")
    m = r.json() if r.status_code == 200 else {}
    check("model info", r.status_code == 200 and m.get("alias") == "production", str(m.get("name")))
    r = c.get("/v1/score", params={"date": "2024-06-14"})
    check("score past date", r.status_code == 200 and "score" in r.json(), r.text[:100])
    r = c.get("/v1/score", params={"date": "2020-02-24"})
    check("known event day alarm (COVID-19 onset)", r.status_code == 200 and r.json()["alarm"])
    r = c.get("/v1/score", params={"date": "2020-02-23"})
    check("weekend rejected", r.status_code == 422)
    r = c.get("/metrics")
    check("metrics", r.status_code == 200 and "anomaly_requests_total" in r.text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
