#!/usr/bin/env bash
set -euo pipefail
finanomaly refresh
finanomaly export-model
finanomaly serve-api --port 8000 &
PID=$!
trap 'kill $PID' EXIT
for i in $(seq 1 40); do curl -sf http://127.0.0.1:8000/v1/health >/dev/null && break; sleep 2; done
python deploy/minikube/smoke_test.py http://127.0.0.1:8000
