#!/usr/bin/env bash
# See up.ps1. Run from the repository root.
set -euo pipefail
TAG="${1:-v$(python -c "import json;print(json.load(open('models_export/manifest.json'))['model_version'])")}"
REL="${2:-anomaly}"
minikube start --driver=docker --cpus=2 --memory=3072
minikube image load "finanomaly-api:${TAG}"
helm upgrade --install "$REL" deploy/helm/anomaly-api --set image.tag="$TAG" --wait --timeout 300s
kubectl rollout status "deployment/${REL}-anomaly-api" --timeout=300s
kubectl port-forward "svc/${REL}-anomaly-api" 18000:8000 &
PF=$!
trap 'kill $PF' EXIT
sleep 5
python deploy/minikube/smoke_test.py http://127.0.0.1:18000
