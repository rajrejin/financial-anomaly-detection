#!/usr/bin/env bash
# Build the API image; the tag equals the model version in models_export/manifest.json.
set -euo pipefail
finanomaly export-model
VERSION=$(python -c "import json;print(json.load(open('models_export/manifest.json'))['model_version'])")
FAMILY=$(python -c "import json;print(json.load(open('models_export/manifest.json'))['family'])")
docker build -f docker/Dockerfile.api --build-arg FAMILY="$FAMILY" -t "finanomaly-api:v${VERSION}" .
