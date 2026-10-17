# Build the API image; the tag equals the model version in models_export/manifest.json.
$ErrorActionPreference = "Stop"
finanomaly export-model
$m = Get-Content models_export/manifest.json | ConvertFrom-Json
docker build -f docker/Dockerfile.api --build-arg FAMILY=$($m.family) -t "finanomaly-api:v$($m.model_version)" .
