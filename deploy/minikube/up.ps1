# Start Minikube (Docker driver, 2 CPUs, ~3 GB), load the local image, install the chart,
# wait for readiness and run the smoke test through a port-forward.
# Requires: docker, minikube, kubectl, helm. Run from the repository root.
param(
  [string]$Tag = "",
  [string]$Release = "anomaly",
  [int]$LocalPort = 18000
)
$ErrorActionPreference = "Stop"
$t0 = Get-Date
if (-not $Tag) {
  $m = Get-Content models_export/manifest.json | ConvertFrom-Json
  $Tag = "v$($m.model_version)"
}
minikube start --driver=docker --cpus=2 --memory=3072
minikube image load "finanomaly-api:$Tag"
helm upgrade --install $Release deploy/helm/anomaly-api --set image.tag=$Tag --wait --timeout 300s
kubectl rollout status deployment/$Release-anomaly-api --timeout=300s
$pf = Start-Process -PassThru -NoNewWindow kubectl -ArgumentList "port-forward", "svc/$Release-anomaly-api", "${LocalPort}:8000"
try {
  Start-Sleep 5
  python deploy/minikube/smoke_test.py "http://127.0.0.1:$LocalPort"
  $rc = $LASTEXITCODE
} finally { Stop-Process -Id $pf.Id -Force }
Write-Host ("elapsed seconds: {0:N0}" -f ((Get-Date) - $t0).TotalSeconds)
exit $rc
