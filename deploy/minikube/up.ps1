# Start Minikube (Docker driver, 2 CPUs, ~3 GB), load the local image, install the chart,
# wait for readiness and run the smoke test through a port-forward.
# Requires: docker, minikube, kubectl, helm on PATH. Run from the repository root.
# (A portable minikube.exe in .tools\ is picked up automatically.)
param(
  [string]$Tag = "",
  [string]$Release = "anomaly",
  [int]$LocalPort = 18000,
  [int]$MemoryMb = 3072
)
# Native tools write progress to stderr; do not treat that as failure, check exit codes instead.
$ErrorActionPreference = "Continue"
if (Test-Path ".tools") { $env:Path = "$((Resolve-Path .tools).Path);$env:Path" }
function Step([string]$what, [scriptblock]$cmd) {
  Write-Host "== $what"
  & $cmd
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what (exit $LASTEXITCODE)"; exit 1 }
}
$t0 = Get-Date
if (-not $Tag) {
  $m = Get-Content models_export/manifest.json | ConvertFrom-Json
  $Tag = "v$($m.model_version)"
}
Step "minikube start" { minikube start --driver=docker --cpus=2 --memory=$MemoryMb }
Step "load image" { minikube image load "finanomaly-api:$Tag" }
Step "helm install" { helm upgrade --install $Release deploy/helm/anomaly-api --set image.tag=$Tag --wait --timeout 300s }
Step "rollout" { kubectl rollout status deployment/$Release-anomaly-api --timeout=300s }
$pf = Start-Process -PassThru -NoNewWindow kubectl -ArgumentList "port-forward", "svc/$Release-anomaly-api", "${LocalPort}:8000"
try {
  Start-Sleep 5
  python deploy/minikube/smoke_test.py "http://127.0.0.1:$LocalPort"
  $rc = $LASTEXITCODE
} finally { Stop-Process -Id $pf.Id -Force }
Write-Host ("elapsed seconds: {0:N0}" -f ((Get-Date) - $t0).TotalSeconds)
exit $rc
