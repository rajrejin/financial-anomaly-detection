# End to end: refresh data, export the production model, serve, smoke test.
$ErrorActionPreference = "Stop"
finanomaly refresh
finanomaly export-model
$p = Start-Process -PassThru -NoNewWindow finanomaly -ArgumentList "serve-api", "--port", "8000"
try {
  for ($i = 0; $i -lt 40; $i++) {
    try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/v1/health | Out-Null; break } catch { Start-Sleep 2 }
  }
  python deploy/minikube/smoke_test.py http://127.0.0.1:8000
} finally { Stop-Process -Id $p.Id -Force }
