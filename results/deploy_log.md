# Deployment log

Recorded on the development machine (Windows 11, Docker Desktop with a 3.3 GiB VM). Values are
observed command output from this session.

## Docker API image

* Command: `docker build -f docker/Dockerfile.api --build-arg FAMILY=isolation_forest -t finanomaly-api:v1 .`
* Build time: about 58 seconds with a warm layer cache (first build 81 seconds, with 1.89 GB image
  before the family-specific dependency split).
* Image size: 1.06 GB. Non-root user (uid 10001), read-only root filesystem with a tmpfs for `/tmp`.
* Run: `docker run --read-only --tmpfs /tmp --memory 512m -p 8001:8000 finanomaly-api:v1`.
  Container health status: healthy. Memory in use: about 400 MiB of the 512 MiB limit.
* `deploy/minikube/smoke_test.py` against the container: 7 of 7 checks passed (health, ready, model
  info, score for a past date, alarm on the COVID-19 onset day, weekend rejected, metrics).

## Docker Compose

* `docker compose -f docker/docker-compose.yml up -d --build mlflow api`: both services reached
  the healthy state. The MLflow server (version 3.17.1) was killed for exceeding memory at limits of
  768 MiB and 1.5 GiB (exit code 137, OOMKilled) and uses about 1.6 GiB at start-up, so the limit
  was set to 2 GiB. The API service passed the same 7 smoke checks through the compose port mapping.
* The `trainer` profile (`--profile train`) is defined but was not run: it needs `data/raw`
  mounted and runs the same experiments already performed locally.

## Kubernetes

* **Chart validated, not deployed.** Minikube is not installed: `winget install Kubernetes.minikube`
  needs administrator rights (the UAC prompt was cancelled; no user-scope installer exists), and
  the specification asks to record the gap rather than use another route.
* Helm 4.3.1 and kubeconform 0.8.0 were installed at user scope with winget.
* `helm lint deploy/helm/anomaly-api`: 1 chart linted, 0 failed.
* `helm template rel deploy/helm/anomaly-api | kubeconform -strict -kubernetes-version 1.30.0`:
  3 resources valid, 0 invalid. With `--set apiKey.enabled=true` 4 resources validate (0 invalid).
* The scripts `deploy/minikube/up.ps1`, `up.sh`, `down.ps1` and `down.sh` were written but have not
  been executed. To run them: install minikube, run `scripts/build_image.ps1`, then
  `deploy/minikube/up.ps1` from the repository root.
