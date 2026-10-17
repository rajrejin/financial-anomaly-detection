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

## Compose trainer profile

* `docker compose --profile train up trainer` builds `docker/Dockerfile.trainer`, waits for the
  MLflow service to be healthy and runs `finanomaly experiment --family all` against the server over
  HTTP. Two things had to be fixed first: the MLflow server rejected the compose host name (403
  invalid Host header) until `--allowed-hosts` was added, and the experiment must not set a local
  artifact location when a server is used.
* The container logged all 7 Isolation Forest runs to the server (a rerun skipped all 7 as
  finished, confirming resumability over HTTP). The full set was started but the Docker VM (3.5 GB,
  shared with another project's container) stopped responding while MLflow (1.6 GiB) and the trainer
  (1.4 GiB limit) ran together; Docker Desktop was restarted. The full 28-run set through compose was
  therefore not completed; it was run locally instead (`results/runtime.json`).
* The trainer mounted `results/` and overwrote `runs_summary.csv` with its 7 rows once; this was
  restored from git and the mount now points to `results/trainer/`.

## Kubernetes (Minikube)

* Minikube v1.39.0 could not be installed with winget without administrator rights, so the official
  portable `minikube-windows-amd64.exe` release was downloaded to the git-ignored `.tools/` folder.
  Helm 4.3.1 and kubeconform 0.8.0 are user-scope winget installs.
* `deploy/minikube/up.ps1` (Docker driver, 2 CPUs, 3072 MB): cluster start, image load, `helm
  upgrade --install --wait`, rollout and port-forward smoke test completed in 219 seconds from a
  cold cluster, including a 508 MB base image download. Smoke test: 7 of 7 checks passed against the
  pod. Pod 1/1 Running, ClusterIP service on 8000, uid 10001, requests 250m and 256Mi, limits 1 CPU
  and 768Mi.
* `deploy/minikube/down.ps1`: release uninstalled and the cluster stopped in 21 seconds.
* `helm lint`: 1 chart linted, 0 failed. `helm template ... | kubeconform -strict -kubernetes-version
  1.30.0`: 3 resources valid; with `--set apiKey.enabled=true`, 4 valid.
