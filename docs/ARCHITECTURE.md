# Architecture

Everything below matches the code in `src/finanomaly`, `api/`, `docker/` and `deploy/`.

## Module map

```mermaid
flowchart TB
  subgraph data[finanomaly.data]
    http[http.py: cached, retried requests]
    bb[bundesbank.py]
    ecb[ecb.py]
    dax[dax.py: yfinance, Stooq fallback]
    build[build.py: daily table]
    quality[quality.py]
    eda[eda.py]
    refresh[refresh.py]
  end
  events[events.py: windows, labels, freeze]
  features[features.py: base features, trailing z-scores]
  splits[splits.py]
  ctx[context.py: Context]
  subgraph models[finanomaly.models]
    rules[rules.py]
    iforest[iforest.py]
    ae[autoencoder.py]
    trees[trees.py: LightGBM, XGBoost, Platt]
  end
  subgraph exp[finanomaly.experiments]
    runner[runner.py: val and final stage]
    tracking[tracking.py: MLflow runs]
    summary[summary.py]
    registry[registry.py: register, aliases, promotion]
    promotion[promotion.py: pure gate logic]
    decision[decision_doc.py]
  end
  subgraph ev[finanomaly.evaluation]
    episodes[episodes.py]
    metrics[metrics.py]
    testeval[test_eval.py]
    robust[robustness.py]
    review[review.py]
  end
  subgraph serve[finanomaly.serving + api]
    export[export.py]
    loader[model_loader.py]
    scoring[scoring.py]
    api[api/main.py: FastAPI]
  end
  bb --> build
  ecb --> build
  dax --> build
  http --> bb
  http --> ecb
  build --> ctx
  features --> ctx
  events --> ctx
  splits --> ctx
  ctx --> runner
  models --> runner
  runner --> tracking
  tracking --> registry
  promotion --> registry
  registry --> export
  export --> loader
  loader --> scoring
  scoring --> api
  ctx --> testeval
  registry --> testeval
  episodes --> metrics
  metrics --> runner
  metrics --> testeval
```

## Lifecycle: data to cluster

```mermaid
sequenceDiagram
  participant U as User / CLI
  participant D as Data (data/raw cache)
  participant M as MLflow (SQLite + artifacts)
  participant R as Registry
  participant X as Export dir
  participant I as Docker image
  participant K as Cluster (Helm, one pod)
  U->>D: fetch, build-dataset, quality
  U->>U: events-freeze (hash of events.yaml and windows)
  U->>M: experiment (7 configs per family, validation stage runs)
  U->>R: register (best per family, 5 seed study, refit on pre-2019 data, alias staging)
  U->>R: promote (gate: recall, FA budget, seed spread, explanations, serving contract)
  R-->>R: alias production on the champion
  U->>M: evaluate (test period, after the decision)
  U->>X: export-model (MLflow model, features, history, manifest with checksums)
  X->>I: docker build, tag = model version
  I->>K: helm install (minikube image load for a local cluster)
  U->>K: smoke_test.py through a port-forward
```

## Training stages and what each may see

| Stage | Fit data | Threshold calibration | Evaluated on | Used for |
|---|---|---|---|---|
| Validation stage | before 2015 (supervised: events with onset before 2015) | non-event days of 2015 to 2018 | 2015 to 2018 | configuration selection, gate |
| Final stage | all data before 2019 | unsupervised: non-event days of 2015 to 2018 (in sample); supervised: validation stage probability threshold | test period (2019 onward), after the decision | registered model, test numbers |

## Serving

```mermaid
flowchart LR
  subgraph image[Docker image, tag = model version]
    mdl[models_export/mlflow_model]
    hist[models_export/history/daily.parquet + status.json]
    man[manifest.json: checksums, run id, hashes]
    app[uvicorn api.main:app, non-root]
  end
  man -->|verified at start-up| app
  mdl --> app
  hist -->|features rebuilt at start-up| app
  app -->|/v1/health /v1/ready /v1/model /v1/score /v1/anomalies /metrics| client[client / probes / Prometheus]
```

The API refuses to load when the manifest is missing or the checksums do not match, answers
`/v1/health` regardless, and reports `ready: false` (HTTP 503) until a model is loaded. The refresh job
(`finanomaly refresh`) publishes new data only when the quality checks pass; otherwise it records
`quality_ok: false` in `history/status.json` and the API marks scores as stale.

## Kubernetes objects (Helm chart `deploy/helm/anomaly-api`)

One Deployment (one replica, non-root, read-only root filesystem, probes on `/v1/health` and
`/v1/ready`), one ClusterIP Service, one ConfigMap, and an optional Secret for the API key. No
Ingress, autoscaler or persistent volume.
