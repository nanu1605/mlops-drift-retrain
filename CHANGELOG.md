# Changelog

All notable changes per phase. Conventional Commits.

## Phase 0 — Scaffold, tooling, data versioning
- chore(scaffold): project skeleton, `pyproject.toml` (uv), `ruff.toml`, pytest, Makefile.
- feat(config): typed `pydantic-settings` loader for `configs/config.yaml` + `thresholds.yaml`.
- feat(utils): structured JSON logging, central seeding, IO helpers.
- feat(data): synthetic intrusion generator (reference+drift, new attack sub-population,
  delayed labels) and `ingest.py` (`make data`): CICIDS2017 if present, else synthetic.
- test: config-loads / missing-key, synthetic determinism + drift-shape, smoke.

## Phase 3 — Serving + operational monitoring (local)
- feat(serving): `model_loader.py` `ChampionLoader` — resolves `models:/<name>@champion`, loads
  it as a sklearn Pipeline (`predict` + `predict_proba`), thread-safe hot-swap. Two-pronged
  reload: `POST /reload` (Phase 5 hook) + a daemon thread re-resolving every
  `serving.model_refresh_seconds`; swaps only when the alias version changes.
- feat(serving): `app.py` FastAPI via `create_app(cfg, tracking_uri, refresh)` factory —
  `POST /predict` (raw `f*` batch → pandera-validate `require_label=False` → predict → log),
  `GET /health` (503 if no champion), `GET /metrics` (Prometheus), `POST /reload`. Prometheus
  counters: `predictions_total{predicted_class}`, `predict_latency_seconds`,
  `requests_total{endpoint,code}`, `prediction_errors_total`, `model_version` gauge, `model` Info.
- feat(serving): `logging_store.py` `RequestStore` — SQLite (WAL) append of each prediction
  (`ts, model_version, f0..f11, pred, proba`); `read_window(seconds=…)` for Phase 4. Explicit
  f-cols so the monitor reads a tidy frame.
- feat(serving): `schemas.py` (pydantic request/response), `smoke.py` (`make smoke`: valid 200 +
  schema, bad input 4xx, `/metrics` check).
- config: `paths.serving` + `serving.request_db`; `Config.serving_dir` / `request_db_path`.
- make: `up` → uvicorn serving (no MLflow server — reads sqlite store directly); `smoke` →
  smoke client. `.gitignore` += `/data/serving/`.
- deliverables (written, NOT deployed — local mode, spec §0.3): `deploy/docker/serving.Dockerfile`,
  `deploy/k8s/serving-{deployment,service}.yaml`, `deploy/prometheus/scrape.yml`,
  `deploy/grafana/serving_dashboard.json`.
- test: `test_serving.py` (health/predict/reload/metrics/4xx via TestClient on a tmp store,
  `refresh=False`), `test_logging_store.py` (append + windowed read). 36 tests total.

Bad input → 4xx (missing feature / non-finite → 400; empty batch → 422), never 500. Champion
loaded as a self-contained sklearn Pipeline on raw `f*` cols. `model_version` stays the champion
alias's version (promotion is Phase 5). Verified live: `make up` + `make smoke` → valid 200,
`/metrics` counters present, `/reload` 200, request DB written under `data/serving/`.

## Phase 2 — Training pipeline + MLflow tracking & registry
- feat(training): `train.py` (`make train`) — ingest→validate→time-aware split→fit sklearn
  Pipeline(median-impute→RandomForest)→evaluate; logs params/metrics/seed/git-SHA/DVC-md5 +
  artifacts (model w/ signature, feature_schema.json, **reference window** reference.parquet,
  confusion_matrix + pr_curve plots).
- feat(training): `register.py` — registers version, sets `@challenger` alias; first run also
  `@champion`. Aliases (not stages). `mlflow_utils.py` — sqlite backend setup + `data_md5`.
- feat(pipeline): `dvc.yaml` (ingest→train) + `params.yaml`; `metrics/train_metrics.json`
  DVC-tracked. `make mlflow` (UI), `make repro`.
- config: `mlflow` block → sqlite backend (`sqlite:///mlflow.db`), artifact_location,
  registered_model, ui host/port; `Config.resolved_tracking_uri()`.
- test: `test_training_pipeline.py` (run/params/metrics/tags/artifacts, alias resolution,
  champion load+predict, determinism). 28 tests total.

MLflow backend = **direct sqlite file** (hermetic, no server needed for train/register/tests);
`make mlflow` serves the UI on the same db. Model = single sklearn Pipeline on raw `f*` cols.
`dvc repro` reproduces data→train; second run up-to-date (deterministic).

## Phase 1 — Baseline model (exploration + honest metrics)
- feat(data): `split.py` time-aware split (asserts no timestamp overlap), `features.py`
  deterministic median-impute pipeline (excludes `t`/`period`/`label`; joblib save/load),
  `validation.py` pandera schema (finite floats + binary label); wired validation into ingest.
- feat(training): `evaluate.py` (precision/recall/F1/PR-AUC/confusion — F1+PR-AUC primary),
  `baseline.py` runner (`make baseline`) → `artifacts/baseline_metrics.json`.
- test: `test_features.py`, `test_data_validation.py`, `test_training.py` (24 passing total).
- docs: `notebooks/01_eda.ipynb` (EDA only).

Baseline (synthetic, seed 42): RF in-distribution **F1 0.576 / PR-AUC 0.738**, beats majority
trivial (F1 0, PR-AUC 0.176) on both. Held-out drift period RF F1 drops to **0.278** (the
degradation the experiment recovers). Honest note: the linear LogisticRegression reference
(F1 0.688 / PR-AUC 0.800) beats RF — synthetic data has a linear decision direction; RF stays
the registered family per spec, logistic is a reference only.

### Environment path taken (per spec §0.3 / §12)
- **Run mode: pure-local.** No Docker/kind/make in this environment → K8s manifests and
  Dockerfiles are written as deliverables but not deployed; services run as local processes.
- **Python: 3.12** via `uv` (system Python is 3.14, too new for the ML wheel set).
- **Dataset: synthetic** generator (real CICIDS2017 not downloaded here); ingest auto-detects
  real CSVs in `data/raw/` if ever provided.
