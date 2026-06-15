# Changelog

All notable changes per phase. Conventional Commits.

## Phase 0 — Scaffold, tooling, data versioning
- chore(scaffold): project skeleton, `pyproject.toml` (uv), `ruff.toml`, pytest, Makefile.
- feat(config): typed `pydantic-settings` loader for `configs/config.yaml` + `thresholds.yaml`.
- feat(utils): structured JSON logging, central seeding, IO helpers.
- feat(data): synthetic intrusion generator (reference+drift, new attack sub-population,
  delayed labels) and `ingest.py` (`make data`): CICIDS2017 if present, else synthetic.
- test: config-loads / missing-key, synthetic determinism + drift-shape, smoke.

## Phase 6 — Drift experiment + recovery plot + README + diagram — final
- feat(experiments): `experiments/replay.py` `stream(url, df, cols, batch_size, on_batch, client)`
  — POST the drift period to `/predict` in batches (injectable httpx client for ASGI tests);
  `pipelines/replay.py` thin spec-path wrapper (`make replay`).
- feat(experiments): `experiments/run.py` `run_experiment` — boots a real uvicorn serving
  subprocess on a fresh reference-only champion, replays drift batches, drives the **real
  controller** (retrain→promote→`/reload`) between batches, records per-batch realized F1, and
  plots `_plot_recovery` → `docs/images/drift_recovery.png`. `make experiment`.
- feat(experiments): `experiments/architecture.py` `render` → `docs/images/architecture.png`
  (matplotlib boxes+arrows of the loop). `make diagram`.
- docs: full `README.md` (what+why, architecture + money-shot plot, quickstart, how-it-works,
  design decisions, tech stack, limitations/future); `docs/architecture.md` (component table +
  loop); `experiments/drift_experiment.md` (hypothesis, setup, plot, thresholds, failure modes,
  honest limitations incl. label latency + holdout overlap).
- make: `replay`/`experiment`/`diagram` targets (were placeholders).
- test: `test_replay.py` (stream over ASGI TestClient + on_batch hook), `test_experiment_plot.py`
  (recovery + architecture PNGs render). 52 tests total.

Verified live (`make experiment`): stale champion on drift realized F1 **≈ 0.27** → drift detected
→ retrain → promote → `/reload` → recovered **≈ 0.78**; `drift_recovery.png` shows the dip +
recovery with both markers. Project complete — all six phases done.

## Phase 5 — Close the loop (controller + champion/challenger + CI) — centerpiece
- feat(promotion): `champion_challenger.py` — `evaluate_pair` scores `@champion` + `@challenger`
  on the drift-period holdout; `decide_promotion` gate = `chal_f1 - champ_f1 >= promotion.f1_margin`
  **and** `chal_f1 >= validation.f1_floor` (never tie/regression/below-floor; same-version guard);
  `run_promotion` moves the `@champion` alias (alias-only, old version retained/rollback-able).
- feat(promotion): `validate.py` — model-validation gate: in-distribution (reference-test) F1 of
  the live champion `>= f1_floor`; `make validate` / CI exit nonzero below floor.
- feat(controller): `controller/loop.py` — `tick` = evaluate_once (drift signal) → if drift +
  cooled down → `run_training(periods=("reference","drift"))` (in-process retrain) → `run_promotion`
  → on promote `POST /reload`; debounced by `cooldown_seconds`; appends a decision trail. `run_loop`
  drives `tick` every `poll_seconds`. `make loop`.
- feat(training): `run_training(..., periods=("reference",))` — additive; controller passes
  `("reference","drift")` so the challenger learns the drift sub-population (drops the trailing
  `label_delay_steps` as not-yet-arrived). Baseline `make train` unchanged.
- ci: `.github/workflows/{ci,retrain}.yml` — deliverables (no remote here): ci = ruff+pytest+
  validate gate; retrain = workflow_dispatch+cron → train+promotion. Not executed locally.
- config: `controller.decisions` log; `Config.{controller_log_path,serving_url}`. thresholds
  `validation.f1_floor` 0.70→**0.50** (synthetic RF in-dist F1 ~0.576; 0.70 unreachable).
- test: `test_promotion.py` (strong promoted / weak rejected / gate rules), `test_controller.py`
  (closed loop + debounce), `test_model_behavior.py` (validate gate + determinism + degenerate
  input). 49 tests total.

Verified live (no human in path): champion v1 drift F1 **0.278** → 300 drift requests → drift
detected (share 1.0) → retrain → challenger drift F1 **0.832** → promote (delta 0.554 ≥ 0.01,
≥ floor) → `/reload` → **live champion advances to the new version**; `make validate` PASS
(in-dist F1 0.946). Promotion holdout overlaps retrain data → optimistic; the honest
out-of-sample view is Phase 4's realized-perf series (noted as a limitation for the writeup).

## Phase 4 — Drift detection + realized performance
- feat(monitoring): `drift.py` `detect_drift(reference, current, cfg) -> DriftResult` — Evidently
  `DataDriftPreset` over shared `f*` cols; extracts `share_of_drifted_columns` + per-feature
  `drift_by_columns` by **searching result keys** (version-robust). Boolean signal =
  `share >= thresholds.drift.dataset_drift_share` (config-driven, not Evidently's default);
  per-feature test threshold from `feature_stattest_threshold`. Label-free.
- feat(monitoring): `performance.py` — realized perf under **delayed labels**, offline over the
  drift stream. `realized_at_cursor` (label arrives iff `cursor_t - t >= label_delay_steps`),
  `realized_series` (sliding F1/PR-AUC time series for Phase 6), `realized_latest`,
  `champion_predict_fn` (reuses serving `ChampionLoader`). Reuses `evaluate_classification`.
- feat(monitoring): `metrics_exporter.py` — **dedicated** `CollectorRegistry` (no collision with
  serving's default registry); gauges `drift_detected/drift_share/drift_features_drifted/
  realized_f1/realized_pr_auc/realized_labels_arrived/monitor_model_version`; `write_prom`
  (Prometheus textfile-collector format).
- feat(monitoring): `monitor.py` `evaluate_once(cfg, tracking_uri, store)` — drift over the latest
  served window (skips <30 rows as noise) + realized perf; appends history JSONL, writes prom
  textfile, returns dict. `make monitor` runs it once and prints JSON. Phase 5 controller polls
  this for the (label-free) drift signal.
- config: `monitoring.history` + `monitoring.prom_textfile`; `Config.history_path` /
  `prom_textfile_path`. `.gitignore` += monitor runtime outputs.
- test: `test_drift.py` (shifted→detected, in-dist→not), `test_performance.py` (delayed-label
  gating + series), `test_monitor.py` (integration, tmp store). 41 tests total.

Verified live: 300 drift-period requests → `drift_detected true`, `share 1.0`, all 12 features
drifted; realized F1 **0.279** (the champion's drift-period degradation — the dip Phase 6
recovers). Drift signal is label-free (triggers the controller); realized F1 (delayed labels) is
for honest eval + the recovery plot. Evidently pinned `<0.5` — `as_dict()` parsed by result keys.

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
