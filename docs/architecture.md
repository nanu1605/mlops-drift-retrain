# Architecture

![architecture](images/architecture.png)

A closed drift-triggered retraining loop. Each component does one job; the controller wires
them into a loop with no human in the path.

## Components

| Module | Responsibility |
|---|---|
| `data/` | `ingest` (synthetic generator, or real CICIDS2017 CSVs if present) → `validation` (pandera) → `features` (median-impute on raw `f*`) → `split` (time-aware, no leakage). |
| `training/` | `train.run_training` fits `Pipeline(impute→RF)`, evaluates (F1 + PR-AUC primary), logs params/metrics/seed/git-SHA/DVC-hash + the reference window to MLflow; `register` sets `@challenger` (+`@champion` on first run) **aliases**. |
| `serving/` | FastAPI `app` — `POST /predict`, `GET /health`, `GET /metrics` (Prometheus), `POST /reload`. `model_loader.ChampionLoader` resolves `models:/…@champion` as a sklearn Pipeline (predict + predict_proba), hot-swappable. `logging_store.RequestStore` appends every prediction to SQLite. |
| `monitoring/` | `drift` (Evidently `DataDriftPreset`, **label-free**) over the served window vs the reference; `performance` (realized F1/PR-AUC under delayed labels); `monitor.evaluate_once` = one-shot drift + realized perf (→ history JSONL + Prometheus textfile). |
| `promotion/` | `champion_challenger` scores `@champion` vs `@challenger` on the drift holdout and promotes iff `Δf1 ≥ f1_margin` **and** `f1 ≥ f1_floor` (alias-only move); `validate` is the CI F1-floor gate. |
| `controller/` | `loop.tick` = evaluate_once → on drift (debounced by `cooldown_seconds`) retrain on `reference+drift` → promotion gate → `POST /reload`. Logs every decision. |

## The loop
1. **Serve** — `@champion` answers `/predict`; every request is logged to SQLite.
2. **Monitor** — a rolling window of served features is compared to the reference window;
   Evidently yields a label-free drift signal.
3. **Detect** — the controller polls the signal; a breach (debounced) triggers a retrain.
4. **Retrain** — in-process training on `reference + drift` (so the challenger learns the new
   sub-population) registers a new `@challenger`.
5. **Validate & promote** — challenger vs champion on the drift holdout; promote only if it
   wins by the margin and clears the floor; `@champion` alias moves atomically.
6. **Reload** — the controller `POST /reload`s serving, which hot-swaps the new champion. Loop.

## Environment note (pure-local)
This repo runs as local processes (uvicorn serving, in-process controller, hermetic sqlite
MLflow store). The Kubernetes manifests, Dockerfile, Prometheus scrape config, Grafana
dashboard, and GitHub Actions workflows under `deploy/` and `.github/` are written as
deliverables but **not executed here** — in a cluster the retrain would be a K8s Job and the
controller a Deployment. See `CHANGELOG.md` for the adaptations taken.
