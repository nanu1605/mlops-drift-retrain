# mlops-drift-retrain

An end-to-end MLOps platform that trains a network-intrusion classifier, **detects data drift in
production, auto-retrains, validates the challenger against the champion, and promotes only if it
wins — then hot-reloads serving.** A closed loop with no human in the path. The deliverable is the
*system* and its honest, reproducible evaluation — not model accuracy.

![architecture](docs/images/architecture.png)

## The result

When the production distribution shifts (a new attack sub-population the champion has never seen),
realized F1 collapses — then the loop detects the drift, retrains, promotes a better model, and
recovers, automatically:

![drift recovery](docs/images/drift_recovery.png)

Stale champion on drift ≈ **0.27 F1** → after auto-promotion ≈ **0.78 F1**. Reproduce with
`make experiment`.

## Quickstart

```bash
uv sync --extra dev          # Python 3.12 env + deps
make train                   # train + register the champion (MLflow, sqlite-backed)
make up                      # serve the champion (uvicorn) on :8000
make smoke                   # POST a sample → 200 + valid prediction
make monitor                 # one-shot drift + realized-perf report
make experiment              # full loop end-to-end → docs/images/drift_recovery.png
make test                    # 52 tests
```

No Docker/cluster needed — everything runs as local processes (see *Environment* below).

## How it works

1. **Serve** — `@champion` answers `POST /predict`; every request is logged to SQLite.
2. **Monitor** — a rolling window of served features is compared to the persisted reference
   window with Evidently → a **label-free** drift signal (`monitoring/drift.py`).
3. **Detect** — the controller polls the signal; a breach (debounced by a cooldown) triggers a
   retrain (`controller/loop.py`).
4. **Retrain** — in-process training on `reference + drift` so the challenger learns the new
   sub-population; registered as `@challenger` (`training/train.py`).
5. **Validate & promote** — challenger vs champion on a common holdout; promote **iff**
   `Δf1 ≥ f1_margin` **and** `f1 ≥ f1_floor`; the `@champion` alias moves atomically
   (`promotion/champion_challenger.py`). Never on tie/regression.
6. **Reload** — the controller `POST /reload`s serving, which hot-swaps the new champion. Loop.

See [`docs/architecture.md`](docs/architecture.md) and the experiment writeup
[`experiments/drift_experiment.md`](experiments/drift_experiment.md).

## Key design decisions

- **Registry aliases, not stages** — `@champion`/`@challenger` (MLflow ≥ 2.9 deprecated stages).
- **Time-aware splits, no leakage** — train on earlier rows, evaluate on later; asserted in tests.
  `t`/`period`/`label` are never features.
- **Label-free drift trigger, delayed-label realized perf** — the loop triggers on a signal that
  needs no labels (they arrive late); realized F1 is computed separately for honest evaluation.
- **Imbalanced metrics** — F1 + PR-AUC primary, never accuracy alone.
- **Config-driven** — every path/threshold/hyperparameter lives in `configs/`; no magic numbers.
- **Reproducible** — fixed seed; every run logs the DVC data hash + git SHA + params + the
  reference window; `dvc repro` reproduces data→train.

## Tech stack
Python 3.12 · `uv` · scikit-learn · MLflow (tracking + registry, sqlite-backed) · Evidently ·
FastAPI + uvicorn · prometheus-client · pandera · pydantic-settings · structlog · DVC · ruff ·
pytest.

## Environment (pure-local adaptations)
This environment has no Docker/Kubernetes, so services run as **local processes** and a
**synthetic** drift dataset stands in for real CICIDS2017 (auto-used if CSVs are dropped in
`data/raw/`). The Kubernetes manifests, Dockerfile, Prometheus scrape config, Grafana dashboard
(`deploy/`), and GitHub Actions workflows (`.github/`) are written as **deliverables but not
executed here** — in a cluster the retrain would be a K8s Job and serving a Deployment. Every
adaptation is recorded in [`CHANGELOG.md`](CHANGELOG.md).

## Limitations & future work
- Synthetic data with an engineered (detectable) shift; the promotion holdout overlaps retrain
  data (optimistic) — the honest view is the per-batch realized-F1 series.
- Realized F1 lags by the label delay in production (revealed immediately only for the plot).
- Single SQLite writer + in-process model swap assume one serving worker.
- **Future work:** real CICIDS2017 ingest; a disjoint future holdout for promotion; a feature
  store; multi-worker serving with a shared model cache; deploy the K8s/Grafana stack;
  concept-drift (not just covariate-drift) detection. *(Intentionally out of scope per the
  build's scope guardrails — build the loop, then stop.)*

## Layout
```
src/mlops_drift/{data,training,serving,monitoring,promotion,controller,experiments,utils}
configs/   config.yaml + thresholds.yaml      # all knobs
deploy/    docker/ k8s/ prometheus/ grafana/  # deliverables (not deployed locally)
pipelines/ replay.py        experiments/ drift_experiment.md       docs/ architecture.md
tests/     52 tests                           .github/workflows/    ci.yml + retrain.yml
```
