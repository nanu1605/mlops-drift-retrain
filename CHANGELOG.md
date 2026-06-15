# Changelog

All notable changes per phase. Conventional Commits.

## Phase 0 — Scaffold, tooling, data versioning
- chore(scaffold): project skeleton, `pyproject.toml` (uv), `ruff.toml`, pytest, Makefile.
- feat(config): typed `pydantic-settings` loader for `configs/config.yaml` + `thresholds.yaml`.
- feat(utils): structured JSON logging, central seeding, IO helpers.
- feat(data): synthetic intrusion generator (reference+drift, new attack sub-population,
  delayed labels) and `ingest.py` (`make data`): CICIDS2017 if present, else synthetic.
- test: config-loads / missing-key, synthetic determinism + drift-shape, smoke.

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
