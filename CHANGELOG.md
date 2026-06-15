# Changelog

All notable changes per phase. Conventional Commits.

## Phase 0 — Scaffold, tooling, data versioning
- chore(scaffold): project skeleton, `pyproject.toml` (uv), `ruff.toml`, pytest, Makefile.
- feat(config): typed `pydantic-settings` loader for `configs/config.yaml` + `thresholds.yaml`.
- feat(utils): structured JSON logging, central seeding, IO helpers.
- feat(data): synthetic intrusion generator (reference+drift, new attack sub-population,
  delayed labels) and `ingest.py` (`make data`): CICIDS2017 if present, else synthetic.
- test: config-loads / missing-key, synthetic determinism + drift-shape, smoke.

### Environment path taken (per spec §0.3 / §12)
- **Run mode: pure-local.** No Docker/kind/make in this environment → K8s manifests and
  Dockerfiles are written as deliverables but not deployed; services run as local processes.
- **Python: 3.12** via `uv` (system Python is 3.14, too new for the ML wheel set).
- **Dataset: synthetic** generator (real CICIDS2017 not downloaded here); ingest auto-detects
  real CSVs in `data/raw/` if ever provided.
