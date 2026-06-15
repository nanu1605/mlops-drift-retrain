# MLOps Drift-Triggered Retraining Platform — Implementation Spec

> **Audience:** This document is an execution spec for an autonomous coding agent (Claude Code).
> Build the project described here **phase by phase**, in order. Do not skip ahead.
> At the end of every phase, run that phase's acceptance checks and the full test suite, then commit.

---

## 0. How to use this document (instructions to the executing agent)

1. **Work strictly phase by phase (Phase 0 → Phase 6).** Each phase is a self-contained, runnable vertical slice. Do not begin a phase until the previous phase's *Acceptance criteria* all pass.
2. **After each phase:** run `make test` and the phase's acceptance checks, then make a single git commit using Conventional Commits (e.g. `feat(serving): add FastAPI prediction endpoint`). Append a one-line entry to `CHANGELOG.md`.
3. **If a phase has an external dependency that is unavailable** (no internet to download the dataset, no Docker daemon, no Kubernetes cluster), **do not fail silently.** Use the documented fallback (synthetic dataset / docker-compose mode) and clearly note in `CHANGELOG.md` which path was taken.
4. **Keep the model simple on purpose.** The system is the deliverable, not model accuracy. Do not substitute a deep-learning model or add model families beyond what is specified.
5. **Respect the scope guardrails in §11.** When tempted to add a feature not in this spec, don't — note it under "Future work" in the README instead.
6. **Prefer clarity over cleverness.** This is a portfolio project for graduate-school applications. Reviewers read the README and the drift-experiment writeup first. Reproducibility, clean structure, honest evaluation, and a demonstrable closed loop matter more than anything else.

### Priority order when trade-offs arise
`correctness > reproducibility > clarity of README & writeup > demonstrable drift→retrain loop > breadth of features`

---

## 1. Project overview

Build an **end-to-end machine-learning lifecycle platform** for a supervised classifier that **detects distribution shift ("drift") in production, automatically retrains, validates the new model against the current one (champion/challenger), and promotes it only if it wins** — closing the loop with no human in the path.

The chosen domain is **network-intrusion detection** (normal vs. attack traffic), because the public dataset has a *naturally occurring* drift property: later time periods contain attack types absent from earlier ones. This lets the drift experiment use real shift rather than synthetic noise.

### What "done" looks like in one sentence
A reviewer can clone the repo, run a handful of `make` commands, watch a model get deployed, watch a replay of newer traffic degrade its accuracy, watch the drift detector fire, watch retraining + promotion happen automatically, and see a single plot showing accuracy dip and recover.

---

## 2. Definition of done (success criteria)

The project is complete when **all** of the following are true:

- [ ] A model is trained via a reproducible pipeline; the exact training data version (DVC hash), params, metrics, and artifacts are logged to MLflow.
- [ ] The current production model is served behind a containerized FastAPI `/predict` endpoint running on a local Kubernetes cluster (kind).
- [ ] Operational metrics (latency, throughput, request count, current model version) are exposed to Prometheus and visualized in Grafana.
- [ ] A monitoring service computes **data drift** (label-free) and **realized performance** (when delayed labels arrive) on rolling windows of live traffic and records both over time.
- [ ] When a drift threshold is crossed, retraining is triggered **automatically** (no manual step).
- [ ] A challenger model is validated against the champion on a held-out set; it is promoted to production **only if** it beats the champion by a configured margin; otherwise it is archived and an alert is recorded.
- [ ] The serving layer picks up the newly promoted model automatically.
- [ ] CI (GitHub Actions) runs lint, tests, and a **model-validation gate** that blocks promotion of a model below threshold.
- [ ] The drift experiment (§9) is executed and produces `docs/images/drift_recovery.png` (accuracy over time with "drift detected" and "model promoted" markers).
- [ ] `README.md` and `experiments/drift_experiment.md` are written to the standard in §10.
- [ ] `make up`, `make smoke`, and `make experiment` work from a clean clone (given a cluster or the compose fallback).

---

## 3. Architecture

Closed feedback loop:

```
            +------------------+        +------------------+
  data ---> | Training pipeline| -----> |  Model registry  |
 (DVC)      | trains + logs    |        | @champion /      |
   ^        | (MLflow)         |        | @challenger      |
   |        +------------------+        +---------+--------+
   |                ^                              |
   | drift→retrain  | trigger                      v
   |        +-------+----------+        +------------------+
   +------- |   Controller     | <----- |   Serving API    |
            | poll drift →     |        |  FastAPI on K8s  |
            | retrain → promote|        +---------+--------+
            +------------------+                  |
                     ^                            v
            +--------+---------+        prediction + (delayed) label logs
            |    Monitoring    | <----------------+
            | drift + realized |
            | performance      |
            +------------------+
```

### Components
| Component | Responsibility | Primary tech |
|---|---|---|
| Data + features | Versioned raw → processed features; schema validation | DVC, pandas, pandera |
| Training pipeline | Train, evaluate, log run, register model version | scikit-learn, MLflow |
| Model registry | Version models; hold `@champion` / `@challenger` aliases | MLflow Model Registry |
| Promotion | Compare challenger vs champion on holdout; promote if it wins | custom (MLflow client) |
| Serving API | Resolve `@champion`, expose `/predict`, log requests, expose metrics | FastAPI, prometheus-client |
| Monitoring | Compute data drift + realized performance on rolling windows | Evidently |
| Controller | Poll drift status; on breach trigger retrain → validate → promote → reload serving | custom service / K8s Job |
| Observability | Dashboards for ops + drift + accuracy-over-time | Prometheus, Grafana |
| Replayer | Stream later-period traffic into `/predict` at a controlled rate | custom script |

### Model resolution & hot reload
- The registry uses **MLflow Model Registry aliases** (`@champion`, `@challenger`) — *not* the deprecated stage API. (MLflow ≥ 2.9 deprecated `stages` in favor of `aliases`/`tags`.)
- Serving resolves `models:/<name>@champion` on startup and on a periodic refresh (configurable), and exposes a `POST /reload` endpoint the controller calls immediately after a promotion. On Kubernetes, the controller may also patch a deployment annotation to trigger a rolling restart (optional).

### Two run modes
- **Primary:** Kubernetes via `kind` (local). All services as Deployments/Services; retraining runs as a Kubernetes `Job`.
- **Fallback:** `docker-compose` (use when no cluster is available). Same images, retraining runs as a one-off container the controller starts. Document which mode was used.

---

## 4. Tech stack

Pin to these (use the latest compatible patch; resolve conflicts toward the lower-friction option). Python **3.11+**.

| Area | Choice | Notes |
|---|---|---|
| Dependency mgmt | `uv` (preferred) or `pip` + `pyproject.toml` | Lockfile committed |
| Data versioning | `dvc` | Local remote is fine (`.dvc/cache`) |
| Tracking + registry | `mlflow` (≥ 2.9) | Use a local backend store + artifact dir; aliases not stages |
| Modeling | `scikit-learn` (RandomForest or HistGradientBoosting) | `lightgbm` permitted but optional; keep one model family |
| Data validation | `pandera` | Schema + value-range checks on features |
| Drift detection | `evidently` | DataDriftPreset + classification performance |
| Serving | `fastapi` + `uvicorn` | Sync or async; keep it simple |
| Metrics | `prometheus-client` | `/metrics` endpoint on serving + monitoring |
| Config | `pydantic-settings` + YAML in `configs/` | No hardcoded paths or thresholds |
| Logging | `structlog` (or stdlib `logging` with JSON formatter) | Structured logs |
| Testing | `pytest`, `pytest-cov` | Plus behavioral model tests |
| Lint/format | `ruff` (lint + format) | Run in CI |
| Containers | Docker | One image per service (multi-stage) |
| Orchestration | Kubernetes via `kind`; manifests in `deploy/k8s` | `kube-prometheus-stack` Helm for Prom/Grafana |
| CI | GitHub Actions | `ci.yml` (lint+test+validate), `retrain.yml` (dispatch+schedule) |
| Task runner | `make` (Makefile) | Single entrypoint for every workflow |

---

## 5. Dataset

**Primary:** CICIDS2017 (Canadian Institute for Cybersecurity, UNB) — the pre-extracted **"MachineLearningCSV"** version (8 CSVs, one per capture session across the week). It is labeled (`BENIGN` vs. specific attack classes) and **attack types differ by day**, which is the source of natural drift.

- Collapse labels to a binary target: `0 = BENIGN`, `1 = attack` (any non-benign class).
- Treat capture day/session order as the **time axis**. Earlier sessions = training/reference period; later sessions (with attack types unseen earlier) = the "drift" period the replayer streams.

**Acquisition:** The dataset must be downloaded from UNB (manual; not redistributable). Provide a `make data` target that: (a) checks for the CSVs in `data/raw/`; (b) if absent and download is not possible, **falls back to the synthetic generator** below so the whole pipeline is still runnable end to end.

**Synthetic fallback (`src/mlops_drift/data/synthetic.py`):** Generate a tabular binary-classification stream where:
- a "reference" period draws features from distribution A,
- a "drift" period shifts a subset of feature means/correlations and introduces a new minority sub-population that the reference model misclassifies,
- ground-truth labels are available but exposed to the monitoring layer **with a configurable delay** to simulate real label latency.

The fallback must reproduce the same *shape* of result (accuracy dips on drift, recovers after retrain) so the drift experiment is demonstrable without the real dataset.

**Reference window:** Persist a fixed reference feature set to `data/reference/` at training time (a sample of the training distribution). Evidently compares live windows against this reference.

---

## 6. Repository structure

Create exactly this layout (add files as phases require; the tree is the target end state):

```
mlops-drift-retrain/
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock                         # or requirements.txt + requirements-dev.txt
├── Makefile
├── ruff.toml
├── .dvc/  .dvcignore  dvc.yaml  params.yaml
├── .github/workflows/
│   ├── ci.yml                      # lint + test + model-validation gate
│   └── retrain.yml                 # workflow_dispatch + scheduled retrain
├── configs/
│   ├── config.yaml                 # paths, model name, windows, serving
│   └── thresholds.yaml             # drift + promotion thresholds
├── data/                           # DVC-tracked; contents gitignored
│   ├── raw/  processed/  reference/
├── src/mlops_drift/
│   ├── __init__.py
│   ├── config.py                   # pydantic-settings loader
│   ├── data/
│   │   ├── ingest.py  features.py  validation.py  synthetic.py
│   ├── training/
│   │   ├── train.py  evaluate.py  register.py
│   ├── promotion/
│   │   └── champion_challenger.py
│   ├── serving/
│   │   ├── app.py  model_loader.py  schemas.py  logging_store.py
│   ├── monitoring/
│   │   ├── drift.py  performance.py  metrics_exporter.py
│   ├── controller/
│   │   └── loop.py
│   └── utils/
│       ├── logging.py  io.py  seeds.py
├── pipelines/
│   └── replay.py                   # streams later-period traffic to /predict
├── tests/
│   ├── test_data_validation.py  test_features.py  test_training.py
│   ├── test_serving.py  test_promotion.py  test_model_behavior.py
├── deploy/
│   ├── docker/ serving.Dockerfile training.Dockerfile monitoring.Dockerfile controller.Dockerfile
│   ├── k8s/ kind-cluster.yaml mlflow.yaml serving-*.yaml monitoring-*.yaml
│   │        controller-*.yaml retrain-job.yaml prometheus-grafana/
│   └── compose/ docker-compose.yaml
├── notebooks/
│   └── 01_eda.ipynb                # Phase 1 exploration only
├── experiments/
│   └── drift_experiment.md
└── docs/
    ├── architecture.md
    └── images/ drift_recovery.png
```

---

## 7. Global conventions (apply in every phase)

- **Typed Python.** Full type hints; functions do one thing; no notebooks in `src/`.
- **Determinism.** Centralize seeding in `utils/seeds.py`; set seeds for numpy/sklearn everywhere randomness occurs. Record the seed in every MLflow run.
- **No leakage.** Splits are **time-aware** (train on earlier, validate/test on later). Never random-shuffle across the time boundary. Enforce this in code and assert it in a test.
- **Config-driven.** All paths, window sizes, thresholds, model name, and hyperparameters live in `configs/` and load through `config.py`. No magic numbers in logic.
- **Reproducibility link.** Every training run logs the DVC data hash, the git commit SHA, params, metrics, and the serialized model + reference window as artifacts.
- **Validation everywhere.** `pandera` schema validates features at ingest and at serving time; reject/flag malformed inputs.
- **Structured logging.** JSON logs with a request/run id; no bare `print`.
- **Tests are part of the deliverable, not an afterthought.** Each phase adds tests; `make test` must stay green.
- **Makefile is the API.** Every workflow has a `make` target (see Appendix A). A reviewer should never need to remember a raw command.
- **Commits.** Conventional Commits, one per phase minimum. Keep `CHANGELOG.md` current.

---

## 8. Phases

Each phase: **Objective → Tasks → Deliverables → Acceptance criteria.** A phase is "done" only when every acceptance box is checkable.

### Phase 0 — Scaffold, tooling, data versioning

**Objective:** A clean, reproducible project skeleton with dependency, lint, test, config, and data-versioning plumbing in place.

**Tasks**
- Initialize repo, `pyproject.toml`, lockfile, `ruff.toml`, `pytest` config, `Makefile`, `.gitignore`.
- Implement `config.py` (pydantic-settings) reading `configs/config.yaml` + `configs/thresholds.yaml`.
- Implement `utils/{logging,seeds,io}.py`.
- `dvc init`; create `data/{raw,processed,reference}`; add a local DVC remote (cache dir).
- Implement `data/synthetic.py` generator (reference + drift periods, delayed labels) so later phases are runnable without the real dataset.
- Implement `data/ingest.py` (`make data`): use CICIDS2017 CSVs if present, else synthetic; write `data/raw/` and track with DVC.
- Add a trivial passing test and a config-loads test.

**Deliverables:** runnable `make setup`, `make lint`, `make test`, `make data`; DVC-tracked raw data; loaded config object.

**Acceptance criteria**
- [ ] `make setup && make lint && make test` succeed from a clean clone.
- [ ] `make data` produces tracked data in `data/raw/` (real or synthetic) and a `.dvc` pointer.
- [ ] `config.py` raises clearly if a required key is missing.

---

### Phase 1 — Baseline model (exploration + honest metrics)

**Objective:** A correctly evaluated baseline classifier you understand, with no leakage.

**Tasks**
- `notebooks/01_eda.ipynb`: inspect class balance, feature ranges, and the time axis / drift periods. (EDA only — no production code lives here.)
- `data/features.py`: deterministic feature pipeline (impute, encode, scale as needed) returning a fixed, documented feature schema.
- `data/validation.py`: `pandera` schema for the feature frame.
- `training/evaluate.py`: metrics for imbalanced binary classification — precision, recall, F1, PR-AUC, confusion matrix. (Accuracy alone is misleading here; report F1/PR-AUC primarily.)
- A baseline `train` entry that fits RandomForest/HistGradientBoosting on the **earlier** period and evaluates on a **later** period (time-aware split). Compare against a trivial baseline (majority class / simple logistic) and record the delta.

**Deliverables:** reproducible baseline metrics; documented feature schema; evaluation module.

**Acceptance criteria**
- [ ] Train/test split is time-aware; a test asserts no timestamp overlap/leakage.
- [ ] Baseline beats the trivial baseline on F1 and PR-AUC, with numbers recorded.
- [ ] Feature pipeline is deterministic (same input → identical output; asserted in a test).

---

### Phase 2 — Training pipeline + MLflow tracking & registry

**Objective:** Turn the baseline into a reproducible, tracked, registered pipeline.

**Tasks**
- Stand up a local MLflow tracking server + backend store + artifact store (`make mlflow`).
- `training/train.py`: orchestrate ingest → validate → features → fit → evaluate; log params, metrics, seed, git SHA, DVC data hash; log artifacts (model, feature schema, **reference window**, evaluation plots).
- `training/register.py`: register the model to the MLflow Model Registry; set alias `@challenger` on the new version. If no `@champion` exists yet (first run), also set `@champion`.
- Wire a DVC pipeline (`dvc.yaml`, `params.yaml`) so `dvc repro` reproduces data→features→train.

**Deliverables:** `make train` runs the full pipeline and produces a registered, aliased model version with a complete MLflow run.

**Acceptance criteria**
- [ ] `make train` creates an MLflow run with params, metrics, seed, git SHA, DVC hash, and artifacts (incl. reference window).
- [ ] A new registry version is created and aliased `@challenger`; first ever run also sets `@champion`.
- [ ] `dvc repro` reproduces the pipeline; re-running with the same data yields identical metrics (determinism).

---

### Phase 3 — Serving + operational monitoring

**Objective:** Serve the champion behind an API on Kubernetes with ops observability. (This phase leans on existing DevOps skills.)

**Tasks**
- `serving/model_loader.py`: resolve `models:/<name>@champion` from the registry; cache; periodic refresh; expose current version.
- `serving/schemas.py`: pydantic request/response; validate inputs against the feature schema.
- `serving/app.py`: `POST /predict`, `GET /health`, `GET /metrics`, `POST /reload`. Instrument with prometheus-client: request count, latency histogram, prediction class counts, `model_version` gauge.
- `serving/logging_store.py`: append every request's features + prediction + timestamp + model version to an append-only store (parquet files or a small SQLite/Postgres). This log feeds monitoring later.
- `deploy/docker/serving.Dockerfile` (multi-stage). `deploy/k8s/`: `kind-cluster.yaml`, MLflow manifest, serving Deployment + Service, and `kube-prometheus-stack` (Helm values) with a Grafana dashboard JSON for the serving metrics.
- `make up` brings up kind + MLflow + serving + Prom/Grafana; `make smoke` posts a sample request and asserts a 200 + valid schema.

**Deliverables:** a live `/predict` on kind; Grafana dashboard showing latency/throughput/version.

**Acceptance criteria**
- [ ] `make up` then `make smoke` returns a valid prediction; `/metrics` exposes the counters/histograms.
- [ ] Grafana shows latency, request rate, and current `model_version`.
- [ ] Bad input is rejected with a 4xx and a clear validation error.
- [ ] (Fallback) If no cluster, `make up-compose` brings the same services up via docker-compose; note which path was used.

---

### Phase 4 — Drift detection + realized performance

**Objective:** Detect that the deployed model is going stale — first without labels, then with delayed labels.

**Tasks**
- `monitoring/drift.py`: on a rolling window of recent serving logs, run Evidently `DataDriftPreset` against the persisted reference window; emit a drift score + per-feature flags + a boolean "drift detected" against the configured threshold. **Label-free** — this is the early-warning signal.
- `monitoring/performance.py`: as delayed ground-truth labels arrive (real: later-session labels; synthetic: delayed reveal), join them to logged predictions and compute realized F1/PR-AUC over the same windows.
- `monitoring/metrics_exporter.py`: expose drift score, "drift detected", and realized F1 to Prometheus; add a Grafana panel charting drift and realized F1 over time.
- Tests: a window with injected shift trips the drift flag; a window without shift does not.

**Deliverables:** a monitoring service surfacing drift + realized performance over time on the dashboard.

**Acceptance criteria**
- [ ] Feeding a known-shifted window raises "drift detected"; an in-distribution window does not (asserted in tests).
- [ ] Drift score and realized F1 are charted over time in Grafana.
- [ ] Performance computation correctly handles **delayed** labels (predictions logged now, scored when labels arrive).

---

### Phase 5 — Close the loop (controller, champion/challenger, CI gates)

**Objective:** Make drift trigger an automatic retrain → validate → promote → reload, gated by quality checks. **This is the centerpiece phase.**

**Tasks**
- `promotion/champion_challenger.py`: load `@challenger` and `@champion`; evaluate both on a **fresh, common holdout** from the most recent period; promote challenger (move `@champion` alias to it, archive/relabel the old one) **iff** `challenger_F1 - champion_F1 ≥ promotion_margin` (from `thresholds.yaml`); else keep champion and record a rejected-challenger event/alert. Never promote on a tie or regression.
- `controller/loop.py`: poll the monitoring "drift detected" signal. On a breach (with cooldown/debounce to avoid thrashing): trigger retraining (Kubernetes `Job` from `deploy/k8s/retrain-job.yaml`, or a one-off container in compose mode), then run promotion logic, then call serving `POST /reload` (and/or patch the deployment to roll). Log every decision.
- Behavioral/invariance tests in `tests/test_model_behavior.py`: e.g., trivial perturbations don't flip predictions; obvious-attack rows are flagged. The CI gate uses these plus the F1 threshold.
- CI: `ci.yml` runs ruff + pytest + a **model-validation gate** (loads candidate, asserts F1 ≥ floor and behavioral tests pass; fail = block). `retrain.yml`: `workflow_dispatch` + a `schedule` cron that runs the same retrain→validate→promote path.
- `make loop` runs the controller against the live cluster.

**Deliverables:** a hands-off loop: drift → retrain → validated promotion → serving reload, with CI gates.

**Acceptance criteria**
- [ ] With drift present, the controller triggers retraining **with no manual step**, and the promotion decision is logged.
- [ ] A challenger that does **not** beat the champion by the margin is **rejected** (asserted with a deliberately weak challenger in tests).
- [ ] After a successful promotion, serving returns predictions from the new version (verify via `model_version`).
- [ ] CI blocks a model below the F1 floor or failing behavioral tests.
- [ ] Controller debounces (does not retrain repeatedly within the cooldown window).

---

### Phase 6 — Drift experiment, writeup, README, diagram

**Objective:** Run the experiment that proves the loop, and document the whole project to portfolio standard.

**Tasks**
- Execute the drift experiment per §9; save the timeline plot to `docs/images/drift_recovery.png`.
- Write `experiments/drift_experiment.md`: hypothesis, setup, what was measured, the plot, threshold-tuning discussion, failure modes, and an honest limitations section (esp. label latency vs. concept drift).
- Write `README.md` to the §10 standard (incl. the architecture diagram and the money-shot plot).
- `docs/architecture.md`: component responsibilities + the loop, with the same diagram.
- Final pass: `make experiment` reproduces the plot end to end; ensure `README` quickstart works from a clean clone.

**Acceptance criteria**
- [ ] `make experiment` reproduces `drift_recovery.png` (dip + recovery, with "drift detected" and "model promoted" markers).
- [ ] `experiments/drift_experiment.md` includes the plot, threshold discussion, and honest limitations.
- [ ] `README.md` meets the §10 checklist; quickstart works from scratch.

---

## 9. The drift experiment (the centerpiece)

This experiment is the most important artifact in the project. Run it deliberately and document it.

**Protocol**
1. **Baseline.** Train the champion on the earlier period. Deploy it. Record baseline realized F1 = **X** on an in-distribution window.
2. **Introduce drift.** Use `pipelines/replay.py` to stream the later period (real: sessions containing attack types unseen in training; synthetic: the shifted distribution) into `/predict` at a controlled rate.
3. **Detect (label-free first).** Confirm the **data-drift** detector fires *before* labels are available. Then, as delayed labels arrive, confirm realized F1 falls to **Y < X**.
4. **Auto-retrain.** The controller fires on the drift breach and retrains on a recent window that includes the new period.
5. **Validate + promote.** The challenger is compared to the champion on a fresh holdout; it wins by ≥ margin and is promoted; serving reloads.
6. **Recover.** Realized F1 climbs back toward **X** (call it **Z ≈ X**).

**The plot (`docs/images/drift_recovery.png`)**
- x-axis = time; y-axis = realized F1 (and optionally drift score on a secondary axis).
- Two vertical markers: **"drift detected"** and **"model promoted."**
- The curve must visibly show the dip (X→Y) and the recovery (Y→Z).

**Required written analysis (in `experiments/drift_experiment.md`)**
- The drift threshold and promotion margin chosen, and **why** — discuss the trade-off (too sensitive → retrain thrashing; too loose → slow reaction). Show at least one alternative setting and its effect.
- Whether a challenger was ever correctly **rejected**, and what that tells the reviewer.
- Failure modes encountered and how they were handled.
- **Honest limitations:** in the real world labels are delayed or absent, so concept drift is hard to catch in time — which is *why* label-free data-drift detection is the early signal. State this explicitly.

---

## 10. README requirements

`README.md` is read first; treat it as a deliverable, not documentation debt. It must contain, in order:

1. **One-paragraph what + why** (problem, and that the model self-heals via a drift→retrain→promote loop).
2. **Architecture diagram** (embed the loop; reuse `docs/architecture.md` image).
3. **The money-shot plot** (`drift_recovery.png`) with a one-line caption.
4. **Quickstart** — exact `make` commands from clone to a working `/predict` and to reproducing the experiment.
5. **How it works** — concise tour of the loop and each component.
6. **Design decisions & trade-offs** — model-CI vs. code-CI gating; aliases vs. stages; threshold/margin choices; time-aware splits and why.
7. **Tech stack.**
8. **Limitations / future work** (label latency; single model family; local-only cluster; anything deferred per §11).

Tone: precise, honest, technical. No marketing language. Show the trade-offs you reasoned about — that is the signal reviewers look for.

---

## 11. Scope guardrails — do NOT build these

Building any of these is out of scope and will dilute the project. List them under "Future work" instead:

- A feature store, multi-cloud or cloud-provider-specific infra, or Terraform beyond what kind needs.
- More than one model family, hyperparameter sweeps/AutoML, or deep-learning models.
- A custom web UI/dashboard beyond Grafana.
- Real authn/authz/RBAC, secrets management, or production-grade security hardening on the serving API.
- Autoscaling/HPA tuning, multi-node clusters, or load-testing infrastructure.
- A message bus (Kafka/RabbitMQ) — rolling windows over the prediction log are sufficient.
- Anything that turns this into a general MLOps platform. Build *this loop*, end to end, then stop and document.

---

## 12. Risks & troubleshooting (read before starting)

- **No internet to UNB / dataset unavailable:** use the synthetic generator (§5). The loop and experiment must still be demonstrable. Note the path taken in `CHANGELOG.md`.
- **No Docker/Kubernetes in the environment:** use the docker-compose fallback (`make up-compose`). If neither is available, still complete Phases 0–2 and 4–5 logic with the FastAPI app run locally (`uvicorn`) and document the limitation.
- **MLflow stages deprecation:** use **aliases** (`@champion`/`@challenger`), not `transition_model_version_stage`. Don't reintroduce stages.
- **Retrain thrashing:** the controller must debounce (cooldown) and the drift threshold must be tuned; document the chosen values.
- **Label leakage:** the single most common way this kind of project looks amateur. Enforce and **test** time-aware splits; never let post-boundary data into training/reference.
- **Over-engineering:** when in doubt, choose the simpler component and note the richer option as future work.

---

## Appendix A — Makefile targets (single entrypoint for every workflow)

Implement at least these:

```
make setup        # install deps (uv/pip), pre-commit, dvc init
make lint         # ruff check + format --check
make test         # pytest with coverage
make data         # fetch/generate + DVC-track the dataset
make mlflow       # start local MLflow tracking server
make train        # run training pipeline -> registered @challenger (+@champion if first)
make up           # kind cluster + MLflow + serving + Prom/Grafana
make up-compose   # docker-compose fallback (no kind)
make smoke        # post a sample request to /predict and assert valid response
make monitor      # start the monitoring service (drift + realized performance)
make loop         # start the controller (drift -> retrain -> validate -> promote -> reload)
make replay       # stream later-period traffic into /predict
make experiment   # run the full drift experiment and regenerate drift_recovery.png
make down         # tear everything down
```

## Appendix B — Suggested build order within a working session

1. `make setup` → `make lint` → `make test` (Phase 0 green).
2. EDA + features + evaluation (Phase 1) → commit.
3. `make mlflow` → `make train` produces a registered, aliased model (Phase 2) → commit.
4. `make up` → `make smoke`; Grafana ops dashboard (Phase 3) → commit.
5. `make monitor`; drift + realized performance on the dashboard (Phase 4) → commit.
6. `make loop`; verify auto retrain + gated promotion + reload; wire CI (Phase 5) → commit.
7. `make experiment`; write `drift_experiment.md`, `README.md`, `architecture.md` (Phase 6) → commit.

---

*End of spec. Build Phase 0 first. Keep the model simple, keep the loop honest, and make the drift experiment reproducible.*
