# Makefile is the API. Every workflow has a target. Run mode here = pure-local
# (no Docker/kind in this environment); cluster targets degrade gracefully.

UV ?= uv
RUN := $(UV) run
PY := $(RUN) python

.DEFAULT_GOAL := help
.PHONY: help setup lint fmt test data baseline mlflow train repro up up-compose smoke \
        monitor loop replay experiment down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps (uv) + dvc init
	$(UV) sync --extra dev
	@test -d .dvc || $(RUN) dvc init -q --no-scm 2>/dev/null || $(RUN) dvc init -q || true
	@echo "setup complete"

lint: ## ruff check + format --check
	$(RUN) ruff check src tests pipelines
	$(RUN) ruff format --check src tests pipelines

fmt: ## ruff format (write)
	$(RUN) ruff format src tests pipelines
	$(RUN) ruff check --fix src tests pipelines

test: ## pytest with coverage
	$(RUN) pytest

data: ## Fetch/generate + DVC-track the dataset
	$(PY) -m mlops_drift.data.ingest
	@$(RUN) dvc add data/raw/dataset.parquet 2>/dev/null && echo "dvc-tracked" || \
	  echo "dvc add skipped (dvc not initialized?)"

baseline: ## Train+evaluate Phase-1 baseline -> artifacts/baseline_metrics.json
	$(PY) -m mlops_drift.training.baseline

mlflow: ## Start local MLflow UI server (sqlite backend) at ui_host:ui_port
	$(RUN) mlflow server --backend-store-uri sqlite:///mlflow.db \
	  --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

train: ## Run training pipeline -> MLflow run + registered @challenger (+@champion first)
	$(PY) -m mlops_drift.training.train

repro: ## dvc repro the data->train pipeline (reproducible; metrics in metrics/)
	$(RUN) dvc repro

up: ## Local: MLflow + uvicorn serving (Phase 3)
	@echo "Phase 3 target — implemented in Phase 3."

up-compose: ## docker-compose fallback
	@echo "No Docker in this environment — using pure-local mode. Use 'make up'."

smoke: ## Post a sample request to /predict and assert valid response (Phase 3)
	@echo "Phase 3 target — implemented in Phase 3."

monitor: ## Start monitoring service (drift + realized performance) (Phase 4)
	@echo "Phase 4 target — implemented in Phase 4."

loop: ## Start controller: drift -> retrain -> validate -> promote -> reload (Phase 5)
	@echo "Phase 5 target — implemented in Phase 5."

replay: ## Stream later-period traffic into /predict (Phase 6)
	@echo "Phase 6 target — implemented in Phase 6."

experiment: ## Run full drift experiment + regenerate drift_recovery.png (Phase 6)
	@echo "Phase 6 target — implemented in Phase 6."

down: ## Tear everything down
	@pkill -f "mlflow server" 2>/dev/null || true
	@pkill -f "uvicorn" 2>/dev/null || true
	@echo "down"

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
