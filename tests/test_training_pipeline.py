"""Phase 2: MLflow tracking + registry. Isolated to a tmp sqlite store per test."""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from mlops_drift.config import get_config
from mlops_drift.training.train import run_training


def _tmp_uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/mlflow.db"


def test_run_logs_params_metrics_tags_artifacts(tmp_path):
    cfg = get_config()
    r = run_training(cfg, tracking_uri=_tmp_uri(tmp_path))

    mlflow.set_tracking_uri(_tmp_uri(tmp_path))
    client = MlflowClient()
    run = client.get_run(r["run_id"])

    # params
    assert run.data.params["seed"] == str(cfg.seed)
    assert run.data.params["train_frac"] == str(cfg.split.train_frac)
    # metrics present + sane
    assert 0.0 < run.data.metrics["f1"] <= 1.0
    assert "pr_auc" in run.data.metrics
    # reproducibility tags
    assert run.data.tags["git_sha"]
    assert run.data.tags["dvc_data_md5"]
    assert run.data.tags["data_source"] in {"synthetic", "cicids2017"}
    # artifacts incl model + reference window
    arts = {a.path for a in client.list_artifacts(r["run_id"])}
    assert {"model", "feature_schema.json", "reference.parquet"} <= arts


def test_aliases_challenger_and_first_champion(tmp_path):
    cfg = get_config()
    r = run_training(cfg, tracking_uri=_tmp_uri(tmp_path))

    mlflow.set_registry_uri(_tmp_uri(tmp_path))
    client = MlflowClient()
    name = cfg.mlflow.registered_model
    champ = client.get_model_version_by_alias(name, "champion")
    chal = client.get_model_version_by_alias(name, "challenger")
    # first ever run: both aliases point at the same (v1)
    assert champ.version == chal.version == r["version"]


def test_champion_model_loads_and_predicts(tmp_path):
    cfg = get_config()
    run_training(cfg, tracking_uri=_tmp_uri(tmp_path))

    mlflow.set_registry_uri(_tmp_uri(tmp_path))
    mlflow.set_tracking_uri(_tmp_uri(tmp_path))
    model = mlflow.sklearn.load_model(f"models:/{cfg.mlflow.registered_model}@champion")
    df = pd.read_parquet(cfg.raw_dir / "dataset.parquet")
    fc = [f"f{i}" for i in range(cfg.data.n_features)]
    preds = model.predict(df[fc].head(5))
    assert len(preds) == 5
    assert set(map(int, preds)) <= {0, 1}


def test_determinism_same_metrics(tmp_path):
    cfg = get_config()
    r1 = run_training(cfg, tracking_uri=f"sqlite:///{tmp_path}/a.db")
    r2 = run_training(cfg, tracking_uri=f"sqlite:///{tmp_path}/b.db")
    assert r1["metrics"]["f1"] == pytest.approx(r2["metrics"]["f1"], abs=1e-9)
    assert r1["metrics"]["pr_auc"] == pytest.approx(r2["metrics"]["pr_auc"], abs=1e-9)
