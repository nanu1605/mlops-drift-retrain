"""Phase 3: FastAPI serving — isolated to a tmp MLflow store + tmp request DB.

A champion is trained into a per-test tmp sqlite store; the app points at it with the
background refresh thread disabled (determinism). ``TestClient`` as a context manager runs
the lifespan (startup loads the champion, shutdown stops/closes cleanly).
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_drift.config import load_config
from mlops_drift.serving.app import create_app
from mlops_drift.training.train import run_training


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serving")
    uri = f"sqlite:///{tmp}/mlflow.db"
    cfg = load_config()
    cfg.serving.request_db = str(tmp / "requests.db")  # isolate store from the repo
    result = run_training(cfg, tracking_uri=uri)
    return cfg, uri, result


@pytest.fixture(scope="module")
def client(trained):
    cfg, uri, _ = trained
    app = create_app(cfg, tracking_uri=uri, refresh=False)
    with TestClient(app) as c:
        yield c, cfg


def _sample_instances(cfg, n=3):
    df = pd.read_parquet(cfg.reference_dir / "reference.parquet")
    cols = [f"f{i}" for i in range(cfg.data.n_features)]
    return df[cols].head(n).to_dict(orient="records"), cols


def test_health_reports_version_and_features(client):
    c, cfg = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert len(body["feature_cols"]) == cfg.data.n_features


def test_predict_valid_returns_schema_and_logs(client, trained):
    c, cfg = client
    instances, _ = _sample_instances(cfg, 3)
    r = c.post("/predict", json={"instances": instances})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 3
    assert all(p in (0, 1) for p in body["predictions"])
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])
    assert body["model_version"] == c.get("/health").json()["model_version"]


def test_predict_missing_feature_is_4xx(client):
    c, cfg = client
    instances, cols = _sample_instances(cfg, 1)
    instances[0].pop(cols[0])  # drop f0
    r = c.post("/predict", json={"instances": instances})
    assert 400 <= r.status_code < 500
    assert "detail" in r.json()


def test_predict_empty_instances_is_422(client):
    c, _ = client
    r = c.post("/predict", json={"instances": []})
    assert r.status_code == 422  # pydantic min_length


def test_metrics_exposes_counters(client, trained):
    c, cfg = client
    instances, _ = _sample_instances(cfg, 2)
    c.post("/predict", json={"instances": instances})
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "predictions_total" in r.text
    assert "predict_latency_seconds" in r.text


def test_reload_resolves_version(client, trained):
    c, _ = client
    _, _, result = trained
    r = c.post("/reload")
    assert r.status_code == 200
    body = r.json()
    assert body["reloaded"] is True
    assert body["model_version"] == str(result["version"])
