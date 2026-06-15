"""Phase 5: behavioral / invariance tests + the validation gate (in-dist F1 floor)."""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from mlops_drift.config import get_config
from mlops_drift.promotion.validate import validate_champion
from mlops_drift.training.train import run_training


def _uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/mlflow.db"


def _champion(cfg, uri):
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return mlflow.sklearn.load_model(f"models:/{cfg.mlflow.registered_model}@champion")


def test_validate_champion_passes_floor(tmp_path):
    cfg = get_config()
    uri = _uri(tmp_path)
    run_training(cfg, tracking_uri=uri)
    assert validate_champion(cfg, tracking_uri=uri) is True


def test_predictions_deterministic(tmp_path):
    cfg = get_config()
    uri = _uri(tmp_path)
    run_training(cfg, tracking_uri=uri)
    model = _champion(cfg, uri)
    cols = [f"f{i}" for i in range(cfg.data.n_features)]
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(20, cfg.data.n_features)), columns=cols)
    assert np.array_equal(model.predict(x), model.predict(x))  # same input → same output


def test_degenerate_input_no_crash(tmp_path):
    cfg = get_config()
    uri = _uri(tmp_path)
    run_training(cfg, tracking_uri=uri)
    model = _champion(cfg, uri)
    cols = [f"f{i}" for i in range(cfg.data.n_features)]
    zeros = pd.DataFrame(np.zeros((5, cfg.data.n_features)), columns=cols)
    preds = model.predict(zeros)
    assert len(preds) == 5
    assert set(map(int, preds)) <= {0, 1}
