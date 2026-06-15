"""Phase 4: one-shot monitor integration — tmp MLflow store + synthetic served window."""

from __future__ import annotations

import numpy as np

from mlops_drift.config import load_config
from mlops_drift.data.ingest import ingest
from mlops_drift.monitoring.monitor import evaluate_once
from mlops_drift.serving.logging_store import RequestStore
from mlops_drift.training.train import run_training
from mlops_drift.utils.io import read_parquet


def test_evaluate_once_flags_drift_and_writes_outputs(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    cfg = load_config()
    cfg.monitoring.history = str(tmp_path / "history.jsonl")
    cfg.monitoring.prom_textfile = str(tmp_path / "monitor.prom")
    run_training(cfg, tracking_uri=uri)

    # fill the request store with drift-period rows (the "served traffic")
    feature_cols = [f"f{i}" for i in range(cfg.data.n_features)]
    path, _ = ingest(cfg)
    drift = read_parquet(path)
    drift = drift[drift["period"] == "drift"].head(300).reset_index(drop=True)
    store = RequestStore(tmp_path / "requests.db", feature_cols=feature_cols)
    store.log(
        drift[feature_cols],
        preds=np.zeros(len(drift), dtype=int),
        probas=np.zeros(len(drift)),
        version="1",
        ts=1.0,
    )

    result = evaluate_once(cfg, tracking_uri=uri, store=store)
    store.close()

    assert set(result) >= {
        "drift_detected",
        "drift_share",
        "realized_f1",
        "realized_pr_auc",
        "n_labels_arrived",
        "model_version",
    }
    assert result["drift_detected"] is True
    assert result["drift_share"] >= cfg.thresholds.drift.dataset_drift_share
    assert result["n_labels_arrived"] > 0
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "monitor.prom").exists()
