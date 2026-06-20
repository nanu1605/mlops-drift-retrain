"""Phase 5: closed-loop controller — drift -> retrain -> promote -> reload, + debounce."""

from __future__ import annotations

import socket
import urllib.request

import numpy as np
from mlflow.tracking import MlflowClient

from mlops_drift.config import load_config
from mlops_drift.controller.loop import ControllerState, run_loop, tick
from mlops_drift.data.ingest import ingest
from mlops_drift.serving.logging_store import RequestStore
from mlops_drift.training.train import run_training
from mlops_drift.utils.io import read_parquet


def _drift_store(cfg, tmp_path, n=300):
    feature_cols = [f"f{i}" for i in range(cfg.data.n_features)]
    path, _ = ingest(cfg)
    drift = read_parquet(path)
    drift = drift[drift["period"] == "drift"].head(n).reset_index(drop=True)
    store = RequestStore(tmp_path / "requests.db", feature_cols=feature_cols)
    store.log(
        drift[feature_cols],
        np.zeros(len(drift), dtype=int),
        np.zeros(len(drift)),
        version="1",
        ts=1.0,
    )
    return store


def _cfg(tmp_path):
    cfg = load_config()
    cfg.controller.decisions = str(tmp_path / "controller_log.jsonl")
    cfg.monitoring.history = str(tmp_path / "history.jsonl")
    cfg.monitoring.prom_textfile = str(tmp_path / "monitor.prom")
    return cfg


def test_closed_loop_retrains_and_promotes(tmp_path):
    cfg = _cfg(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    run_training(cfg, tracking_uri=uri)  # champion v1 (reference-only)
    store = _drift_store(cfg, tmp_path)

    reloads = []
    state = ControllerState()
    event = tick(
        cfg,
        state,
        tracking_uri=uri,
        store=store,
        reload_fn=lambda: reloads.append(1) or True,
        now=10_000.0,
    )
    store.close()

    assert event["drift_detected"] is True
    assert event["action"] == "promoted"
    assert state.retrains == 1 and state.promotions == 1
    assert reloads == [1]  # serving reload triggered
    client = MlflowClient()
    champ = client.get_model_version_by_alias(cfg.mlflow.registered_model, "champion")
    assert str(champ.version) == str(event["challenger_version"])  # advanced past v1
    assert (tmp_path / "controller_log.jsonl").exists()


def test_debounce_within_cooldown(tmp_path):
    cfg = _cfg(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    run_training(cfg, tracking_uri=uri)
    store = _drift_store(cfg, tmp_path)

    reloads = []
    state = ControllerState()
    t0 = 10_000.0
    tick(
        cfg,
        state,
        tracking_uri=uri,
        store=store,
        reload_fn=lambda: reloads.append(1) or True,
        now=t0,
    )
    # second tick inside cooldown → no retrain
    event2 = tick(
        cfg,
        state,
        tracking_uri=uri,
        store=store,
        reload_fn=lambda: reloads.append(1) or True,
        now=t0 + cfg.controller.cooldown_seconds - 1,
    )
    store.close()

    assert event2["action"] == "cooldown_skip"
    assert state.retrains == 1  # only the first tick retrained
    assert len(reloads) == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_loop_serves_monitor_metrics(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.controller.metrics_port = _free_port()

    # max_iters=0 → no tick (no champion needed); we only assert the metrics bridge binds.
    run_loop(cfg, max_iters=0, sleep_fn=lambda _s: None, serve_metrics=True)

    url = f"http://127.0.0.1:{cfg.controller.metrics_port}/metrics"
    with urllib.request.urlopen(url, timeout=5) as resp:
        body = resp.read().decode("utf-8")

    # the monitor's dedicated gauges are exposed (not serving's counters)
    assert "drift_detected" in body
    assert "realized_f1" in body
