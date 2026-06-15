"""Phase 6: replay streams batches to /predict (ASGI, in-process — no real server)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_drift.config import load_config
from mlops_drift.experiments import replay
from mlops_drift.serving.app import create_app
from mlops_drift.training.train import run_training


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("replay")
    uri = f"sqlite:///{tmp}/mlflow.db"
    cfg = load_config()
    cfg.serving.request_db = str(tmp / "requests.db")
    run_training(cfg, tracking_uri=uri)
    app = create_app(cfg, tracking_uri=uri, refresh=False)
    # TestClient is a sync httpx client over the ASGI app — use it as the replay client.
    with TestClient(app) as client:
        yield cfg, client


def test_stream_posts_batches_and_fires_hook(served):
    cfg, client = served
    cols = [f"f{i}" for i in range(cfg.data.n_features)]
    df = pd.read_parquet(cfg.reference_dir / "reference.parquet")[cols].head(120).copy()

    seen = {"batches": 0, "rows": 0}

    def on_batch(batch, preds, probas):
        seen["batches"] += 1
        seen["rows"] += len(batch)
        assert len(preds) == len(batch)
        assert all(0.0 <= p <= 1.0 for p in probas)

    summaries = replay.stream(
        "http://test", df, cols, batch_size=50, on_batch=on_batch, client=client
    )
    assert seen["rows"] == 120
    assert seen["batches"] == 3  # 50 + 50 + 20
    assert sum(s["n"] for s in summaries) == 120
    assert all(s["model_version"] for s in summaries)
