"""Phase 3: SQLite request store round-trip + time-window read."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_drift.serving.logging_store import RequestStore

COLS = ["f0", "f1", "f2"]


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({c: np.arange(n, dtype=float) + i for i, c in enumerate(COLS)})


def test_log_then_read_roundtrip(tmp_path):
    store = RequestStore(tmp_path / "requests.db", feature_cols=COLS)
    df = _frame(5)
    n = store.log(
        df, preds=np.array([0, 1, 0, 1, 1]), probas=np.linspace(0, 1, 5), version="3", ts=100.0
    )
    assert n == 5
    assert store.count() == 5

    out = store.read_window()
    assert len(out) == 5
    assert set(COLS) <= set(out.columns)
    assert {"pred", "proba", "model_version", "ts"} <= set(out.columns)
    assert out["model_version"].unique().tolist() == ["3"]
    assert out["pred"].tolist() == [0, 1, 0, 1, 1]
    # WAL file created
    assert (tmp_path / "requests.db-wal").exists()
    store.close()


def test_read_window_filters_by_ts(tmp_path):
    store = RequestStore(tmp_path / "requests.db", feature_cols=COLS)
    store.log(_frame(2), np.array([0, 0]), np.array([0.1, 0.2]), version="1", ts=10.0)
    store.log(_frame(3), np.array([1, 1, 1]), np.array([0.8, 0.9, 0.7]), version="2", ts=100.0)

    recent = store.read_window(seconds=5.0, now=100.0)  # keep ts > 95 → only the 3 newer rows
    assert len(recent) == 3
    assert recent["model_version"].unique().tolist() == ["2"]
    store.close()
