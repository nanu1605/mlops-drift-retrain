"""Phase 4: realized performance under delayed labels. No MLflow — stub predict_fn."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_drift.config import get_config
from mlops_drift.monitoring import performance as perf


def _frame(n=500, delay_period=True):
    rng = np.random.default_rng(0)
    cols = {f"f{i}": rng.normal(size=n) for i in range(12)}
    df = pd.DataFrame(cols)
    df["t"] = np.arange(1000, 1000 + n)  # drift-period-like increasing time
    df["label"] = rng.integers(0, 2, size=n)
    df["period"] = "drift"
    return df


def test_delayed_label_gating():
    cfg = get_config()
    df = _frame()
    delay = cfg.data.label_delay_steps  # 200
    preds = df["label"].to_numpy()  # perfect predictions
    probas = preds.astype(float)

    t_min = int(df["t"].min())
    # cursor just below first arrival: nothing has aged `delay` yet
    early = perf.realized_at_cursor(
        df,
        preds,
        probas,
        cursor_t=t_min + delay - 1,
        time_col="t",
        target="label",
        delay=delay,
    )
    assert early["n_arrived"] == 0

    # cursor where exactly rows with t <= cursor-delay have arrived
    cursor = t_min + delay + 50
    mid = perf.realized_at_cursor(
        df,
        preds,
        probas,
        cursor_t=cursor,
        time_col="t",
        target="label",
        delay=delay,
    )
    expected = int((df["t"].to_numpy() <= cursor - delay).sum())
    assert mid["n_arrived"] == expected
    assert mid["f1"] == 1.0  # perfect preds → F1 == 1 on arrived subset


def test_realized_series_accumulates(monkeypatch):
    cfg = get_config()
    df = _frame(n=800)

    def predict_fn(frame: pd.DataFrame):
        # near-perfect predictor; deterministic
        y = df.loc[frame.index, "label"].to_numpy()
        return y, y.astype(float)

    series = perf.realized_series(
        cfg, predict_fn, df=df, window=cfg.monitoring.window_size, step=100
    )
    assert len(series) >= 2
    arrived = [p["n_arrived"] for p in series]
    assert arrived[0] <= arrived[-1]
    assert series[-1]["n_arrived"] > 0
