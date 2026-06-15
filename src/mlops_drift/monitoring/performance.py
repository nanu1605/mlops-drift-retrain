"""Realized performance under delayed labels (offline over the dataset).

Labels arrive ``label_delay_steps`` after a prediction. Realized performance at a given
"current time" cursor only counts rows whose label has arrived: ``cursor_t - t >= delay``.
We re-score the **live champion** over the drift stream so the metric reflects whatever model
is currently promoted (the recovery the Phase 6 plot shows). This is intentionally decoupled
from the request store: the drift signal (label-free) is what triggers the controller; realized
F1 is for honest evaluation only.

``realized_at_cursor`` / ``realized_series`` accept a ``predict_fn`` so they're testable without
MLflow; ``champion_predict_fn`` wires the real champion via the serving ``ChampionLoader``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from mlops_drift.config import Config
from mlops_drift.data.features import select_feature_cols
from mlops_drift.data.ingest import ingest
from mlops_drift.training.evaluate import evaluate_classification
from mlops_drift.utils.io import read_parquet
from mlops_drift.utils.logging import get_logger

log = get_logger("monitoring.performance")

PredictFn = Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]]


def realized_at_cursor(
    df_sorted: pd.DataFrame,
    preds: np.ndarray,
    probas: np.ndarray,
    cursor_t: float,
    *,
    time_col: str,
    target: str,
    delay: int,
    window: int | None = None,
) -> dict:
    """Realized metrics over rows whose labels have arrived by ``cursor_t``.

    A row is eligible iff ``cursor_t - t >= delay``. If ``window`` is given, also restrict to
    ``t > cursor_t - window`` (a sliding window of recent-but-arrived labels).
    """
    t = df_sorted[time_col].to_numpy()
    arrived = (cursor_t - t) >= delay
    if window is not None:
        arrived &= t > (cursor_t - window)

    n = int(arrived.sum())
    if n == 0:
        return {
            "f1": float("nan"),
            "pr_auc": float("nan"),
            "n_arrived": 0,
            "cursor_t": float(cursor_t),
        }

    y = df_sorted[target].to_numpy()[arrived]
    m = evaluate_classification(y, preds[arrived], probas[arrived])
    return {
        "f1": float(m["f1"]),
        "pr_auc": float(m["pr_auc"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "n_arrived": n,
        "cursor_t": float(cursor_t),
    }


def _drift_stream(cfg: Config, df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    if df is None:
        path, _ = ingest(cfg)
        df = read_parquet(path)
    feats = select_feature_cols(df, target_col=cfg.data.target_col, time_col=cfg.data.time_col)
    stream = df[df["period"] == "drift"].sort_values(cfg.data.time_col).reset_index(drop=True)
    return stream, feats


def realized_series(
    cfg: Config,
    predict_fn: PredictFn,
    df: pd.DataFrame | None = None,
    window: int | None = None,
    step: int | None = None,
) -> list[dict]:
    """Sliding realized F1/PR-AUC over the drift stream — the time series Phase 6 plots."""
    stream, feats = _drift_stream(cfg, df)
    if stream.empty:
        return []
    preds, probas = predict_fn(stream[feats])
    preds, probas = np.asarray(preds).astype(int), np.asarray(probas, dtype=float)

    time_col, target = cfg.data.time_col, cfg.data.target_col
    delay = cfg.data.label_delay_steps
    window = window or cfg.monitoring.window_size
    step = step or max(1, window // 4)

    t = stream[time_col].to_numpy()
    t_min, t_max = float(t.min()), float(t.max())
    points: list[dict] = []
    cursor = t_min + delay
    while cursor <= t_max:
        points.append(
            realized_at_cursor(
                stream,
                preds,
                probas,
                cursor,
                time_col=time_col,
                target=target,
                delay=delay,
                window=window,
            )
        )
        cursor += step
    # always include the final cursor (all labels arrived)
    points.append(
        realized_at_cursor(
            stream,
            preds,
            probas,
            t_max,
            time_col=time_col,
            target=target,
            delay=delay,
            window=window,
        )
    )
    return points


def realized_latest(cfg: Config, predict_fn: PredictFn, df: pd.DataFrame | None = None) -> dict:
    """Realized perf at the latest cursor over the whole drift stream (cumulative arrived)."""
    stream, feats = _drift_stream(cfg, df)
    if stream.empty:
        return {"f1": float("nan"), "pr_auc": float("nan"), "n_arrived": 0, "cursor_t": 0.0}
    preds, probas = predict_fn(stream[feats])
    preds, probas = np.asarray(preds).astype(int), np.asarray(probas, dtype=float)
    t_max = float(stream[cfg.data.time_col].max())
    return realized_at_cursor(
        stream,
        preds,
        probas,
        t_max,
        time_col=cfg.data.time_col,
        target=cfg.data.target_col,
        delay=cfg.data.label_delay_steps,
    )


def champion_predict_fn(cfg: Config, tracking_uri: str | None = None) -> PredictFn:
    """Wire the live champion (serving ChampionLoader) as a predict callable."""
    from mlops_drift.serving.model_loader import ChampionLoader

    loader = ChampionLoader(cfg, tracking_uri=tracking_uri)
    if not loader.ensure_loaded():
        raise RuntimeError("no champion model available for realized performance")

    def _predict(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return loader.predict(frame)

    return _predict
