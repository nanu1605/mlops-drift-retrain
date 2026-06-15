"""Time-aware splitting. Train on earlier rows, evaluate on later — never shuffle
across the time boundary. The single most important guard against leakage."""

from __future__ import annotations

import pandas as pd


def time_aware_split(
    df: pd.DataFrame,
    train_frac: float,
    time_col: str = "t",
    period: str | None = None,
    period_col: str = "period",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by ``time_col`` and cut at ``train_frac``: earliest → train, latest → test.

    If ``period`` is given, restrict to that period first (e.g. only the in-distribution
    reference period for the honest baseline). Asserts the train/test time ranges do not
    overlap.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    work = df
    if period is not None:
        if period_col not in work.columns:
            raise KeyError(f"period column {period_col!r} not in frame")
        work = work[work[period_col] == period]
    work = work.sort_values(time_col).reset_index(drop=True)
    if len(work) < 2:
        raise ValueError("not enough rows to split")
    cut = int(len(work) * train_frac)
    cut = max(1, min(cut, len(work) - 1))
    train, test = work.iloc[:cut].copy(), work.iloc[cut:].copy()
    # No leakage: every train timestamp strictly precedes every test timestamp.
    assert train[time_col].max() < test[time_col].min(), "time-aware split overlap"
    return train, test
