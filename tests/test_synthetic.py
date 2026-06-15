"""Synthetic generator: determinism + drift shape (accuracy should degrade on drift)."""

from __future__ import annotations

import pandas as pd

from mlops_drift.config import get_config
from mlops_drift.data import synthetic


def test_deterministic():
    cfg = get_config()
    a = synthetic.generate(cfg)
    b = synthetic.generate(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_shape_and_schema():
    cfg = get_config()
    df = synthetic.generate(cfg)
    assert len(df) == cfg.data.n_reference + cfg.data.n_drift
    assert cfg.data.target_col in df.columns
    assert cfg.data.time_col in df.columns
    # time axis strictly increasing; reference precedes drift
    assert df[cfg.data.time_col].is_monotonic_increasing
    assert (
        df.loc[df["period"] == "reference", cfg.data.time_col].max()
        < df.loc[df["period"] == "drift", cfg.data.time_col].min()
    )


def test_drift_period_harder_for_reference_model():
    # A model trained on reference should do worse on drift than on held-out reference.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    cfg = get_config()
    df = synthetic.generate(cfg)
    feat = [c for c in df.columns if c.startswith("f")]
    ref = df[df["period"] == "reference"]
    drift = df[df["period"] == "drift"]
    cut = int(len(ref) * 0.8)
    clf = RandomForestClassifier(n_estimators=80, random_state=cfg.seed, n_jobs=-1)
    clf.fit(ref.iloc[:cut][feat], ref.iloc[:cut][cfg.data.target_col])

    f1_ref = f1_score(ref.iloc[cut:][cfg.data.target_col], clf.predict(ref.iloc[cut:][feat]))
    f1_drift = f1_score(drift[cfg.data.target_col], clf.predict(drift[feat]))
    assert f1_drift < f1_ref  # drift genuinely degrades the reference model
