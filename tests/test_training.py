"""Time-aware split (no leakage) and baseline-beats-trivial acceptance."""

from __future__ import annotations

import pytest

from mlops_drift.config import get_config
from mlops_drift.data import synthetic
from mlops_drift.data.split import time_aware_split
from mlops_drift.training.evaluate import beats


def test_time_aware_split_no_overlap():
    cfg = get_config()
    df = synthetic.generate(cfg)
    train, test = time_aware_split(df, cfg.split.train_frac, time_col=cfg.data.time_col)
    # every train timestamp strictly precedes every test timestamp
    assert train[cfg.data.time_col].max() < test[cfg.data.time_col].min()
    assert len(train) + len(test) == len(df)


def test_split_restricts_to_period():
    cfg = get_config()
    df = synthetic.generate(cfg)
    train, test = time_aware_split(
        df, cfg.split.train_frac, time_col=cfg.data.time_col, period="reference"
    )
    assert set(train["period"]).union(test["period"]) == {"reference"}


def test_bad_train_frac_raises():
    cfg = get_config()
    df = synthetic.generate(cfg)
    with pytest.raises(ValueError):
        time_aware_split(df, 1.5, time_col=cfg.data.time_col)


def test_baseline_beats_trivial_on_f1_and_pr_auc():
    # End-to-end (small): RF must beat majority on BOTH f1 and pr_auc, and the runner
    # must be deterministic.
    from mlops_drift.training.baseline import run_baseline

    r1 = run_baseline()
    rf = r1["metrics"]["random_forest_in_distribution"]
    dummy = r1["metrics"]["majority_trivial"]
    assert beats(rf, dummy, keys=("f1", "pr_auc"))
    assert r1["beats_trivial_on_f1_and_pr_auc"] is True

    # determinism: same seed -> identical headline metrics (to numerical tolerance;
    # PR-AUC summation is not bit-stable across thread pools, ~1 ULP jitter).
    r2 = run_baseline()
    rf2 = r2["metrics"]["random_forest_in_distribution"]
    assert rf["f1"] == pytest.approx(rf2["f1"], abs=1e-9)
    assert rf["pr_auc"] == pytest.approx(rf2["pr_auc"], abs=1e-9)


def test_drift_preview_is_worse_than_in_distribution():
    # Sanity: the held-out drift period degrades the reference model (sets up the
    # whole experiment narrative). Not an acceptance gate, but a guard on the data.
    from mlops_drift.training.baseline import run_baseline

    r = run_baseline()
    indist = r["metrics"]["random_forest_in_distribution"]["f1"]
    drift = r["metrics"]["random_forest_drift_preview"]["f1"]
    assert drift < indist
