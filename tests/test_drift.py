"""Phase 4: Evidently drift detection — shifted window trips, in-dist does not. No MLflow."""

from __future__ import annotations

from mlops_drift.config import get_config
from mlops_drift.data.ingest import ingest
from mlops_drift.monitoring.drift import detect_drift
from mlops_drift.utils.io import read_parquet


def _dataset(cfg):
    path, _ = ingest(cfg)
    return read_parquet(path)


def test_shifted_window_detects_drift():
    cfg = get_config()
    df = _dataset(cfg)
    reference = df[df["period"] == "reference"].sample(n=1000, random_state=cfg.seed)
    current = df[df["period"] == "drift"].sample(n=1000, random_state=cfg.seed)

    res = detect_drift(reference, current, cfg)
    assert res.detected is True
    assert res.share >= cfg.thresholds.drift.dataset_drift_share
    assert res.n_features == 12
    assert len(res.per_feature) == 12
    assert any(v["drift_detected"] for v in res.per_feature.values())


def test_in_distribution_window_no_drift():
    cfg = get_config()
    df = _dataset(cfg)
    ref_rows = df[df["period"] == "reference"]
    reference = ref_rows.sample(n=1000, random_state=1)
    current = ref_rows.sample(n=1000, random_state=2)  # same distribution, different rows

    res = detect_drift(reference, current, cfg)
    assert res.detected is False
    assert res.share < cfg.thresholds.drift.dataset_drift_share
