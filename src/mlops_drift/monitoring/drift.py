"""Label-free data drift detection (Evidently).

Compares a rolling window of served traffic (raw ``f*`` features) against the persisted
reference window. Evidently's ``DataDriftPreset`` gives a per-feature drift flag/score and the
share of drifted columns; the **boolean signal is our own threshold** on that share
(``thresholds.drift.dataset_drift_share``), not Evidently's internal default — config-driven so
Phase 5 can tune the trip point.

Evidently 0.4.40's ``Report.as_dict()`` returns two metric blocks (``DatasetDriftMetric`` and
``DataDriftTable``); we extract by **searching for result keys** rather than positional index so
a patch bump can't silently break parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from mlops_drift.config import Config
from mlops_drift.data.features import select_feature_cols
from mlops_drift.utils.logging import get_logger

log = get_logger("monitoring.drift")


@dataclass
class DriftResult:
    detected: bool
    share: float
    n_drifted: int
    n_features: int
    per_feature: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "drift_detected": self.detected,
            "drift_share": self.share,
            "n_drifted": self.n_drifted,
            "n_features": self.n_features,
        }


def _find_result(metrics: list[dict], key: str) -> dict | None:
    for m in metrics:
        r = m.get("result", {})
        if key in r:
            return r
    return None


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: Config) -> DriftResult:
    """Run Evidently DataDrift over the shared feature columns and apply our share threshold."""
    feats = select_feature_cols(
        reference, target_col=cfg.data.target_col, time_col=cfg.data.time_col
    )
    feats = [c for c in feats if c in current.columns]
    if not feats:
        raise ValueError("no shared feature columns between reference and current windows")

    ref = reference[feats].reset_index(drop=True)
    cur = current[feats].reset_index(drop=True)

    report = Report(
        metrics=[
            DataDriftPreset(stattest_threshold=cfg.thresholds.drift.feature_stattest_threshold)
        ]
    )
    report.run(reference_data=ref, current_data=cur)
    metrics = report.as_dict().get("metrics", [])

    share_res = _find_result(metrics, "share_of_drifted_columns")
    table_res = _find_result(metrics, "drift_by_columns")
    if share_res is None:
        raise RuntimeError("Evidently output missing share_of_drifted_columns; API changed?")

    share = float(share_res["share_of_drifted_columns"])
    n_drifted = int(share_res.get("number_of_drifted_columns", 0))
    n_features = int(share_res.get("number_of_columns", len(feats)))

    per_feature: dict[str, dict] = {}
    if table_res is not None:
        for name, info in table_res["drift_by_columns"].items():
            per_feature[name] = {
                "drift_detected": bool(info.get("drift_detected")),
                "drift_score": info.get("drift_score"),
                "stattest": info.get("stattest_name"),
            }

    detected = share >= cfg.thresholds.drift.dataset_drift_share
    log.info("drift.detect", detected=detected, share=round(share, 4), n_drifted=n_drifted)
    return DriftResult(
        detected=detected,
        share=share,
        n_drifted=n_drifted,
        n_features=n_features,
        per_feature=per_feature,
    )
