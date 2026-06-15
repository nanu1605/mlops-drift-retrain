"""Prometheus exposition for the monitor.

Uses a **dedicated registry** (not the serving default registry) so importing this module
never collides with serving's metric definitions. The one-shot monitor writes a Prometheus
**textfile** (node_exporter textfile-collector format) that a local Prometheus can scrape
without a push gateway — the pragmatic local-mode path (spec §0.3).
"""

from __future__ import annotations

import math
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, generate_latest, write_to_textfile

from mlops_drift.utils.io import ensure_dir

REGISTRY = CollectorRegistry()

DRIFT_DETECTED = Gauge("drift_detected", "Dataset drift flagged (1/0)", registry=REGISTRY)
DRIFT_SHARE = Gauge("drift_share", "Share of drifted features", registry=REGISTRY)
DRIFT_FEATURES_DRIFTED = Gauge(
    "drift_features_drifted", "Number of drifted features", registry=REGISTRY
)
REALIZED_F1 = Gauge("realized_f1", "Realized F1 on arrived delayed labels", registry=REGISTRY)
REALIZED_PR_AUC = Gauge("realized_pr_auc", "Realized PR-AUC on arrived labels", registry=REGISTRY)
REALIZED_LABELS_ARRIVED = Gauge(
    "realized_labels_arrived", "Count of delayed labels arrived", registry=REGISTRY
)
MONITOR_MODEL_VERSION = Gauge(
    "monitor_model_version", "Champion version the monitor scored", registry=REGISTRY
)


def _set(gauge: Gauge, value) -> None:
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if not math.isnan(v):
        gauge.set(v)


def update_gauges(result: dict) -> None:
    """Set the registry gauges from an ``evaluate_once`` result dict."""
    _set(DRIFT_DETECTED, 1.0 if result.get("drift_detected") else 0.0)
    _set(DRIFT_SHARE, result.get("drift_share"))
    _set(DRIFT_FEATURES_DRIFTED, result.get("n_drifted"))
    _set(REALIZED_F1, result.get("realized_f1"))
    _set(REALIZED_PR_AUC, result.get("realized_pr_auc"))
    _set(REALIZED_LABELS_ARRIVED, result.get("n_labels_arrived"))
    _set(MONITOR_MODEL_VERSION, result.get("model_version"))


def render_prom(result: dict) -> str:
    update_gauges(result)
    return generate_latest(REGISTRY).decode("utf-8")


def write_prom(result: dict, path: Path | str) -> Path:
    update_gauges(result)
    p = Path(path)
    ensure_dir(p.parent)
    write_to_textfile(str(p), REGISTRY)
    return p
