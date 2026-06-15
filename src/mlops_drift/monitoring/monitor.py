"""One-shot monitor: drift + realized performance over the current window.

``evaluate_once`` is the unit the Phase 5 controller polls each tick for the (label-free) drift
signal. It also computes realized performance offline (delayed labels) for honest eval, appends
a row to the history file (Phase 6 plot), and writes a Prometheus textfile. ``make monitor``
runs it once and prints the result as JSON.
"""

from __future__ import annotations

import json
import sys
import time

from mlops_drift.config import Config, get_config
from mlops_drift.monitoring import performance as perf
from mlops_drift.monitoring.drift import detect_drift
from mlops_drift.monitoring.metrics_exporter import write_prom
from mlops_drift.serving.logging_store import RequestStore
from mlops_drift.serving.model_loader import ChampionLoader
from mlops_drift.utils.io import ensure_dir, read_parquet
from mlops_drift.utils.logging import get_logger

log = get_logger("monitoring.monitor")

# Minimum served rows before drift detection is meaningful; below this we report drift as n/a
# (Evidently on a handful of rows is noise) but still compute offline realized performance.
MIN_DRIFT_ROWS = 30


def evaluate_once(
    cfg: Config | None = None,
    tracking_uri: str | None = None,
    store: RequestStore | None = None,
) -> dict:
    cfg = cfg or get_config()

    loader = ChampionLoader(cfg, tracking_uri=tracking_uri)
    if not loader.ensure_loaded():
        raise RuntimeError("no champion model available; run training first")
    feature_cols = loader.feature_cols

    # --- drift over the latest served window (label-free) ---
    own_store = store is None
    store = store or RequestStore(cfg.request_db_path, feature_cols=feature_cols)
    current = store.read_window(limit=cfg.monitoring.window_size)
    drift = None
    if len(current) >= MIN_DRIFT_ROWS:
        reference = read_parquet(cfg.reference_dir / "reference.parquet")
        drift = detect_drift(reference, current, cfg)
    else:
        log.info("monitor.drift_skipped", rows=int(len(current)), min=MIN_DRIFT_ROWS)
    if own_store:
        store.close()

    # --- realized performance offline (delayed labels) ---
    realized = perf.realized_latest(cfg, loader.predict)

    result = {
        "ts": time.time(),
        "model_version": loader.version,
        "n_served": int(len(current)),
        "drift_detected": (drift.detected if drift else None),
        "drift_share": (drift.share if drift else None),
        "n_drifted": (drift.n_drifted if drift else None),
        "realized_f1": realized["f1"],
        "realized_pr_auc": realized["pr_auc"],
        "n_labels_arrived": realized["n_arrived"],
    }

    _append_history(cfg, result)
    write_prom(result, cfg.prom_textfile_path)
    log.info(
        "monitor.done",
        drift=result["drift_detected"],
        share=result["drift_share"],
        realized_f1=round(result["realized_f1"], 4)
        if result["realized_f1"] == result["realized_f1"]
        else None,
    )
    return result


def _append_history(cfg: Config, result: dict) -> None:
    path = cfg.history_path
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")


def main() -> int:
    result = evaluate_once()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
