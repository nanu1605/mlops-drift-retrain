"""Close-the-loop controller: drift -> retrain -> validate -> promote -> reload.

One ``tick`` polls the (label-free) drift signal; on a breach — debounced by
``controller.cooldown_seconds`` — it retrains **in-process** with the drift period included
(so the challenger can learn the new sub-population), runs the champion/challenger promotion
gate, and on promotion hot-reloads serving (``POST /reload``). Every branch is logged and
appended to a decision trail. ``run_loop`` drives ``tick`` every ``poll_seconds``; tests call
``tick`` directly with an injected store/clock/reload callback for determinism.

In a cluster the retrain would be a K8s Job; here it is a direct call (pure-local mode).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

from mlops_drift.config import Config, get_config
from mlops_drift.monitoring.metrics_exporter import REGISTRY as MONITOR_REGISTRY
from mlops_drift.monitoring.metrics_exporter import update_gauges
from mlops_drift.monitoring.monitor import evaluate_once
from mlops_drift.promotion.champion_challenger import run_promotion
from mlops_drift.serving.logging_store import RequestStore
from mlops_drift.training.train import run_training
from mlops_drift.utils.io import ensure_dir
from mlops_drift.utils.logging import get_logger

log = get_logger("controller")


@dataclass
class ControllerState:
    last_retrain_ts: float = 0.0
    retrains: int = 0
    promotions: int = 0
    rejections: int = 0
    ticks: int = 0


def default_reload_fn(cfg: Config):
    """POST /reload to serving; never crash the loop if serving is down."""

    def _reload() -> bool:
        import httpx

        try:
            r = httpx.post(f"{cfg.serving_url}/reload", timeout=10.0)
            r.raise_for_status()
            return True
        except Exception as exc:  # serving not up / transient
            log.warning("controller.reload_failed", error=str(exc))
            return False

    return _reload


def tick(
    cfg: Config,
    state: ControllerState,
    *,
    tracking_uri: str | None = None,
    store: RequestStore | None = None,
    reload_fn=None,
    now: float | None = None,
) -> dict:
    """One control step. Returns an event dict describing what happened."""
    now = time.time() if now is None else now
    state.ticks += 1
    mon = evaluate_once(cfg, tracking_uri=tracking_uri, store=store)
    # Refresh the monitor's Prometheus gauges so the /metrics endpoint (served by run_loop)
    # reflects the latest drift + realized-performance signal each poll.
    update_gauges(mon)
    event: dict = {
        "ts": now,
        "drift_detected": mon["drift_detected"],
        "drift_share": mon["drift_share"],
        "action": "none",
    }

    if not mon["drift_detected"]:
        event["action"] = "no_drift"
    elif now - state.last_retrain_ts < cfg.controller.cooldown_seconds:
        event["action"] = "cooldown_skip"
        log.info("controller.cooldown_skip", since_last=round(now - state.last_retrain_ts, 1))
    else:
        # --- drift confirmed + cooled down: retrain (in-process) ---
        log.info("controller.retrain_start", drift_share=mon["drift_share"])
        res = run_training(cfg, tracking_uri=tracking_uri, periods=("reference", "drift"))
        state.last_retrain_ts = now
        state.retrains += 1
        event["challenger_version"] = res["version"]

        decision = run_promotion(cfg, tracking_uri=tracking_uri)
        event["promotion"] = decision.to_dict()
        if decision.promote:
            state.promotions += 1
            reload_fn = reload_fn or default_reload_fn(cfg)
            event["reloaded"] = bool(reload_fn())
            event["action"] = "promoted"
            log.info("controller.promoted", version=decision.challenger_version)
        else:
            state.rejections += 1
            event["action"] = "rejected"
            log.info("controller.rejected", reason=decision.reason)

    _append_decision(cfg, event)
    return event


def _append_decision(cfg: Config, event: dict) -> None:
    path = cfg.controller_log_path
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, default=str) + "\n")


def _serve_monitor_metrics(port: int) -> bool:
    """Expose the monitor registry over HTTP so Prometheus can scrape drift/realized-F1.

    Best-effort: a bind failure (port taken, restricted env) is logged but must not crash
    the control loop — the drift→retrain→promote path matters more than the metrics bridge.
    """
    from prometheus_client import start_http_server

    try:
        start_http_server(port, registry=MONITOR_REGISTRY)
        log.info("controller.metrics_serving", port=port)
        return True
    except OSError as exc:
        log.warning("controller.metrics_bind_failed", port=port, error=str(exc))
        return False


def run_loop(
    cfg: Config | None = None,
    max_iters: int | None = None,
    sleep_fn=time.sleep,
    tracking_uri: str | None = None,
    reload_fn=None,
    serve_metrics: bool = True,
) -> ControllerState:
    """Drive ``tick`` every ``poll_seconds`` until ``max_iters`` (None = forever)."""
    cfg = cfg or get_config()
    if serve_metrics:
        _serve_monitor_metrics(cfg.controller.metrics_port)
    state = ControllerState()
    i = 0
    while max_iters is None or i < max_iters:
        tick(cfg, state, tracking_uri=tracking_uri, reload_fn=reload_fn)
        i += 1
        if max_iters is None or i < max_iters:
            sleep_fn(cfg.controller.poll_seconds)
    return state


def main() -> int:
    log.info("controller.start")
    run_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
