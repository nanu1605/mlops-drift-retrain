"""Drift experiment — live stack end to end, producing docs/images/drift_recovery.png.

Boots a real serving subprocess on a fresh reference-only champion, replays the drift period
to ``POST /predict`` in time-ordered batches, and drives the **real controller** (retrain →
promote → ``POST /reload``) between batches. Each batch's realized F1 (live champion's
predictions vs that batch's labels) is the timeline; the curve dips while the stale champion
serves drifted traffic, then recovers once the loop promotes a drift-trained challenger.

Reproducible: the run resets to a fresh baseline champion and clears runtime state up front, so
the recovery curve is the same every time (models are deterministic); only wall-clock varies.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import subprocess

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

from mlops_drift.config import REPO_ROOT, Config, get_config
from mlops_drift.controller.loop import ControllerState, tick
from mlops_drift.experiments import replay
from mlops_drift.training.evaluate import evaluate_classification
from mlops_drift.training.mlflow_utils import CHAMPION, setup_mlflow
from mlops_drift.training.train import run_training
from mlops_drift.utils.io import ensure_dir
from mlops_drift.utils.logging import get_logger

log = get_logger("experiment")

DEFAULT_OUT = REPO_ROOT / "docs" / "images" / "drift_recovery.png"
BATCH_SIZE = 400
WARMUP_BATCHES = 4  # serve this many drifted batches before allowing a retrain (shapes the dip)


def _reset_baseline_champion(cfg: Config) -> str:
    """Train a fresh reference-only model and force it to be @champion (start at the dip)."""
    res = run_training(cfg, periods=("reference",))
    client = MlflowClient()
    client.set_registered_model_alias(cfg.mlflow.registered_model, CHAMPION, res["version"])
    log.info("experiment.baseline", version=res["version"])
    return str(res["version"])


def _clear_runtime(cfg: Config) -> None:
    for p in (cfg.request_db_path, cfg.controller_log_path, cfg.history_path):
        for suffix in ("", "-wal", "-shm"):
            f = Path(str(p) + suffix)
            if f.exists():
                f.unlink()


def _wait_health(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"serving did not become healthy at {url}")


def run_experiment(
    cfg: Config | None = None, batch_size: int = BATCH_SIZE, out: Path | str = DEFAULT_OUT
) -> dict:
    cfg = cfg or get_config()
    setup_mlflow(cfg)
    _reset_baseline_champion(cfg)
    _clear_runtime(cfg)

    host = "127.0.0.1"
    port = cfg.serving.port
    url = f"http://{host}:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mlops_drift.serving.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
    )
    timeline: list[dict] = []
    promoted_t: float | None = None
    detected_t: float | None = None
    try:
        _wait_health(url)
        stream_df, cols = replay.drift_stream(cfg)
        state = ControllerState(last_retrain_ts=-1e9)  # first eligible tick may retrain
        batch_idx = {"i": 0}

        def on_batch(batch, preds, probas):
            nonlocal promoted_t, detected_t
            i = batch_idx["i"]
            m = evaluate_classification(batch[cfg.data.target_col].to_numpy(), preds, probas)
            t_mid = float(batch[cfg.data.time_col].median())
            client = MlflowClient()
            version = client.get_model_version_by_alias(
                cfg.mlflow.registered_model, CHAMPION
            ).version
            timeline.append({"t": t_mid, "f1": float(m["f1"]), "version": str(version)})

            if i >= WARMUP_BATCHES:
                ev = tick(cfg, state, now=float(i))
                if ev.get("drift_detected") and detected_t is None:
                    detected_t = t_mid
                if ev.get("action") == "promoted" and promoted_t is None:
                    promoted_t = t_mid
            batch_idx["i"] = i + 1

        replay.stream(url, stream_df, cols, batch_size=batch_size, on_batch=on_batch)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    out = _plot_recovery(timeline, detected_t, promoted_t, out)
    pre = [p["f1"] for p in timeline if promoted_t is None or p["t"] <= (promoted_t or 0)]
    post = [p["f1"] for p in timeline if promoted_t is not None and p["t"] > promoted_t]
    summary = {
        "out": str(out),
        "pre_f1": round(sum(pre) / len(pre), 4) if pre else None,
        "post_f1": round(sum(post) / len(post), 4) if post else None,
        "detected_t": detected_t,
        "promoted_t": promoted_t,
        "n_batches": len(timeline),
    }
    log.info("experiment.done", **summary)
    return summary


def _plot_recovery(timeline, detected_t, promoted_t, out: Path | str) -> Path:
    ts = [p["t"] for p in timeline]
    f1 = [p["f1"] for p in timeline]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ts, f1, marker="o", ms=4, lw=1.8, color="#0050b0", label="realized F1 (per batch)")
    if detected_t is not None:
        ax.axvline(detected_t, color="#b00020", ls="--", lw=1.5, label="drift detected")
    if promoted_t is not None:
        ax.axvline(promoted_t, color="#0a7d2c", ls="-.", lw=1.5, label="model promoted")
    ax.set_xlabel("time (t, drift period)")
    ax.set_ylabel("realized F1")
    ax.set_ylim(0, 1)
    ax.set_title("Drift → detect → retrain → promote → recover", fontsize=12, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    s = run_experiment()
    print(
        f"experiment: pre_f1={s['pre_f1']} post_f1={s['post_f1']} "
        f"promoted_t={s['promoted_t']} -> {s['out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
