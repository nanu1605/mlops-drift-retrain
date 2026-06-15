"""Replay production-like traffic into the serving API.

Streams rows (the drift period, by default) to ``POST /predict`` in time-ordered batches —
the stand-in for live inference traffic. The Phase 6 experiment passes an ``on_batch`` hook to
compute per-batch realized F1 and drive a controller tick between batches; ``make replay`` runs
it standalone to demonstrate the serving path + throughput.

``stream`` accepts an injected httpx ``client`` so tests can target the ASGI app in-process
(``httpx.ASGITransport``) without a real server.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import httpx
import pandas as pd

from mlops_drift.config import get_config
from mlops_drift.data.ingest import ingest
from mlops_drift.utils.io import read_parquet
from mlops_drift.utils.logging import get_logger

log = get_logger("replay")

OnBatch = Callable[[pd.DataFrame, list[int], list[float]], None]


def drift_stream(cfg) -> tuple[pd.DataFrame, list[str]]:
    """The drift period in time order + the feature columns."""
    path, _ = ingest(cfg)
    df = read_parquet(path)
    cols = [f"f{i}" for i in range(cfg.data.n_features)]
    stream = df[df["period"] == "drift"].sort_values(cfg.data.time_col).reset_index(drop=True)
    return stream, cols


def stream(
    url: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int = 500,
    on_batch: OnBatch | None = None,
    client: httpx.Client | None = None,
) -> list[dict]:
    """POST ``df`` to ``{url}/predict`` in batches. Returns one summary dict per batch."""
    own = client is None
    client = client or httpx.Client(base_url=url, timeout=30.0)
    summaries: list[dict] = []
    try:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            instances = batch[feature_cols].to_dict(orient="records")
            r = client.post("/predict", json={"instances": instances})
            r.raise_for_status()
            body = r.json()
            preds = [int(p) for p in body["predictions"]]
            probas = [float(p) for p in body["probabilities"]]
            if on_batch is not None:
                on_batch(batch, preds, probas)
            summaries.append(
                {"start": start, "n": len(batch), "model_version": body["model_version"]}
            )
    finally:
        if own:
            client.close()
    return summaries


def main() -> int:
    cfg = get_config()
    stream_df, cols = drift_stream(cfg)
    out = stream(cfg.serving_url, stream_df, cols)
    total = sum(s["n"] for s in out)
    log.info("replay.done", batches=len(out), rows=total, url=cfg.serving_url)
    print(f"replayed {total} rows in {len(out)} batches -> {cfg.serving_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
