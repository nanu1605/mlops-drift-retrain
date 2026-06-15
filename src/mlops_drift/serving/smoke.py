"""Smoke test for a running serving process (`make smoke`).

Posts a valid batch (sampled from the reference window) and asserts a schema-correct 200;
posts a malformed body and asserts a 4xx; checks ``/metrics`` exposes the prediction counter.
Exits nonzero on any failure. Assumes serving is already up (``make up``) and a champion has
been trained (``make train``).
"""

from __future__ import annotations

import sys

import httpx

from mlops_drift.config import get_config
from mlops_drift.utils.io import read_parquet
from mlops_drift.utils.logging import get_logger

log = get_logger("serving.smoke")


def main() -> int:
    cfg = get_config()
    host = "127.0.0.1" if cfg.serving.host == "0.0.0.0" else cfg.serving.host
    base = f"http://{host}:{cfg.serving.port}"
    cols = [f"f{i}" for i in range(cfg.data.n_features)]

    ref = read_parquet(cfg.reference_dir / "reference.parquet")
    instances = ref[cols].head(3).to_dict(orient="records")

    with httpx.Client(base_url=base, timeout=10.0) as client:
        # valid request
        r = client.post("/predict", json={"instances": instances})
        assert r.status_code == 200, f"/predict expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert len(body["predictions"]) == 3, body
        assert all(0.0 <= p <= 1.0 for p in body["probabilities"]), body
        assert body["model_version"], body
        log.info("smoke.predict_ok", version=body["model_version"], preds=body["predictions"])

        # malformed request → 4xx (missing a feature)
        bad = [dict.fromkeys(cols[:-1], 0.0)]
        rb = client.post("/predict", json={"instances": bad})
        assert 400 <= rb.status_code < 500, f"bad input expected 4xx, got {rb.status_code}"
        log.info("smoke.bad_input_ok", code=rb.status_code)

        # metrics exposed
        rm = client.get("/metrics")
        assert rm.status_code == 200 and "predictions_total" in rm.text, "metrics missing"
        log.info("smoke.metrics_ok")

    print("smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
