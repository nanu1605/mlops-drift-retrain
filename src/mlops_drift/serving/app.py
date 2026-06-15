"""FastAPI serving app (local process).

Endpoints:
  POST /predict  — score a batch of raw-feature instances; logs each to the request store.
  GET  /health   — liveness + live model version (503 if no champion resolves).
  GET  /metrics  — Prometheus exposition.
  POST /reload   — force a champion re-resolve (Phase 5 controller calls this on promotion).

Built via ``create_app(cfg, tracking_uri, refresh)`` so tests can point at a tmp MLflow
store and disable the background refresh thread for determinism. Module-level ``app`` is the
uvicorn entrypoint (``mlops_drift.serving.app:app``).
"""

from __future__ import annotations

import contextlib
import time
from contextlib import asynccontextmanager

import pandas as pd
import pandera.errors as pa_errors
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from mlops_drift.config import Config, get_config
from mlops_drift.data.validation import validate
from mlops_drift.serving.logging_store import RequestStore
from mlops_drift.serving.model_loader import ChampionLoader
from mlops_drift.serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)
from mlops_drift.utils.io import git_sha
from mlops_drift.utils.logging import get_logger

log = get_logger("serving.app")

# --- Prometheus metrics (module-scope singletons: re-creating the app must not re-register) ---
PREDICTIONS = Counter(
    "predictions_total", "Predictions served, by predicted class", ["predicted_class"]
)
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Latency of /predict in seconds")
REQUESTS = Counter(
    "requests_total", "HTTP requests by endpoint and status code", ["endpoint", "code"]
)
PREDICTION_ERRORS = Counter("prediction_errors_total", "Rejected/failed prediction requests")
MODEL_VERSION = Gauge("model_version", "Live champion model version (numeric)")
MODEL_INFO = Info("model", "Live champion model metadata")


def create_app(
    cfg: Config | None = None, tracking_uri: str | None = None, refresh: bool = True
) -> FastAPI:
    cfg = cfg or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader = ChampionLoader(cfg, tracking_uri=tracking_uri)
        loader.ensure_loaded()
        store = RequestStore(
            cfg.request_db_path, feature_cols=loader.feature_cols or _fallback_cols(cfg)
        )
        app.state.loader = loader
        app.state.store = store
        app.state.start_time = time.time()
        _publish_version(loader)
        if refresh:
            loader.start_refresh()
        log.info("serving.startup", version=loader.version, refresh=refresh)
        try:
            yield
        finally:
            loader.stop()
            store.close()

    app = FastAPI(title="mlops-drift serving", version="0.3.0", lifespan=lifespan)

    @app.post("/predict")
    def predict(req: PredictRequest, request: Request) -> Response:
        loader: ChampionLoader = request.app.state.loader
        store: RequestStore = request.app.state.store
        if loader.version is None and not loader.ensure_loaded():
            PREDICTION_ERRORS.inc()
            REQUESTS.labels("/predict", "503").inc()
            return JSONResponse({"detail": "no champion model available"}, status_code=503)

        cols = loader.feature_cols
        frame = pd.DataFrame(req.instances)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            PREDICTION_ERRORS.inc()
            REQUESTS.labels("/predict", "400").inc()
            return JSONResponse({"detail": f"missing feature columns: {missing}"}, status_code=400)

        try:
            frame = validate(frame[cols], cols, require_label=False)
        except pa_errors.SchemaError as exc:
            PREDICTION_ERRORS.inc()
            REQUESTS.labels("/predict", "400").inc()
            return JSONResponse({"detail": f"invalid features: {exc}"}, status_code=400)

        start = time.perf_counter()
        preds, probas = loader.predict(frame)
        PREDICT_LATENCY.observe(time.perf_counter() - start)

        store.log(frame, preds, probas, version=loader.version, ts=time.time())
        for p in preds:
            PREDICTIONS.labels(str(int(p))).inc()
        REQUESTS.labels("/predict", "200").inc()
        return JSONResponse(
            PredictResponse(
                predictions=[int(p) for p in preds],
                probabilities=[float(p) for p in probas],
                model_version=loader.version,
            ).model_dump()
        )

    @app.get("/health")
    def health(request: Request) -> Response:
        loader: ChampionLoader = request.app.state.loader
        uptime = time.time() - request.app.state.start_time
        ok = loader.version is not None
        code = "200" if ok else "503"
        REQUESTS.labels("/health", code).inc()
        body = HealthResponse(
            status="ok" if ok else "no_model",
            model_version=loader.version,
            feature_cols=loader.feature_cols,
            uptime_s=round(uptime, 3),
        ).model_dump()
        return JSONResponse(body, status_code=int(code))

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/reload")
    def reload(request: Request) -> Response:
        loader: ChampionLoader = request.app.state.loader
        ok = loader.reload()
        _publish_version(loader)
        REQUESTS.labels("/reload", "200").inc()
        return JSONResponse(ReloadResponse(reloaded=ok, model_version=loader.version).model_dump())

    return app


def _fallback_cols(cfg: Config) -> list[str]:
    return [f"f{i}" for i in range(cfg.data.n_features)]


def _publish_version(loader: ChampionLoader) -> None:
    if loader.version is not None:
        with contextlib.suppress(TypeError, ValueError):
            MODEL_VERSION.set(float(loader.version))
        MODEL_INFO.info({"version": str(loader.version), "git_sha": git_sha()})


app = create_app()
