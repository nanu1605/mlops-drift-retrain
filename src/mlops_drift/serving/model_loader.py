"""Champion model loader with hot-reload.

Resolves ``models:/<name>@champion`` from the local sqlite MLflow registry and loads it as
a **sklearn Pipeline** (``mlflow.sklearn.load_model``) so serving has both ``predict`` and
``predict_proba`` (the pyfunc flavor exposes only ``predict``). The imputer lives inside the
pipeline, so input is the raw ``f*`` columns — no separate feature transform at serving.

Hot-reload is two-pronged (see plan):
  * ``reload()`` forces an immediate re-resolve (the Phase 5 controller calls this after a
    promotion, via ``POST /reload``).
  * ``start_refresh()`` spawns a daemon thread that re-resolves every ``refresh_seconds`` and
    swaps only when the champion alias points at a new version.

Swaps are guarded by a lock so in-flight predictions always see a consistent model.
"""

from __future__ import annotations

import threading
import time

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops_drift.config import Config
from mlops_drift.training.mlflow_utils import CHAMPION, setup_mlflow
from mlops_drift.utils.io import read_json
from mlops_drift.utils.logging import get_logger

log = get_logger("serving.loader")

SCHEMA_ARTIFACT = "feature_schema.json"


class ChampionLoader:
    """Load + hot-reload the ``@champion`` model version."""

    def __init__(self, cfg: Config, tracking_uri: str | None = None):
        self.cfg = cfg
        self.name = cfg.mlflow.registered_model
        self.refresh_seconds = cfg.serving.model_refresh_seconds
        self.tracking_uri = setup_mlflow(cfg, tracking_uri=tracking_uri)
        self._client = MlflowClient()

        self._lock = threading.Lock()
        self._model = None
        self._version: str | None = None
        self._feature_cols: list[str] = []
        self._loaded_at: float = 0.0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- resolution ---
    def _champion_version(self) -> str | None:
        try:
            return str(self._client.get_model_version_by_alias(self.name, CHAMPION).version)
        except Exception:
            return None

    def _resolve_feature_cols(self, run_id: str | None) -> list[str]:
        """Feature columns from the champion run's ``feature_schema.json`` artifact;
        fall back to ``f0..f{n_features-1}`` from config."""
        if run_id:
            try:
                local = self._client.download_artifacts(run_id, SCHEMA_ARTIFACT)
                return list(read_json(local)["feature_cols"])
            except Exception as exc:  # pragma: no cover - network/artifact edge
                log.warning("loader.schema_fallback", error=str(exc))
        return [f"f{i}" for i in range(self.cfg.data.n_features)]

    def _resolve(self, force: bool = False) -> bool:
        """(Re)load the champion if its version changed (or ``force``). Returns whether a
        swap happened."""
        version = self._champion_version()
        if version is None:
            return False
        if version == self._version and not force and self._model is not None:
            return False

        model = mlflow.sklearn.load_model(f"models:/{self.name}@{CHAMPION}")
        mv = self._client.get_model_version(self.name, version)
        feature_cols = self._resolve_feature_cols(mv.run_id)

        with self._lock:
            old = self._version
            self._model = model
            self._version = version
            self._feature_cols = feature_cols
            self._loaded_at = time.time()
        log.info("loader.loaded", model=self.name, version=version, previous=old)
        return True

    def ensure_loaded(self) -> bool:
        """Load on first use. Returns True if a champion is available."""
        if self._model is None:
            self._resolve(force=True)
        return self._model is not None

    def reload(self) -> bool:
        """Force an immediate re-resolve (controller hook). Returns whether a model is live."""
        self._resolve(force=True)
        return self._model is not None

    # --- inference ---
    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict on raw ``f*`` columns. Returns ``(preds, positive_class_proba)``."""
        with self._lock:
            model, cols = self._model, self._feature_cols
        if model is None:
            raise RuntimeError("no champion model loaded")
        x = df[cols]
        preds = model.predict(x)
        probas = model.predict_proba(x)[:, 1]
        return np.asarray(preds).astype(int), np.asarray(probas, dtype=float)

    # --- background refresh ---
    def start_refresh(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop, name="champion-refresh", daemon=True
        )
        self._thread.start()

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            try:
                self._resolve()
            except Exception as exc:  # pragma: no cover - resilience
                log.warning("loader.refresh_error", error=str(exc))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # --- introspection ---
    @property
    def version(self) -> str | None:
        return self._version

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    @property
    def loaded_at(self) -> float:
        return self._loaded_at
