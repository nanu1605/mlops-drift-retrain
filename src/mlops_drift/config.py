"""Typed configuration loader (pydantic-settings).

Loads ``configs/config.yaml`` + ``configs/thresholds.yaml`` into a single typed
object. Raises clearly if a required key is missing. No magic numbers anywhere else
in the codebase — everything funnels through here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/mlops_drift/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
DEFAULT_THRESHOLDS = REPO_ROOT / "configs" / "thresholds.yaml"


class Paths(BaseModel):
    data_raw: str
    data_processed: str
    data_reference: str
    artifacts: str


class DataCfg(BaseModel):
    n_reference: int
    n_drift: int
    n_features: int
    attack_rate: float
    drift_attack_rate: float
    drift_feature_frac: float
    label_delay_steps: int
    target_col: str
    time_col: str


class ModelCfg(BaseModel):
    name: str
    family: str
    params: dict[str, Any]


class SplitCfg(BaseModel):
    train_frac: float


class MonitoringCfg(BaseModel):
    window_size: int
    reference_sample: int


class ServingCfg(BaseModel):
    host: str
    port: int
    model_refresh_seconds: int
    model_uri: str


class MLflowCfg(BaseModel):
    tracking_uri: str  # sqlite:///<relative-or-abs>.db (DB backend required for registry)
    artifact_location: str
    experiment: str
    registered_model: str
    ui_host: str
    ui_port: int


class ControllerCfg(BaseModel):
    poll_seconds: int
    cooldown_seconds: int


class DriftThresholds(BaseModel):
    dataset_drift_share: float
    feature_stattest_threshold: float


class PromotionThresholds(BaseModel):
    f1_margin: float


class ValidationThresholds(BaseModel):
    f1_floor: float


class Thresholds(BaseModel):
    drift: DriftThresholds
    promotion: PromotionThresholds
    validation: ValidationThresholds


class Config(BaseSettings):
    """Top-level config. Nested models give clear errors on missing keys."""

    model_config = SettingsConfigDict(extra="forbid")

    seed: int
    paths: Paths
    data: DataCfg
    model: ModelCfg
    split: SplitCfg
    monitoring: MonitoringCfg
    serving: ServingCfg
    mlflow: MLflowCfg
    controller: ControllerCfg
    thresholds: Thresholds = Field(...)

    # --- absolute-path helpers (resolved against repo root) ---
    def abspath(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def raw_dir(self) -> Path:
        return self.abspath(self.paths.data_raw)

    @property
    def processed_dir(self) -> Path:
        return self.abspath(self.paths.data_processed)

    @property
    def reference_dir(self) -> Path:
        return self.abspath(self.paths.data_reference)

    @property
    def artifacts_dir(self) -> Path:
        return self.abspath(self.paths.artifacts)

    def resolved_tracking_uri(self) -> str:
        """Resolve a ``sqlite:///<relative>`` tracking URI to an absolute path so the
        same DB is hit regardless of process cwd. Non-sqlite URIs pass through."""
        uri = self.mlflow.tracking_uri
        prefix = "sqlite:///"
        if uri.startswith(prefix):
            rel = uri[len(prefix) :]
            if not rel.startswith("/"):
                return f"{prefix}{self.abspath(rel)}"
        return uri


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required config file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping.")
    return data


def load_config(
    config_path: Path | str | None = None,
    thresholds_path: Path | str | None = None,
) -> Config:
    """Load and validate the full configuration.

    Env overrides: ``MLOPS_CONFIG`` / ``MLOPS_THRESHOLDS`` point at alternate files.
    Raises ``pydantic.ValidationError`` with a clear message if a key is missing.
    """
    cfg_file = Path(config_path or os.getenv("MLOPS_CONFIG") or DEFAULT_CONFIG)
    thr_file = Path(thresholds_path or os.getenv("MLOPS_THRESHOLDS") or DEFAULT_THRESHOLDS)
    raw = _read_yaml(cfg_file)
    raw["thresholds"] = _read_yaml(thr_file)
    return Config(**raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached singleton for normal use."""
    return load_config()
