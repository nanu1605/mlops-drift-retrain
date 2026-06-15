"""Pydantic request/response models for the serving API.

The request body carries a batch of instances, each a mapping of feature name → value
(e.g. ``{"f0": 0.1, ..., "f11": 2.3}``). Per-feature presence/finiteness is enforced
downstream by the pandera schema (``data/validation.validate``) against the live model's
feature columns; here we only enforce the envelope shape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}

    instances: list[dict[str, float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    feature_cols: list[str]
    uptime_s: float


class ReloadResponse(BaseModel):
    reloaded: bool
    model_version: str | None
