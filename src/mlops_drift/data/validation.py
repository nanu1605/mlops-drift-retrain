"""pandera schema validation. Used at ingest (Phase 0/1) and at serving (Phase 3) to
reject/flag malformed feature frames before they reach a model."""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema


def feature_schema(feature_cols: list[str], require_label: bool = True) -> DataFrameSchema:
    """Schema for a feature frame: each feature finite float; optional binary label."""
    cols: dict[str, Column] = {
        c: Column(
            float,
            checks=Check(
                lambda s: s.notna().all() & s.apply(_finite).all(), error="non-finite values"
            ),
            coerce=True,
            nullable=False,
        )
        for c in feature_cols
    }
    if require_label:
        cols["label"] = Column(
            int, checks=Check.isin([0, 1], error="label must be 0/1"), coerce=True, nullable=False
        )
    return DataFrameSchema(cols, strict=False)


def _finite(v: float) -> bool:
    return v not in (float("inf"), float("-inf")) and v == v  # last clause rejects NaN


def validate(df: pd.DataFrame, feature_cols: list[str], require_label: bool = True) -> pd.DataFrame:
    """Validate ``df`` against the feature schema. Raises ``pandera.errors.SchemaError``
    on violation. Returns the (coerced) frame on success."""
    return feature_schema(feature_cols, require_label=require_label).validate(df, lazy=False)
