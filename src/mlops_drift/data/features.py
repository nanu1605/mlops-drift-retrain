"""Deterministic feature pipeline.

Tree model (RandomForest) → median-impute only, no scaling (scaling is a needless
no-op for trees). The fitted imputer is persisted so serving applies *identical*
transforms (train-fitted statistics — never refit on live data => no leakage).

Feature columns are exactly the ``f*`` columns. ``t`` (time axis), ``period`` (metadata)
and the target ``label`` are explicitly excluded — feeding ``t`` would leak time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Columns that are never features regardless of dataset.
NON_FEATURE_COLS = ("label", "period", "t")


def select_feature_cols(
    df: pd.DataFrame, target_col: str = "label", time_col: str = "t"
) -> list[str]:
    """All numeric columns except target/time/period. Stable, sorted by the f-index
    when names look like f0..fN, else lexicographic."""
    excluded = set(NON_FEATURE_COLS) | {target_col, time_col}
    cols = [c for c in df.columns if c not in excluded]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    def _key(c: str) -> tuple[int, object]:
        if c.startswith("f") and c[1:].isdigit():
            return (0, int(c[1:]))
        return (1, c)

    return sorted(cols, key=_key)


class FeaturePipeline:
    """Fit-on-train, transform-anything deterministic feature transformer."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols: list[str] = list(feature_cols)
        self._imputer = SimpleImputer(strategy="median")
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        self._imputer.fit(df[self.feature_cols].to_numpy(dtype=float))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit")
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"input missing feature columns: {missing}")
        x = self._imputer.transform(df[self.feature_cols].to_numpy(dtype=float))
        return pd.DataFrame(x, columns=self.feature_cols, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def feature_names(self) -> list[str]:
        return list(self.feature_cols)

    # --- persistence (reused by serving in Phase 3) ---
    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"feature_cols": self.feature_cols, "imputer": self._imputer, "fitted": self._fitted},
            p,
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> FeaturePipeline:
        obj = joblib.load(path)
        fp = cls(obj["feature_cols"])
        fp._imputer = obj["imputer"]
        fp._fitted = obj["fitted"]
        return fp


def build_pipeline(
    df: pd.DataFrame, target_col: str = "label", time_col: str = "t"
) -> FeaturePipeline:
    """Convenience: derive feature columns from ``df`` and return an unfitted pipeline."""
    return FeaturePipeline(select_feature_cols(df, target_col=target_col, time_col=time_col))


def is_finite_frame(x: pd.DataFrame) -> bool:
    return bool(np.isfinite(x.to_numpy(dtype=float)).all())
