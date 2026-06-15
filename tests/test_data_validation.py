"""pandera validation accepts clean frames and rejects malformed ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from mlops_drift.data import validation


def _clean(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"f0": rng.normal(size=n), "f1": rng.normal(size=n), "label": rng.integers(0, 2, n)}
    )


FEATS = ["f0", "f1"]


def test_clean_frame_passes():
    out = validation.validate(_clean(), FEATS, require_label=True)
    assert len(out) == 50


def test_nan_rejected():
    df = _clean()
    df.loc[0, "f0"] = np.nan
    with pytest.raises((SchemaError, SchemaErrors)):
        validation.validate(df, FEATS, require_label=True)


def test_inf_rejected():
    df = _clean()
    df.loc[0, "f1"] = np.inf
    with pytest.raises((SchemaError, SchemaErrors)):
        validation.validate(df, FEATS, require_label=True)


def test_bad_label_rejected():
    df = _clean()
    df.loc[0, "label"] = 2
    with pytest.raises((SchemaError, SchemaErrors)):
        validation.validate(df, FEATS, require_label=True)


def test_label_optional_at_serving():
    df = _clean().drop(columns=["label"])
    out = validation.validate(df, FEATS, require_label=False)
    assert "f0" in out.columns
