"""Feature pipeline: determinism, no leakage, correct column selection."""

from __future__ import annotations

import numpy as np

from mlops_drift.config import get_config
from mlops_drift.data import features as feat
from mlops_drift.data import synthetic


def _df():
    return synthetic.generate(get_config())


def test_excludes_time_period_label():
    cols = feat.select_feature_cols(_df())
    for forbidden in ("t", "period", "label"):
        assert forbidden not in cols
    assert all(c.startswith("f") for c in cols)
    assert cols == [f"f{i}" for i in range(get_config().data.n_features)]


def test_deterministic_transform():
    df = _df()
    cols = feat.select_feature_cols(df)
    a = feat.FeaturePipeline(cols).fit(df).transform(df)
    b = feat.FeaturePipeline(cols).fit(df).transform(df)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_fit_on_train_does_not_peek_at_test():
    # Impute uses TRAIN medians only; a NaN in test is filled with the train median,
    # not the test median.
    df = _df()
    cols = feat.select_feature_cols(df)
    train = df.iloc[:1000].copy()
    test = df.iloc[1000:1100].copy()
    pipe = feat.FeaturePipeline(cols).fit(train)
    train_median_f0 = float(np.median(train["f0"].to_numpy()))

    test.loc[test.index[0], "f0"] = np.nan
    out = pipe.transform(test)
    assert np.isclose(out.iloc[0]["f0"], train_median_f0)
    assert feat.is_finite_frame(out)


def test_save_load_roundtrip(tmp_path):
    df = _df()
    cols = feat.select_feature_cols(df)
    pipe = feat.FeaturePipeline(cols).fit(df)
    p = pipe.save(tmp_path / "fp.joblib")
    loaded = feat.FeaturePipeline.load(p)
    assert loaded.feature_names() == pipe.feature_names()
    assert np.array_equal(loaded.transform(df).to_numpy(), pipe.transform(df).to_numpy())


def test_missing_columns_raise():
    df = _df()
    cols = feat.select_feature_cols(df)
    pipe = feat.FeaturePipeline(cols).fit(df)
    import pytest

    with pytest.raises(KeyError):
        pipe.transform(df.drop(columns=["f0"]))
