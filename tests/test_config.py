"""Config loads correctly and fails loudly on missing keys."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlops_drift.config import Config, get_config, load_config


def test_config_loads_defaults():
    cfg = get_config()
    assert cfg.seed == 42
    assert cfg.model.name
    assert cfg.thresholds.promotion.f1_margin >= 0
    # path helpers resolve to absolute
    assert cfg.raw_dir.is_absolute()


def test_missing_required_key_raises(tmp_path: Path):
    # config.yaml missing the 'model' section -> ValidationError
    bad_cfg = tmp_path / "config.yaml"
    bad_cfg.write_text(
        textwrap.dedent(
            """
            seed: 1
            paths: {data_raw: data/raw, data_processed: data/processed,
                    data_reference: data/reference, artifacts: artifacts}
            """
        )
    )
    thr = tmp_path / "thresholds.yaml"
    thr.write_text(
        "drift: {dataset_drift_share: 0.5, feature_stattest_threshold: 0.05}\n"
        "promotion: {f1_margin: 0.01}\n"
        "validation: {f1_floor: 0.7}\n"
    )
    with pytest.raises(ValidationError):
        load_config(bad_cfg, thr)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", tmp_path / "nada.yaml")


def test_config_type():
    assert isinstance(get_config(), Config)
