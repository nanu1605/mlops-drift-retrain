"""Model-validation gate (CI + ``make validate``).

Loads the live ``@champion`` and asserts realized F1 on the drift holdout clears
``thresholds.validation.f1_floor``. CI blocks a merge/promotion if a sub-floor model is live;
``main`` exits nonzero on failure.
"""

from __future__ import annotations

import sys

import mlflow

from mlops_drift.config import Config, get_config
from mlops_drift.data.features import select_feature_cols
from mlops_drift.data.ingest import ingest
from mlops_drift.data.split import time_aware_split
from mlops_drift.training.evaluate import evaluate_classification
from mlops_drift.training.mlflow_utils import CHAMPION, setup_mlflow
from mlops_drift.utils.io import read_parquet
from mlops_drift.utils.logging import get_logger

log = get_logger("validate")


def validate_champion(cfg: Config | None = None, tracking_uri: str | None = None) -> bool:
    """True iff the live champion's **in-distribution** F1 (reference test) >= f1_floor.

    Soundness gate, independent of drift: a model that can't clear the floor on its own
    in-distribution holdout should never be promoted/merged.
    """
    cfg = cfg or get_config()
    setup_mlflow(cfg, tracking_uri=tracking_uri)
    name = cfg.mlflow.registered_model
    model = mlflow.sklearn.load_model(f"models:/{name}@{CHAMPION}")

    path, _ = ingest(cfg)
    df = read_parquet(path)
    feats = select_feature_cols(df, target_col=cfg.data.target_col, time_col=cfg.data.time_col)
    _, test = time_aware_split(
        df, cfg.split.train_frac, time_col=cfg.data.time_col, period="reference"
    )
    x, y = test[feats], test[cfg.data.target_col].to_numpy()
    m = evaluate_classification(y, model.predict(x), model.predict_proba(x)[:, 1])
    floor = cfg.thresholds.validation.f1_floor
    ok = float(m["f1"]) >= floor
    log.info("validate.champion", f1=round(float(m["f1"]), 4), floor=floor, passed=ok)
    return ok


def main() -> int:
    ok = validate_champion()
    print(f"validation {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
