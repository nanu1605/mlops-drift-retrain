"""Champion/challenger promotion gate.

Scores the live ``@champion`` and ``@challenger`` on a **common holdout = the drift period**
(recent labeled production-like traffic) and promotes the challenger **only if it wins**:

    promote iff (challenger_f1 - champion_f1 >= promotion.f1_margin)
            and  challenger_f1 >= validation.f1_floor

Never on tie, regression, or below-floor. Promotion is alias-only
(``set_registered_model_alias`` — never ``transition_model_version_stage``); the old champion
version is retained (rollback-able), just no longer aliased.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops_drift.config import Config, get_config
from mlops_drift.data.features import select_feature_cols
from mlops_drift.data.ingest import ingest
from mlops_drift.training.evaluate import evaluate_classification
from mlops_drift.training.mlflow_utils import CHALLENGER, CHAMPION, setup_mlflow
from mlops_drift.utils.io import read_parquet
from mlops_drift.utils.logging import get_logger

log = get_logger("promotion")


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    champion_version: str | None
    challenger_version: str | None
    champion_f1: float
    challenger_f1: float
    f1_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def holdout_frame(cfg: Config, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Common holdout = the drift period (features + label)."""
    if df is None:
        path, _ = ingest(cfg)
        df = read_parquet(path)
    feats = select_feature_cols(df, target_col=cfg.data.target_col, time_col=cfg.data.time_col)
    drift = df[df["period"] == "drift"]
    return drift[feats], drift[cfg.data.target_col]


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except Exception:
        return None


def _score(model, x: pd.DataFrame, y) -> dict:
    proba = model.predict_proba(x)[:, 1]
    return evaluate_classification(y.to_numpy(), model.predict(x), proba)


def evaluate_pair(
    cfg: Config, tracking_uri: str | None = None, df: pd.DataFrame | None = None
) -> dict:
    """Load both aliased models and score them on the drift holdout."""
    setup_mlflow(cfg, tracking_uri=tracking_uri)
    name = cfg.mlflow.registered_model
    client = MlflowClient()

    champ_v = _alias_version(client, name, CHAMPION)
    chal_v = _alias_version(client, name, CHALLENGER)
    if champ_v is None or chal_v is None:
        return {
            "same_version": champ_v == chal_v,
            "champion_version": champ_v,
            "challenger_version": chal_v,
        }
    if champ_v == chal_v:
        return {"same_version": True, "champion_version": champ_v, "challenger_version": chal_v}

    x, y = holdout_frame(cfg, df=df)
    champ = mlflow.sklearn.load_model(f"models:/{name}@{CHAMPION}")
    chal = mlflow.sklearn.load_model(f"models:/{name}@{CHALLENGER}")
    return {
        "same_version": False,
        "champion_version": champ_v,
        "challenger_version": chal_v,
        "champion": _score(champ, x, y),
        "challenger": _score(chal, x, y),
    }


def decide_promotion(pair: dict, cfg: Config) -> PromotionDecision:
    """Apply the margin + floor gate. Never promote on tie/regression/below-floor."""
    if pair.get("same_version"):
        return PromotionDecision(
            False,
            "same_version",
            pair.get("champion_version"),
            pair.get("challenger_version"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    champ_f1 = float(pair["champion"]["f1"])
    chal_f1 = float(pair["challenger"]["f1"])
    delta = chal_f1 - champ_f1
    margin = cfg.thresholds.promotion.f1_margin
    floor = cfg.thresholds.validation.f1_floor

    if chal_f1 < floor:
        reason = f"below_floor (challenger_f1={chal_f1:.4f} < {floor})"
        promote = False
    elif delta < margin:
        reason = f"insufficient_margin (delta={delta:.4f} < {margin})"
        promote = False
    else:
        reason = f"promoted (delta={delta:.4f} >= {margin}, f1={chal_f1:.4f} >= {floor})"
        promote = True

    return PromotionDecision(
        promote,
        reason,
        pair["champion_version"],
        pair["challenger_version"],
        champ_f1,
        chal_f1,
        delta,
    )


def run_promotion(
    cfg: Config | None = None, tracking_uri: str | None = None, df: pd.DataFrame | None = None
) -> PromotionDecision:
    """Evaluate champion vs challenger and promote the challenger iff it wins."""
    cfg = cfg or get_config()
    pair = evaluate_pair(cfg, tracking_uri=tracking_uri, df=df)
    decision = decide_promotion(pair, cfg)

    if decision.promote:
        client = MlflowClient()
        client.set_registered_model_alias(
            cfg.mlflow.registered_model, CHAMPION, decision.challenger_version
        )
        log.info("promotion.decision", **decision.to_dict())
    else:
        log.info("promotion.rejected", **decision.to_dict())
    return decision
