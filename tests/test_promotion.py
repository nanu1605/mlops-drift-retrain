"""Phase 5: champion/challenger promotion gate. Tmp sqlite store per test."""

from __future__ import annotations

from mlflow.tracking import MlflowClient

from mlops_drift.config import get_config
from mlops_drift.promotion.champion_challenger import (
    PromotionDecision,
    decide_promotion,
    evaluate_pair,
    run_promotion,
)
from mlops_drift.training.train import run_training


def _uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/mlflow.db"


def test_strong_challenger_promoted(tmp_path):
    cfg = get_config()
    uri = _uri(tmp_path)
    run_training(cfg, tracking_uri=uri)  # v1 champion + challenger (reference-only)
    run_training(
        cfg, tracking_uri=uri, periods=("reference", "drift")
    )  # v2 challenger learns drift

    pair = evaluate_pair(cfg, tracking_uri=uri)
    assert pair["same_version"] is False
    assert pair["challenger"]["f1"] - pair["champion"]["f1"] >= cfg.thresholds.promotion.f1_margin

    decision = run_promotion(cfg, tracking_uri=uri)
    assert decision.promote is True

    client = MlflowClient()
    champ = client.get_model_version_by_alias(cfg.mlflow.registered_model, "champion")
    assert str(champ.version) == decision.challenger_version  # @champion moved


def test_weak_challenger_rejected(tmp_path):
    cfg = get_config()
    uri = _uri(tmp_path)
    run_training(cfg, tracking_uri=uri)  # v1
    run_training(cfg, tracking_uri=uri)  # v2 reference-only ≈ champion → tie

    decision = run_promotion(cfg, tracking_uri=uri)
    assert decision.promote is False

    client = MlflowClient()
    champ = client.get_model_version_by_alias(cfg.mlflow.registered_model, "champion")
    assert str(champ.version) == "1"  # @champion unchanged


def test_decide_promotion_rules():
    cfg = get_config()
    margin = cfg.thresholds.promotion.f1_margin
    floor = cfg.thresholds.validation.f1_floor

    def pair(champ_f1, chal_f1):
        return {
            "same_version": False,
            "champion_version": "1",
            "challenger_version": "2",
            "champion": {"f1": champ_f1},
            "challenger": {"f1": chal_f1},
        }

    # clears margin + floor → promote
    assert decide_promotion(pair(0.60, 0.60 + margin + 0.05), cfg).promote is True
    # tie → reject
    assert decide_promotion(pair(0.60, 0.60), cfg).promote is False
    # regression → reject
    assert decide_promotion(pair(0.60, 0.55), cfg).promote is False
    # beats champion but below floor → reject
    assert decide_promotion(pair(floor - 0.20, floor - 0.05), cfg).promote is False
    # same version → reject
    d = decide_promotion(
        {"same_version": True, "champion_version": "1", "challenger_version": "1"}, cfg
    )
    assert isinstance(d, PromotionDecision) and d.promote is False
