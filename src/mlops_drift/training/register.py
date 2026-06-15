"""Register a trained model to the MLflow Model Registry using aliases (not stages).

Every registration sets ``@challenger`` on the new version. The first ever version (no
``@champion`` yet) also gets ``@champion`` so serving has something to resolve. Promotion
of later challengers is the controller's job (Phase 5).
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from mlops_drift.config import Config
from mlops_drift.training.mlflow_utils import CHALLENGER, CHAMPION
from mlops_drift.utils.logging import get_logger

log = get_logger("register")


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except Exception:
        return None


def register_model_version(
    run_id: str, cfg: Config, model_artifact: str = "model"
) -> tuple[str, str]:
    """Register ``runs:/<run_id>/<model_artifact>`` and set aliases.

    Returns ``(registered_model_name, version)``. Assumes ``setup_mlflow`` already ran.
    """
    name = cfg.mlflow.registered_model
    client = MlflowClient()

    mv = mlflow.register_model(f"runs:/{run_id}/{model_artifact}", name)
    version = mv.version

    client.set_registered_model_alias(name, CHALLENGER, version)
    first_champion = _alias_version(client, name, CHAMPION) is None
    if first_champion:
        client.set_registered_model_alias(name, CHAMPION, version)

    log.info(
        "register.done",
        model=name,
        version=version,
        challenger=True,
        set_champion=first_champion,
    )
    return name, version
