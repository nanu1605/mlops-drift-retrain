"""MLflow setup + small reproducibility helpers.

Pure-local: a sqlite-file tracking/registry backend (no server needed). The Model
Registry requires a DB backend — this module enforces that so alias calls never hit an
unsupported file store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlflow
import yaml

from mlops_drift.config import Config

CHAMPION = "champion"
CHALLENGER = "challenger"


def setup_mlflow(cfg: Config, tracking_uri: str | None = None) -> str:
    """Point MLflow tracking + registry at the (sqlite) backend and select the experiment.

    ``tracking_uri`` overrides config (tests pass a tmp sqlite path). Returns the URI used.
    """
    uri = tracking_uri or cfg.resolved_tracking_uri()
    if not uri.startswith("sqlite:") and "://" not in uri:
        raise ValueError(f"tracking_uri must be a DB/HTTP URI for the registry, got {uri!r}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    return uri


def data_md5(path: Path | str) -> str:
    """Reproducibility hash for the dataset: reuse the DVC ``.dvc`` pointer md5 if present,
    else hash the file bytes."""
    p = Path(path)
    pointer = p.with_suffix(p.suffix + ".dvc")
    if pointer.exists():
        meta = yaml.safe_load(pointer.read_text())
        outs = meta.get("outs", [])
        if outs and "md5" in outs[0]:
            return str(outs[0]["md5"])
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
