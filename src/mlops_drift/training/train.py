"""Reproducible training pipeline with MLflow tracking + registry.

ingest → validate → time-aware split (reference period) → fit sklearn Pipeline
(median-impute → RandomForest) → evaluate (in-distribution test + drift preview) →
log everything to MLflow (params, metrics, seed, git SHA, DVC data hash, artifacts incl.
the reference window + eval plots) → register the model version with @challenger (and
@champion on the first run).

The registered model is a single sklearn Pipeline operating on the raw ``f*`` columns, so
serving loads one artifact and predicts on raw features.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline

from mlops_drift.config import Config, get_config
from mlops_drift.data import features as feat
from mlops_drift.data import validation
from mlops_drift.data.ingest import ingest
from mlops_drift.data.split import time_aware_split
from mlops_drift.training import register as register_mod
from mlops_drift.training.evaluate import evaluate_classification
from mlops_drift.training.mlflow_utils import data_md5, setup_mlflow
from mlops_drift.utils.io import ensure_dir, git_sha, read_parquet, write_json
from mlops_drift.utils.logging import get_logger
from mlops_drift.utils.seeds import set_seed

log = get_logger("train")

MODEL_ARTIFACT = "model"
METRICS_FILE = "metrics/train_metrics.json"  # DVC-tracked metrics output


def build_pipeline(cfg: Config, seed: int) -> Pipeline:
    """sklearn Pipeline: median-impute then RandomForest. Operates on raw f-columns."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(random_state=seed, **cfg.model.params)),
        ]
    )


def _eval_plots(pipe: Pipeline, x_test, y_test, outdir) -> list:
    paths = []
    cm = ConfusionMatrixDisplay.from_estimator(pipe, x_test, y_test)
    cm_path = outdir / "confusion_matrix.png"
    cm.figure_.savefig(cm_path, dpi=110, bbox_inches="tight")
    plt.close(cm.figure_)
    paths.append(cm_path)

    pr = PrecisionRecallDisplay.from_estimator(pipe, x_test, y_test)
    pr_path = outdir / "pr_curve.png"
    pr.figure_.savefig(pr_path, dpi=110, bbox_inches="tight")
    plt.close(pr.figure_)
    paths.append(pr_path)
    return paths


def run_training(cfg: Config | None = None, tracking_uri: str | None = None) -> dict:
    cfg = cfg or get_config()
    seed = set_seed(cfg.seed)
    target, time_col = cfg.data.target_col, cfg.data.time_col
    setup_mlflow(cfg, tracking_uri=tracking_uri)

    # --- data ---
    path, source = ingest(cfg)
    df = read_parquet(path)
    feature_cols = feat.select_feature_cols(df, target_col=target, time_col=time_col)
    validation.validate(df, feature_cols, require_label=True)

    train, test = time_aware_split(df, cfg.split.train_frac, time_col=time_col, period="reference")
    drift = df[df["period"] == "drift"]

    x_train, y_train = train[feature_cols], train[target].to_numpy()
    x_test, y_test = test[feature_cols], test[target].to_numpy()
    x_drift, y_drift = drift[feature_cols], drift[target].to_numpy()

    # --- fit + evaluate ---
    pipe = build_pipeline(cfg, seed)
    pipe.fit(x_train, y_train)
    test_m = evaluate_classification(y_test, pipe.predict(x_test), pipe.predict_proba(x_test)[:, 1])
    drift_m = evaluate_classification(
        y_drift, pipe.predict(x_drift), pipe.predict_proba(x_drift)[:, 1]
    )

    # --- reference window (raw f-cols + label), persisted + logged for Phase 4 ---
    ref_n = min(cfg.monitoring.reference_sample, len(train))
    reference = (
        train[feature_cols + [target]].sample(n=ref_n, random_state=seed).reset_index(drop=True)
    )
    ref_path = ensure_dir(cfg.reference_dir) / "reference.parquet"
    reference.to_parquet(ref_path, index=False)

    metrics = {
        "f1": test_m["f1"],
        "pr_auc": test_m["pr_auc"],
        "precision": test_m["precision"],
        "recall": test_m["recall"],
        "accuracy": test_m["accuracy"],
        "drift_preview_f1": drift_m["f1"],
        "drift_preview_pr_auc": drift_m["pr_auc"],
    }

    sha = git_sha()
    md5 = data_md5(path)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "seed": seed,
                "train_frac": cfg.split.train_frac,
                "model_family": cfg.model.family,
                "n_features": len(feature_cols),
                **{f"rf_{k}": v for k, v in cfg.model.params.items()},
            }
        )
        mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})  # skip NaN
        mlflow.set_tags(
            {
                "git_sha": sha,
                "dvc_data_md5": md5,
                "data_source": source,
                "feature_cols": json.dumps(feature_cols),
            }
        )

        signature = infer_signature(x_test, pipe.predict(x_test))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path=MODEL_ARTIFACT,
            signature=signature,
            input_example=x_test.head(3),
        )

        artdir = ensure_dir(cfg.artifacts_dir / "train_tmp")
        schema_path = artdir / "feature_schema.json"
        write_json({"feature_cols": feature_cols, "target": target}, schema_path)
        mlflow.log_artifact(str(schema_path))
        mlflow.log_artifact(str(ref_path))
        for p in _eval_plots(pipe, x_test, y_test, artdir):
            mlflow.log_artifact(str(p))

    # --- register with aliases ---
    name, version = register_mod.register_model_version(run_id, cfg, model_artifact=MODEL_ARTIFACT)

    result = {
        "run_id": run_id,
        "model_name": name,
        "version": version,
        "model_uri": f"models:/{name}@champion",
        "metrics": metrics,
        "git_sha": sha,
        "dvc_data_md5": md5,
        "data_source": source,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "reference_window_rows": int(len(reference)),
    }

    # DVC metrics file (committed; powers `dvc repro` / `dvc metrics show`)
    write_json({"metrics": metrics, "version": version}, cfg.abspath(METRICS_FILE))

    log.info(
        "train.done",
        run_id=run_id,
        version=version,
        f1=round(metrics["f1"], 4),
        pr_auc=round(metrics["pr_auc"], 4),
        drift_f1=round(metrics["drift_preview_f1"], 4),
    )
    return result


def main() -> int:
    r = run_training()
    m = r["metrics"]
    print(
        f"trained v{r['version']} run={r['run_id'][:8]} "
        f"f1={m['f1']:.4f} pr_auc={m['pr_auc']:.4f} -> {r['model_uri']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
