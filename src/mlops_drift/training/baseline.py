"""Phase 1 baseline runner (no MLflow — that arrives in Phase 2).

ingest → validate → time-aware split (reference period only) → fit RandomForest + a
majority-class trivial baseline + a logistic reference → evaluate all on the in-distribution
test → also score the RF on the held-out drift period (degradation preview, not a gate) →
write artifacts/baseline_metrics.json.

The honest baseline F1 ("X") is the in-distribution RandomForest F1.
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlops_drift.config import Config, get_config
from mlops_drift.data import features as feat
from mlops_drift.data import validation
from mlops_drift.data.ingest import ingest
from mlops_drift.data.split import time_aware_split
from mlops_drift.training.evaluate import beats, evaluate_classification
from mlops_drift.utils.io import read_parquet, write_json
from mlops_drift.utils.logging import get_logger
from mlops_drift.utils.seeds import set_seed

log = get_logger("baseline")


def _proba1(model, x) -> pd.Series | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return None


def run_baseline(cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    seed = set_seed(cfg.seed)
    target = cfg.data.target_col
    time_col = cfg.data.time_col

    # --- data ---
    path, source = ingest(cfg)
    df = read_parquet(path)
    feature_cols = feat.select_feature_cols(df, target_col=target, time_col=time_col)
    validation.validate(df, feature_cols, require_label=True)
    log.info("baseline.data", source=source, rows=len(df), n_features=len(feature_cols))

    # --- honest in-distribution split: reference period only, time-aware ---
    train, test = time_aware_split(df, cfg.split.train_frac, time_col=time_col, period="reference")
    drift = df[df["period"] == "drift"]  # held out, degradation preview only

    pipe = feat.FeaturePipeline(feature_cols).fit(train)
    x_train, y_train = pipe.transform(train), train[target].to_numpy()
    x_test, y_test = pipe.transform(test), test[target].to_numpy()
    x_drift, y_drift = pipe.transform(drift), drift[target].to_numpy()

    # --- models ---
    rf = RandomForestClassifier(random_state=seed, **cfg.model.params)
    rf.fit(x_train, y_train)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    logit.fit(x_train, y_train)

    # --- evaluate on in-distribution test ---
    rf_m = evaluate_classification(y_test, rf.predict(x_test), _proba1(rf, x_test))
    dummy_m = evaluate_classification(y_test, dummy.predict(x_test), _proba1(dummy, x_test))
    logit_m = evaluate_classification(y_test, logit.predict(x_test), _proba1(logit, x_test))

    # --- degradation preview on the held-out drift period (NOT a gate) ---
    rf_drift_m = evaluate_classification(y_drift, rf.predict(x_drift), _proba1(rf, x_drift))

    gate_pass = beats(rf_m, dummy_m, keys=("f1", "pr_auc"))

    result = {
        "seed": seed,
        "data_source": source,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "split": {
            "period": "reference",
            "train_frac": cfg.split.train_frac,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_drift_holdout": int(len(drift)),
        },
        "metrics": {
            "random_forest_in_distribution": rf_m,
            "majority_trivial": dummy_m,
            "logistic_reference": logit_m,
            "random_forest_drift_preview": rf_drift_m,
        },
        "deltas_vs_majority": {
            "f1": rf_m["f1"] - dummy_m["f1"],
            "pr_auc": rf_m["pr_auc"] - dummy_m["pr_auc"],
        },
        "deltas_vs_logistic": {
            "f1": rf_m["f1"] - logit_m["f1"],
            "pr_auc": rf_m["pr_auc"] - logit_m["pr_auc"],
        },
        "beats_trivial_on_f1_and_pr_auc": gate_pass,
    }

    out = cfg.artifacts_dir / "baseline_metrics.json"
    write_json(result, out)
    log.info(
        "baseline.done",
        rf_f1=round(rf_m["f1"], 4),
        rf_pr_auc=round(rf_m["pr_auc"], 4),
        drift_f1=round(rf_drift_m["f1"], 4),
        beats_trivial=gate_pass,
        out=str(out),
    )
    return result


def main() -> int:
    r = run_baseline()
    m = r["metrics"]["random_forest_in_distribution"]
    d = r["metrics"]["random_forest_drift_preview"]
    print(
        f"baseline RF in-dist: f1={m['f1']:.4f} pr_auc={m['pr_auc']:.4f} | "
        f"drift-preview f1={d['f1']:.4f} | beats_trivial={r['beats_trivial_on_f1_and_pr_auc']}"
    )
    return 0 if r["beats_trivial_on_f1_and_pr_auc"] else 1


if __name__ == "__main__":
    sys.exit(main())
