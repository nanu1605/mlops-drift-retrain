"""Classification metrics for imbalanced binary problems.

F1 and PR-AUC (average precision) are the primary metrics — accuracy alone is misleading
when the positive class is a minority. Confusion matrix included for transparency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Sequence[float] | None = None,
) -> dict[str, float | list[list[int]]]:
    """Return precision, recall, f1, pr_auc, accuracy, and the confusion matrix.

    ``y_proba`` is P(class=1); required for pr_auc (else NaN).
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    metrics: dict[str, float | list[list[int]]] = {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "accuracy": float((yt == yp).mean()),
    }
    if y_proba is not None and len(np.unique(yt)) > 1:
        metrics["pr_auc"] = float(average_precision_score(yt, np.asarray(y_proba)))
    else:
        metrics["pr_auc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(yt, yp, labels=[0, 1]).tolist()
    return metrics


def beats(
    model_metrics: dict,
    trivial_metrics: dict,
    keys: tuple[str, ...] = ("f1", "pr_auc"),
) -> bool:
    """True iff ``model_metrics`` strictly exceeds ``trivial_metrics`` on every key."""
    return all(float(model_metrics[k]) > float(trivial_metrics[k]) for k in keys)
