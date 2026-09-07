"""Binary anti-spoofing metrics with a fixed convention: spoof is label 1."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def select_youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select a spoof-positive operating threshold from calibration data only."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    index = int(np.argmax((tpr - fpr)[finite]))
    return float(thresholds[finite][index])


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) != 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[index] + fpr[index]) / 2)


def score_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def binary_metrics(
    labels: np.ndarray,
    spoof_scores: np.ndarray,
    *,
    threshold: float | None = None,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate discrimination, threshold, and optional calibration metrics."""
    y = np.asarray(labels, dtype=int)
    scores = np.asarray(spoof_scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must contain non-empty binary values: bona-fide=0, spoof=1")
    classes = np.unique(y)
    result: dict[str, Any] = {
        "sample_count": int(len(y)),
        "bona_fide_count": int(np.sum(y == 0)),
        "spoof_count": int(np.sum(y == 1)),
        "spoof_fraction": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(classes) == 2 else None,
        "eer": equal_error_rate(y, scores),
        "score_distribution": {
            "all": score_distribution(scores),
            "bona_fide": score_distribution(scores[y == 0]) if np.any(y == 0) else None,
            "spoof": score_distribution(scores[y == 1]) if np.any(y == 1) else None,
        },
    }
    if threshold is not None:
        predicted = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predicted, average="binary", zero_division=0
        )
        matrix = confusion_matrix(y, predicted, labels=[0, 1])
        result.update(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y, predicted)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": matrix.astype(int).tolist(),
                "false_rejection_rate": float(matrix[0, 1] / matrix[0].sum()) if matrix[0].sum() else None,
                "false_acceptance_rate": float(matrix[1, 0] / matrix[1].sum()) if matrix[1].sum() else None,
            }
        )
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        if len(p) != len(y) or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must be in [0, 1] and match labels")
        result["brier_score"] = float(brier_score_loss(y, p))
        if len(y) >= 20 and len(classes) == 2:
            observed, predicted = calibration_curve(y, p, n_bins=min(10, len(y) // 2), strategy="quantile")
            result["reliability_curve"] = [
                {"mean_predicted_probability": float(x), "observed_spoof_fraction": float(z)}
                for x, z in zip(predicted, observed)
            ]
        else:
            result["reliability_curve"] = None
    return result
