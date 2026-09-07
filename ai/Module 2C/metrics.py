"""Evaluation metrics with explicit uncalibrated score semantics."""
from __future__ import annotations
import numpy as np

def equal_error_rate(labels, scores) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr; index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)

def evaluate_scores(labels, logits) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    labels, logits = np.asarray(labels, dtype=int), np.asarray(logits, dtype=float)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    result = {"accuracy": float(accuracy_score(labels, predictions)), "precision": float(precision_score(labels, predictions, zero_division=0)), "recall": float(recall_score(labels, predictions, zero_division=0)), "f1": float(f1_score(labels, predictions, zero_division=0)), "real_recall": float(matrix[0, 0] / max(matrix[0].sum(), 1)), "spoof_recall": float(matrix[1, 1] / max(matrix[1].sum(), 1)), "false_positive_rate": float(matrix[0, 1] / max(matrix[0].sum(), 1)), "false_negative_rate": float(matrix[1, 0] / max(matrix[1].sum(), 1)), "confusion_matrix": matrix.tolist(), "uncalibrated_spoof_probability": "sigmoid(logit), not calibrated"}
    result["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    result["eer"] = equal_error_rate(labels, probabilities) if len(np.unique(labels)) == 2 else None
    return result
