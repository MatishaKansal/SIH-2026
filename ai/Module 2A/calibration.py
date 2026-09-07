"""Version-locked Platt calibration for AASIST spoof-direction scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from metrics import binary_metrics, select_youden_threshold

ARTIFACT_VERSION = 1
MIN_SAMPLES_PER_CLASS = 20


@dataclass
class PlattCalibrator:
    model: LogisticRegression
    metadata: dict[str, Any]

    def predict_proba(self, spoof_scores: np.ndarray | list[float]) -> np.ndarray:
        scores = np.asarray(spoof_scores, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(scores)[:, 1]


def fit_platt_calibrator(
    spoof_scores: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    *,
    model_checkpoint_sha256: str,
    model_identifier: str,
) -> tuple[PlattCalibrator, dict[str, Any]]:
    """Fit calibrated P(spoof|window) from a dedicated, labeled calibration set.

    No class reweighting is used here: probability calibration must preserve the
    calibration set's documented class prior. Class balance is recorded instead.
    """
    scores = np.asarray(spoof_scores, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(scores) != len(y) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and match labels")
    counts = {label: int(np.sum(y == label)) for label in (0, 1)}
    if not np.isin(y, [0, 1]).all() or min(counts.values()) < MIN_SAMPLES_PER_CLASS:
        raise ValueError(
            f"Calibration needs at least {MIN_SAMPLES_PER_CLASS} bona-fide and spoof samples; got {counts}."
        )
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(scores.reshape(-1, 1), y)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "method": "Platt scaling (one-feature logistic regression)",
        "probability_meaning": "P(spoof | window), calibrated to the calibration-set class prior",
        "score_input": "AASIST spoof_score = logit[0] - logit[1]",
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "model_identifier": model_identifier,
        "calibration_sample_count": int(len(y)),
        "calibration_class_counts": {"bona_fide": counts[0], "spoof": counts[1]},
        "class_weight": None,
        "warning": "Fit-set metrics are diagnostics only; use an untouched test split for final estimates.",
    }
    calibrator = PlattCalibrator(model=model, metadata=metadata)
    probabilities = calibrator.predict_proba(scores)
    threshold = select_youden_threshold(y, probabilities)
    calibrator.metadata["operating_threshold"] = threshold
    calibrator.metadata["threshold_selection"] = "Youden J selected on calibration split only"
    metrics = binary_metrics(y, probabilities, threshold=threshold, probabilities=probabilities)
    metrics["threshold_selection"] = "Youden J selected on calibration split only"
    return calibrator, {"metadata": metadata, "fit_diagnostics": metrics, "operating_threshold": threshold}


def save_calibrator(calibrator: PlattCalibrator, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"artifact_version": ARTIFACT_VERSION, "model": calibrator.model, "metadata": calibrator.metadata}, destination)
    return destination


def load_calibrator(path: str | Path, *, expected_checkpoint_sha256: str | None = None) -> PlattCalibrator:
    artifact = joblib.load(path)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported calibration artifact version")
    metadata = artifact["metadata"]
    if expected_checkpoint_sha256 and metadata.get("model_checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("calibration artifact was fitted for a different AASIST checkpoint")
    return PlattCalibrator(model=artifact["model"], metadata=metadata)
