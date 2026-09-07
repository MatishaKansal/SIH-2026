"""Fit a version-locked Platt calibrator from a dedicated score CSV."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from calibration import fit_platt_calibrator, save_calibrator
from config import CHECKPOINT_PATH
from model_utils import checkpoint_model_sha256


def load_score_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"label", "spoof_score"}.issubset(reader.fieldnames):
            raise ValueError("score CSV requires label and spoof_score columns")
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row["spoof_score"]))
    return np.asarray(scores), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate P(spoof|window) from a held-out calibration split")
    parser.add_argument("--scores", required=True, type=Path, help="Predictions from evaluate.py on split=calibration only")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="AASIST")
    args = parser.parse_args()

    scores, labels = load_score_csv(args.scores)
    checkpoint_hash = checkpoint_model_sha256(args.checkpoint)
    calibrator, report = fit_platt_calibrator(
        scores, labels, model_checkpoint_sha256=checkpoint_hash, model_identifier=args.model_id
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = save_calibrator(calibrator, args.output_dir / "calibration_model.joblib")
    report.update({
        "created_at": datetime.now(timezone.utc).isoformat(), "scores_file": str(args.scores.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "calibration_model": str(artifact.resolve()),
    })
    (args.output_dir / "calibration_metrics.json").write_text(json.dumps(report["fit_diagnostics"], indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "thresholds.json").write_text(json.dumps({"spoof_probability_threshold": report["operating_threshold"], "selection": report["fit_diagnostics"]["threshold_selection"]}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "calibration_curve.json").write_text(json.dumps(report["fit_diagnostics"].get("reliability_curve"), indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "score_distribution.json").write_text(json.dumps(report["fit_diagnostics"]["score_distribution"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
