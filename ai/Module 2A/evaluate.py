"""Evaluate pretrained or fine-tuned AASIST only on a named held-out split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration import load_calibrator
from config import CHECKPOINT_PATH
from dataset import AudioManifestDataset, read_manifest
from metrics import binary_metrics
from model_utils import checkpoint_model_sha256, load_aasist_model


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    criterion: torch.nn.Module | None = None,
    calibrator: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run no-grad inference and return window predictions plus metrics."""
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.inference_mode():
        for audio, labels, sample_ids in loader:
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, logits = model(audio)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()))
            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            scores = logits_np[:, 0] - logits_np[:, 1]
            probabilities = calibrator.predict_proba(scores) if calibrator else None
            for index, sample_id in enumerate(sample_ids):
                rows.append({
                    "sample_id": sample_id, "label": int(labels_np[index]),
                    "raw_score": float(logits_np[index, 1]), "spoof_score": float(scores[index]),
                    "spoof_probability": None if probabilities is None else float(probabilities[index]),
                })
    labels = np.array([row["label"] for row in rows])
    scores = np.array([row["spoof_score"] for row in rows])
    threshold_scores = np.array([row["spoof_probability"] for row in rows]) if calibrator else scores
    threshold = calibrator.metadata.get("operating_threshold") if calibrator else None
    metrics = binary_metrics(labels, threshold_scores, threshold=threshold, probabilities=threshold_scores if calibrator else None)
    metrics["mean_loss"] = float(np.mean(losses)) if losses else None
    metrics["threshold_note"] = "No operating threshold supplied; accuracy/precision/recall/F1 require a calibration-selected threshold."
    return rows, metrics


def write_predictions(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "raw_score", "spoof_score", "spoof_probability"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AASIST on a frozen manifest split")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "validation", "calibration", "test"])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--max-samples-per-class", type=int, help="Bounded diagnostic mode; never use for final metrics")
    args = parser.parse_args()

    model, device = load_aasist_model(args.checkpoint, device=args.device)
    model.eval()
    calibration = load_calibrator(args.calibrator, expected_checkpoint_sha256=checkpoint_model_sha256(args.checkpoint)) if args.calibrator else None
    records = read_manifest(args.manifest, split=args.split)
    if args.max_samples_per_class is not None:
        if args.max_samples_per_class <= 0:
            raise ValueError("--max-samples-per-class must be positive")
        bounded = []
        for label in (0, 1):
            bounded.extend([record for record in records if record.label == label][:args.max_samples_per_class])
        records = bounded
    loader = DataLoader(AudioManifestDataset(records), batch_size=args.batch_size, num_workers=args.num_workers)
    rows, metrics = evaluate_model(model, loader, device, calibrator=calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output_dir / f"{args.split}_predictions.csv", rows)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_model_sha256(args.checkpoint),
        "split": args.split, "device": str(device), "calibrator": str(args.calibrator) if args.calibrator else None,
        "diagnostic_sample_limit_per_class": args.max_samples_per_class,
        "metrics": metrics,
    }
    (args.output_dir / f"{args.split}_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
