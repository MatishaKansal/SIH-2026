"""Score semantics and immutable baseline-result recording for Module 2A."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def scores_from_logits(logits: np.ndarray | torch.Tensor) -> dict[str, float | None]:
    """Return native AASIST and project-standard uncalibrated score values.

    The official AASIST protocol maps bona-fide to class 1 and spoof to class
    0. Its evaluator saves ``output[:, 1]``. We retain that exact native score
    as ``raw_score``. The score used by this project is the logit margin
    ``logit[spoof] - logit[bonafide]``: higher means stronger spoof evidence.
    Neither is a calibrated probability.
    """
    values = np.asarray(logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits, dtype=np.float64)
    values = values.reshape(-1)
    if values.size != 2 or not np.isfinite(values).all():
        raise ValueError("AASIST logits must be two finite values ordered [spoof, bona_fide]")
    spoof_logit, bona_fide_logit = values
    return {
        "raw_score": float(bona_fide_logit),
        "spoof_score": float(spoof_logit - bona_fide_logit),
        "spoof_probability": None,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model_metadata(
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    device: str,
    sample_rate: int,
    target_samples: int,
) -> Path:
    """Record the exact immutable model setup used for baseline results."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path).resolve()
    payload = {
        "module": "2A",
        "stage": "baseline",
        "model_name": "AASIST",
        "model_version": "pretrained",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "sample_rate": sample_rate,
        "model_input_samples": target_samples,
        "model_input_duration_seconds": target_samples / sample_rate,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "score_direction": {
            "raw_score": "official native class-1 (bona_fide) logit; higher is native bona-fide evidence",
            "spoof_score": "logit[class 0 spoof] - logit[class 1 bona_fide]; higher is more spoof evidence",
        },
        "score_transformation": "spoof_score = logits[0] - logits[1]",
        "spoof_probability": "not calibrated; always null",
    }
    path = directory / "model_metadata.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def save_baseline_result(output_dir: str | Path, result: dict[str, Any]) -> Path:
    """Persist one JSON result and append its row to the baseline CSV."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    stored = {**result, "timestamp": timestamp}
    source_name = Path(str(result["input_file"])).stem or "audio"
    safe_timestamp = timestamp.replace(":", "").replace("+", "_").replace("-", "")
    json_path = directory / f"{source_name}_{safe_timestamp}.json"
    json_path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")

    csv_path = directory / "baseline_results.csv"
    fields = ["timestamp", "input_file", "duration", "raw_score", "spoof_score", "spoof_probability", "label", "model", "stage"]
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "input_file": stored["input_file"],
                "duration": stored["duration_seconds"],
                "raw_score": stored["raw_score"],
                "spoof_score": stored["spoof_score"],
                "spoof_probability": "",
                "label": stored["label"] or "",
                "model": stored["model"],
                "stage": stored["stage"],
            }
        )
    return json_path
