"""Produce a transparent pretrained-vs-fine-tuned held-out metrics comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ["roc_auc", "eer", "accuracy", "precision", "recall", "f1", "brier_score", "false_acceptance_rate", "false_rejection_rate"]


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two AASIST evaluations performed on the same frozen test manifest")
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--finetuned", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    before, after = read_metrics(args.pretrained), read_metrics(args.finetuned)
    report = {
        "comparison": "pretrained versus fine-tuned AASIST; only valid when both files used the same held-out test manifest and operating threshold policy",
        "pretrained": before, "finetuned": after,
        "delta_finetuned_minus_pretrained": {
            key: (after.get(key) - before.get(key) if isinstance(after.get(key), (int, float)) and isinstance(before.get(key), (int, float)) else None)
            for key in METRICS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
