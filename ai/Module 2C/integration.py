"""Run Module 2B over an existing Module 1 manifest without running VAD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .inference import SpectralSpoofDetector
except ImportError:
    from inference import SpectralSpoofDetector


def run_manifest(
    manifest_path: str | Path,
    checkpoint: str | Path | None = None,
    spectral_type: str = "logmel",
) -> list[dict]:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text())

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Module 1 manifest must contain a 'segments' list")

    detector = SpectralSpoofDetector(
        checkpoint=checkpoint,
        spectral_type=spectral_type,
    )

    results = [
        detector.predict_manifest_record(manifest_path, record)
        for record in segments
    ]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Module 2B on an existing Module 1 manifest."
    )

    parser.add_argument("manifest")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--spectral-type",
        choices=["stft", "logmel", "lfcc"],
        default="logmel",
    )
    parser.add_argument(
        "--output",
        default="output/module1_integration.json",
    )

    args = parser.parse_args()

    results = run_manifest(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        spectral_type=args.spectral_type,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Wrote {len(results)} chunk results to {output_path}")


if __name__ == "__main__":
    main()