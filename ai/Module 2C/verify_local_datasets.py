"""Verify the three local Module 2B dataset roots without preparing chunks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf

EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}

def inspect(dataset: str, root: Path) -> dict:
    audio_root = root / "audio" if (root / "audio").is_dir() else root
    files = sorted(path for path in audio_root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    corrupt = []
    durations = []
    rates = {}
    channels = {}
    for path in files:
        try:
            info = sf.info(str(path))
            audio, rate = sf.read(str(path), always_2d=True, dtype="float32")
            if not np.isfinite(audio).all():
                raise ValueError("non-finite samples")
            durations.append(info.frames / info.samplerate)
            rates[str(rate)] = rates.get(str(rate), 0) + 1
            channels[str(audio.shape[1])] = channels.get(str(audio.shape[1]), 0) + 1
        except Exception as exc:
            corrupt.append({"path": str(path), "error": str(exc)})
    metadata = [name for name in ("metadata.csv", "metadata.jsonl") if (root / name).is_file()]
    return {
        "dataset": dataset,
        "audio_file_count": len(files),
        "metadata_present": metadata,
        "total_duration_seconds": sum(durations),
        "sample_rate_distribution": rates,
        "channel_distribution": channels,
        "corrupt_files": corrupt,
        "missing_metadata": not bool(metadata),
        "status": "ready" if files and not corrupt and metadata else "blocked",
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svarah-path", required=True)
    parser.add_argument("--indic-audio-path", required=True)
    parser.add_argument("--orpheus-path", required=True)
    args = parser.parse_args()
    report = [
        inspect("svarah", Path(args.svarah_path)),
        inspect("indic_audio", Path(args.indic_audio_path)),
        inspect("orpheus", Path(args.orpheus_path)),
    ]
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
