"""Run Module 1 on one audio file.

Change AUDIO_SOURCE only; VAD and preprocessing remain reusable in vad.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from config import DEFAULT_OUTPUT_DIR, SAMPLE_RATE
from vad import SileroVAD

# ----------------------- User configuration -----------------------
AUDIO_SOURCE = "input_audio/test.wav"  # Or e.g. "input_audio/person2.wav"
SAVE_OUTPUT = True
OUTPUT_DIR = "output"
# --------------------------------------------------------------------


def save_chunks(chunks: list[dict], source: str | Path, output_dir: str | Path) -> Path:
    """Write optional debug WAVs and a JSON manifest; callers keep arrays in memory."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_segments = []
    for index, chunk in enumerate(chunks):
        filename = f"speech_{index:03d}.wav"
        sf.write(output_path / filename, chunk["audio"], SAMPLE_RATE, subtype="PCM_16")
        manifest_segments.append(
            {
                "id": index,
                "start": round(chunk["start"], 6),
                "end": round(chunk["end"], 6),
                "duration": round(chunk["end"] - chunk["start"], 6),
                "file": filename,
                "padded": chunk["padded"],
                "region_index": chunk["region_index"],
            }
        )

    manifest = {
        "source": str(source),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "segments": manifest_segments,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    import sys
    
    if len(sys.argv) > 1:
        sources = [sys.argv[1]]
    else:
        # Default to available real and synthetic voice test files
        sources = ["input_audio/real_voice.wav", "input_audio/synthetic_voice.wav"]

    detector = SileroVAD()

    for audio_src in sources:
        if audio_src.lower() == "microphone":
            print(f"Skipping microphone: capture is reserved for streaming adapter.")
            continue

        source = Path(audio_src)
        if not source.is_absolute():
            source = Path(__file__).resolve().parent / source
        if not source.is_file():
            print(f"Audio source not found: {source}")
            continue

        print(f"\n==================================================")
        print(f"Processing Audio Source: {source.name}")
        print(f"==================================================")
        waveform, regions, chunks = detector.process_file(source)
        print(f"Normalized Waveform Samples: {len(waveform)} ({len(waveform)/16000:.2f}s)")
        print(f"Speech regions detected: {len(regions)}")
        for idx, reg in enumerate(regions):
            print(f"  Region {idx+1}: {reg['start']:.3f}s -> {reg['end']:.3f}s (duration: {reg['end'] - reg['start']:.3f}s)")

        print(f"Sliding-window chunks generated (1-3s windows): {len(chunks)}")
        for idx, ch in enumerate(chunks):
            print(f"  Chunk {idx:03d}: start={ch['start']:.3f}s, end={ch['end']:.3f}s, padded={ch['padded']}, region={ch['region_index']}")

        if SAVE_OUTPUT:
            out_subdir = Path(OUTPUT_DIR) / source.stem
            manifest = save_chunks(chunks, audio_src, out_subdir)
            print(f"Debug WAVs and manifest saved to: {manifest}")


if __name__ == "__main__":
    main()

