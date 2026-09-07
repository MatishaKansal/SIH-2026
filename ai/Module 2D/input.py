"""Run one frozen XLS-R extraction for a Module 1 speech WAV."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from config import DEVICE, MODEL_DIR
from xlsr import XLSREmbeddingExtractor

INPUT_AUDIO = "../Module 1/output/real_voice/speech_000.wav"


def main() -> None:
    path = Path(INPUT_AUDIO)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    waveform, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    result = XLSREmbeddingExtractor(MODEL_DIR, device=DEVICE).predict(waveform, sample_rate)
    # Full embeddings belong in NPY storage; this is a compact interactive record.
    print(json.dumps({**result, "embedding": {"shape": list(result["embedding"].shape), "mean": float(result["embedding"].mean())}}, indent=2))


if __name__ == "__main__":
    main()
