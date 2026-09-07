"""Verify frozen XLS-R extraction against the committed Module 1 examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from config import MODEL_DIR, PROJECT_ROOT
from extract_embeddings import load_module1_records
from xlsr import EMBEDDING_DIM, MODEL_ID, XLSREmbeddingExtractor


def _describe(label: str, record: dict, extractor: XLSREmbeddingExtractor) -> dict:
    waveform, sample_rate = sf.read(record["file_path"], always_2d=False, dtype="float32")
    valid_length = min(len(waveform), max(1, round(record["duration"] * sample_rate)))
    embedding = extractor.extract(waveform, sample_rate, valid_length=valid_length)
    result = {
        "label": label, "chunk_id": record["id"], "padded": record["padded"],
        "embedding_shape": tuple(embedding.shape), "embedding_mean": float(embedding.mean()),
        "embedding_std": float(embedding.std()), "nan": bool(np.isnan(embedding).any()), "inf": bool(np.isinf(embedding).any()),
    }
    print(f"{label} chunk {record['id']}: shape = {result['embedding_shape']}, mean = {result['embedding_mean']:.6f}, std = {result['embedding_std']:.6f}, NaN = {result['nan']}, Inf = {result['inf']}")
    if embedding.shape != (EMBEDDING_DIM,) or result["nan"] or result["inf"]:
        raise AssertionError(f"invalid XLS-R embedding for {label} chunk {record['id']}")
    return result


def main() -> None:
    print("=" * 50)
    print("MODULE 2B XLS-R VERIFICATION")
    print("=" * 50)
    print("Model:\nWav2Vec2 XLS-R 300M\n\nInput:\n16 kHz mono Float32")
    extractor = XLSREmbeddingExtractor(MODEL_DIR, device="auto")
    print(f"\nProcessor:\n{extractor.processor_name}\n\nDevice:\n{extractor.device}\n")
    real = load_module1_records(PROJECT_ROOT / "Module 1" / "output" / "real_voice", dataset="real_voice", label=0)
    synthetic = load_module1_records(PROJECT_ROOT / "Module 1" / "output" / "synthetic_voice", dataset="synthetic_voice", label=1)
    results = [_describe("Real", record, extractor) for record in real[:2]]
    results.extend(_describe("Synthetic", record, extractor) for record in synthetic[:2])
    print("\nStatus:\nPASS")
    print(json.dumps({"model": MODEL_ID, "processor": extractor.processor_name, "device": str(extractor.device), "records": results}, indent=2))


if __name__ == "__main__":
    main()
