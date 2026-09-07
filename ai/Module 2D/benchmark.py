"""Latency benchmark for representative one-second Module 1 windows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

from config import MODEL_DIR, MODULE_DIR, PROJECT_ROOT
from extract_embeddings import load_module1_records
from xlsr import EMBEDDING_DIM, MODEL_ID, XLSREmbeddingExtractor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean_ms": float(np.mean(values) * 1_000), "median_ms": float(np.median(values) * 1_000), "p95_ms": float(np.percentile(values, 95) * 1_000)}


def benchmark(*, repetitions: int = 10, batch_size: int = 1, device: str = "auto", output_path: Path | None = None) -> dict:
    if not 10 <= repetitions <= 100:
        raise ValueError("repetitions must be between 10 and 100")
    records = load_module1_records(PROJECT_ROOT / "Module 1" / "output" / "real_voice")
    waveform, sample_rate = sf.read(records[0]["file_path"], always_2d=False, dtype="float32")
    extractor = XLSREmbeddingExtractor(MODEL_DIR, device=device)
    waveforms = [waveform] * batch_size
    valid_lengths = [len(waveform)] * batch_size
    # Warm-up is excluded from reported timing to avoid one-time kernel/setup cost.
    extractor.extract_batch(waveforms, sample_rate, valid_lengths=valid_lengths)
    prep, forward, pool, total = [], [], [], []
    for _ in range(repetitions):
        start = time.perf_counter()
        inputs = extractor._processor_inputs(waveforms, valid_lengths)
        _sync(extractor.device)
        after_prep = time.perf_counter()
        with torch.no_grad():
            hidden = extractor.model(**inputs).last_hidden_state
        _sync(extractor.device)
        after_forward = time.perf_counter()
        with torch.no_grad():
            embedding = extractor._pool(hidden, inputs["attention_mask"])
        _sync(extractor.device)
        finish = time.perf_counter()
        if embedding.shape != (batch_size, EMBEDDING_DIM):
            raise AssertionError(f"unexpected benchmark embedding shape: {embedding.shape}")
        prep.append(after_prep - start)
        forward.append(after_forward - after_prep)
        pool.append(finish - after_forward)
        total.append(finish - start)
    result = {
        "module": "2B", "model": MODEL_ID, "device": str(extractor.device), "batch_size": batch_size,
        "window_samples": len(waveform), "embedding_dimension": EMBEDDING_DIM, "repetitions": repetitions,
        "processor": _stats(prep), "forward": _stats(forward), "mean_pool": _stats(pool), "total": _stats(total),
        "process_rss_mb": round(psutil.Process().memory_info().rss / (1024 ** 2), 2),
        "cuda_allocated_mb": round(torch.cuda.memory_allocated(extractor.device) / (1024 ** 2), 2) if extractor.device.type == "cuda" else None,
    }
    if output_path is None:
        output_path = MODULE_DIR / "output" / "benchmarks" / f"{extractor.device.type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen XLS-R on a representative Module 1 speech window")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(benchmark(repetitions=args.repetitions, batch_size=args.batch_size, device=args.device, output_path=args.output), indent=2))


if __name__ == "__main__":
    main()
