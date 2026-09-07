"""Extract and store XLS-R embeddings for Module 1 speech-window manifests."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from config import DEFAULT_OUTPUT_DIR, MODEL_DIR
from xlsr import EMBEDDING_DIM, MODEL_ID, XLSREmbeddingExtractor

METADATA_FIELDS = [
    "embedding_id", "embedding_index", "dataset", "source_id", "source_audio", "speaker_id", "generator",
    "chunk_id", "start", "end", "duration", "padded", "region_index", "label", "sample_rate", "file",
]


def load_module1_records(module1_dir: str | Path, *, dataset: str | None = None, label: int | None = None) -> list[dict[str, Any]]:
    """Read a Module 1 manifest without changing its IDs or chunk boundaries."""
    chunk_dir = Path(module1_dir)
    manifest_path = chunk_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sample_rate") != 16_000 or manifest.get("channels") != 1:
        raise ValueError(f"{manifest_path} does not meet the Module 1 16 kHz mono contract")
    source = str(manifest.get("source", ""))
    source_id = Path(source).stem or chunk_dir.name
    records: list[dict[str, Any]] = []
    for segment in manifest.get("segments", []):
        file_path = chunk_dir / segment["file"]
        records.append({
            **segment,
            "file_path": file_path,
            "source_audio": source,
            "source_id": source_id,
            "dataset": dataset or chunk_dir.name,
            "label": label,
            "speaker_id": None,
            "generator": None,
            "sample_rate": int(manifest["sample_rate"]),
        })
    return records


def _batches(records: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    iterator = iter(records)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def extract_module1_embeddings(
    module1_dir: str | Path | Iterable[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dataset: str | None = None,
    label: int | None = None,
    batch_size: int = 4,
    device: str = "auto",
) -> dict[str, Any]:
    """Store one 1024-D row per Module 1 chunk plus durable mapping metadata."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    directories = [module1_dir] if isinstance(module1_dir, (str, Path)) else list(module1_dir)
    records = [
        record
        for directory in directories
        for record in load_module1_records(directory, dataset=dataset, label=label)
    ]
    if not records:
        raise ValueError("Module 1 manifest has no segments")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extractor = XLSREmbeddingExtractor(MODEL_DIR, device=device)
    # A .npy memmap prevents dataset-scale feature extraction from keeping all
    # embeddings in RAM. Only the configurable audio batch is resident.
    matrix = np.lib.format.open_memmap(output / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(records), EMBEDDING_DIM))
    padded_count = 0
    label_counts: dict[str, int] = {}
    row_index = 0
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for batch in _batches(records, batch_size):
            waveforms: list[np.ndarray] = []
            valid_lengths: list[int] = []
            for record in batch:
                info = sf.info(record["file_path"])
                if info.channels != 1 or info.samplerate != record["sample_rate"]:
                    raise ValueError(f"chunk violates the Module 1 mono/16 kHz contract: {record['file_path']}")
                waveform, sample_rate = sf.read(record["file_path"], always_2d=False, dtype="float32")
                waveforms.append(waveform)
                # Module 1 writes padded WAVs at one second. The manifest's true
                # duration is therefore the source of truth for masked pooling.
                valid_lengths.append(min(len(waveform), max(1, round(float(record["duration"]) * sample_rate))))
            batch_embeddings = extractor.extract_batch(waveforms, batch[0]["sample_rate"], valid_lengths=valid_lengths)
            matrix[row_index : row_index + len(batch)] = batch_embeddings
            for record in batch:
                metadata = {
                    "embedding_id": f"{record['source_id']}:{record['id']}", "embedding_index": row_index,
                    "dataset": record["dataset"], "source_id": record["source_id"], "source_audio": record["source_audio"],
                    "speaker_id": record["speaker_id"], "generator": record["generator"], "chunk_id": record["id"],
                    "start": record["start"], "end": record["end"], "duration": record["duration"],
                    "padded": record["padded"], "region_index": record["region_index"], "label": record["label"],
                    "sample_rate": record["sample_rate"], "file": record["file"],
                }
                writer.writerow(metadata)
                padded_count += bool(record["padded"])
                label_key = str(record["label"])
                label_counts[label_key] = label_counts.get(label_key, 0) + 1
                row_index += 1
    matrix.flush()
    if row_index != len(records):
        raise AssertionError(f"embedding storage count mismatch: {row_index} != {len(records)}")
    summary = {
        "module": "2B", "model": MODEL_ID, "device": str(extractor.device), "embedding_dimension": EMBEDDING_DIM,
        "processor": extractor.processor_name, "embedding_file": "embeddings.npy", "metadata_file": "metadata.csv", "count": row_index,
        "source_manifests": [str(Path(directory).resolve() / "manifest.json") for directory in directories],
        "padded_count": padded_count, "labels": label_counts,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen XLS-R embeddings from one Module 1 output directory")
    parser.add_argument("--module1-dir", type=Path, required=True, action="append", help="Directory containing manifest.json and speech_*.wav; may be repeated")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", help="Dataset name retained in metadata")
    parser.add_argument("--label", type=int, choices=(0, 1), help="Optional bona-fide=0 / synthetic=1 metadata label")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(extract_module1_embeddings(args.module1_dir, args.output_dir, dataset=args.dataset, label=args.label, batch_size=args.batch_size, device=args.device), indent=2))


if __name__ == "__main__":
    main()
