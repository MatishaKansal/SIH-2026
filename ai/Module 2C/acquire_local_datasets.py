"""Acquire and extract only the three approved Module 2B datasets.

This uses direct Hugging Face file downloads and Parquet extraction, never
Hugging Face IterableDataset streaming. It preserves source audio bytes and
writes the metadata shape expected by prepare_datasets.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

REPOS = {
    "svarah": "ai4bharat/Svarah",
    "indic_audio": "BH-Builds/indic-audio",
    "orpheus": "ar17to/orpheus_tts_english_indian_multispeaker",
}
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def _safe_relative(path: str, default_name: str) -> Path:
    candidate = Path(path or default_name)
    if candidate.is_absolute() or ".." in candidate.parts:
        candidate = Path(candidate.name)
    return candidate


def _write_embedded_audio(audio: dict[str, Any], destination: Path) -> None:
    payload = audio.get("bytes")
    if not payload:
        raise ValueError(f"embedded audio has no bytes: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def acquire_svarah(output_root: Path, api: HfApi) -> dict[str, Any]:
    root = output_root / "svarah"
    audio_root = root / "audio"
    files = sorted(
        file for file in api.list_repo_files(REPOS["svarah"], repo_type="dataset")
        if file.startswith("data/") and file.endswith(".parquet")
    )
    if not files:
        raise RuntimeError("No Svarah parquet shards were found")
    metadata_rows = []
    audio_count = 0
    for remote in files:
        local = hf_hub_download(REPOS["svarah"], remote, repo_type="dataset", local_dir=output_root / ".downloads")
        import pyarrow.parquet as pq
        table = pq.read_table(local)
        for row in table.to_pylist():
            audio = row["audio_filepath"]
            relative = _safe_relative(audio.get("path", f"{audio_count:08d}.wav"), f"{audio_count:08d}.wav")
            destination = audio_root / relative
            _write_embedded_audio(audio, destination)
            metadata_rows.append({
                "filename": relative.as_posix(),
                "duration": row.get("duration"),
                "text": row.get("text"),
                "gender": row.get("gender"),
                "age_group": row.get("age-group"),
                "primary_language": row.get("primary_language"),
                "native_place_state": row.get("native_place_state"),
                "native_place_district": row.get("native_place_district"),
                "highest_qualification": row.get("highest_qualification"),
                "job_category": row.get("job_category"),
                "occupation_domain": row.get("occupation_domain"),
                "speaker_id": "unavailable",
            })
            audio_count += 1
    fields = list(metadata_rows[0])
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata_rows)
    return {"dataset": "svarah", "shards": len(files), "audio_files": audio_count, "speaker_id": "unavailable"}


def acquire_indic_audio(output_root: Path, api: HfApi) -> dict[str, Any]:
    root = output_root / "indic-audio"
    root.mkdir(parents=True, exist_ok=True)
    metadata = hf_hub_download(REPOS["indic_audio"], "metadata.jsonl", repo_type="dataset", local_dir=root)
    rows = [json.loads(line) for line in Path(metadata).read_text(encoding="utf-8").splitlines() if line.strip()]
    audio_count = 0
    for row in rows:
        relative = row["audio"]
        hf_hub_download(REPOS["indic_audio"], relative, repo_type="dataset", local_dir=root)
        audio_count += 1
    return {"dataset": "indic_audio", "metadata": "metadata.jsonl", "audio_files": audio_count, "generator": "Fish Audio S2 Pro"}


def acquire_orpheus(output_root: Path, api: HfApi) -> dict[str, Any]:
    root = output_root / "orpheus"
    audio_root = root / "audio"
    files = sorted(
        file for file in api.list_repo_files(REPOS["orpheus"], repo_type="dataset")
        if file.startswith("data/") and file.endswith(".parquet")
    )
    if not files:
        raise RuntimeError("No Orpheus parquet shards were found")
    metadata_rows = []
    audio_count = 0
    for remote in files:
        local = hf_hub_download(REPOS["orpheus"], remote, repo_type="dataset", local_dir=output_root / ".downloads")
        import pyarrow.parquet as pq
        table = pq.read_table(local)
        for row in table.to_pylist():
            audio = row["audio"]
            relative = _safe_relative(audio.get("path", f"{audio_count:08d}.wav"), f"{audio_count:08d}.wav")
            destination = audio_root / relative
            _write_embedded_audio(audio, destination)
            metadata_rows.append({
                "filename": relative.as_posix(),
                "text": row.get("text"),
                "source": row.get("source"),
                "speaker_id": "unavailable",
                "generator": "unavailable",
            })
            audio_count += 1
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(metadata_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata_rows)
    return {"dataset": "orpheus", "shards": len(files), "audio_files": audio_count, "speaker_id": "unavailable", "generator": "unavailable"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/extract the three approved Module 2B datasets")
    parser.add_argument("--output-root", default="data/raw")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    results = [
        acquire_svarah(output_root, api),
        acquire_indic_audio(output_root, api),
        acquire_orpheus(output_root, api),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
