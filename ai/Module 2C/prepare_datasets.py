"""Prepare the three approved Module 2B datasets into portable manifests.

This utility never runs VAD. It splits original clips first, then writes fixed
one-second chunks and preserves their source metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dataset import AudioRecord, read_manifest, write_manifest
from input import load_audio, normalize_audio, write_wave
from spectral_config import HOP_LENGTH, SAMPLE_RATE, WINDOW_SAMPLES
from split_dataset import split_integrity_report, split_records

DATASET_NAMES = {
    "svarah": (0, "real"),
    "indic_audio": (1, "synthetic"),
    "orpheus": (1, "synthetic"),
}
INDIC_GROUPS = {
    "bed_hindi": "bed_hindi",
    "en_aman": "aman", "hing_aman": "aman",
    "en_ananya": "ananya", "hing_ananya": "ananya",
    "en_arjun": "arjun", "hing_arjun": "arjun",
    "hi_atul": "atul", "hing_atul": "atul",
    "en_dev": "dev", "hing_dev": "dev",
    "en_divya": "divya", "hing_divya": "divya",
    "en_kabir": "kabir", "hing_kabir": "kabir",
    "hi_meera": "meera", "hing_meera": "meera",
    "en_nisha": "nisha", "hing_nisha": "nisha",
    "en_priya": "priya", "hing_priya": "priya",
    "hi_ravi": "ravi", "hing_ravi": "ravi",
    "en_sameer": "sameer", "hing_sameer": "sameer",
    "hi_shivani": "shivani", "hing_shivani": "shivani",
    "en_tara": "tara", "hing_tara": "tara",
}
HF_REPOSITORIES = {
    "svarah": "ai4bharat/Svarah",
    "indic_audio": "BH-Builds/indic-audio",
    "orpheus": "ar17to/orpheus_tts_english_indian_multispeaker",
}
HF_REQUESTED_SPLITS = {
    "svarah": "test",
    "indic_audio": "train",
    "orpheus": "train",
}
HF_FALLBACK_SPLITS = {
    "svarah": ("test",),
    "indic_audio": ("validation",),
    "orpheus": ("train",),
}


@dataclass(frozen=True)
class SourceExample:
    dataset: str
    audio_path: str
    label: int
    source_id: str
    speaker_id: str = "unavailable"
    generator: str = "unavailable"
    sample_id: str = ""
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    waveform: np.ndarray | None = None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _duration_and_shape(path: str | Path) -> tuple[float, int, int]:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate, info.channels


def _example_to_record(example: SourceExample) -> AudioRecord:
    return AudioRecord(
        audio_path=example.audio_path,
        label=example.label,
        source_id=example.source_id,
        speaker_id=example.speaker_id,
        generator=example.generator,
        sample_id=example.sample_id or example.source_id,
        duration_seconds=example.duration_seconds,
    )


def _local_audio(path: str | Path) -> tuple[np.ndarray, int, int]:
    import soundfile as sf
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return audio, int(sample_rate), int(audio.shape[1])


def load_indic_audio(root: str | Path) -> list[SourceExample]:
    root = Path(root)
    metadata_path = root / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Indic Audio metadata not found: {metadata_path}")
    examples = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relative = row.get("audio")
        voice = row.get("voice")
        if not relative or not voice:
            raise ValueError("Indic Audio metadata requires audio and voice")
        audio_path = root / relative
        if not audio_path.exists():
            raise FileNotFoundError(f"Indic Audio file listed in metadata is missing: {audio_path}")
        duration, sample_rate, channels = _duration_and_shape(audio_path)
        examples.append(SourceExample(
            dataset="indic_audio", audio_path=str(audio_path), label=1,
            source_id=f"indic_audio:{relative}",
            speaker_id=INDIC_GROUPS.get(voice, voice),
            generator="Fish Audio S2 Pro", sample_id=f"indic_audio:{relative}",
            duration_seconds=duration, sample_rate=sample_rate, channels=channels,
        ))
    return examples


def load_svarah_export(root: str | Path) -> list[SourceExample]:
    root = Path(root)
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Svarah local input must be an exported directory containing metadata.csv"
        )
    examples = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative = row.get("audio_path") or row.get("filename") or row.get("audio_filepath")
            if not relative:
                raise ValueError("Svarah metadata requires filename/audio_path")
            audio_path = root / "audio" / relative if not Path(relative).is_absolute() else Path(relative)
            if not audio_path.exists():
                raise FileNotFoundError(f"Svarah audio file listed in metadata is missing: {audio_path}")
            duration, sample_rate, channels = _duration_and_shape(audio_path)
            speaker_id = row.get("speaker_id") or "unavailable"
            examples.append(SourceExample(
                dataset="svarah", audio_path=str(audio_path), label=0,
                source_id=f"svarah:{relative}", speaker_id=speaker_id,
                sample_id=f"svarah:{relative}", duration_seconds=float(row.get("duration") or duration),
                sample_rate=sample_rate, channels=channels,
            ))
    return examples


def load_audio_directory(root: str | Path, dataset: str, label: int) -> list[SourceExample]:
    root = Path(root)
    examples = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac", ".ogg", ".mp3"}:
            continue
        duration, sample_rate, channels = _duration_and_shape(path)
        speaker = "unavailable"
        generator = "unavailable"
        if dataset == "indic_audio":
            voice = path.parent.name
            speaker = INDIC_GROUPS.get(voice, voice)
            generator = "Fish Audio S2 Pro"
        relative = path.relative_to(root).as_posix()
        examples.append(SourceExample(
            dataset=dataset, audio_path=str(path), label=label,
            source_id=f"{dataset}:{relative}", speaker_id=speaker,
            generator=generator, sample_id=f"{dataset}:{relative}",
            duration_seconds=duration, sample_rate=sample_rate, channels=channels,
        ))
    return examples


def validate_local_dataset_paths(
    svarah_path: str | Path,
    indic_audio_path: str | Path,
    orpheus_path: str | Path,
) -> dict[str, int]:
    """Validate the three local inputs before creating any manifests/chunks."""
    svarah = Path(svarah_path)
    indic = Path(indic_audio_path)
    orpheus = Path(orpheus_path)
    if not (svarah / "metadata.csv").is_file():
        raise FileNotFoundError(
            "Svarah local input must contain metadata.csv and an audio/ directory"
        )
    if not (svarah / "audio").is_dir():
        raise FileNotFoundError("Svarah local input is missing its audio/ directory")
    if not (indic / "metadata.jsonl").is_file():
        raise FileNotFoundError(
            "Indic Audio local input must contain metadata.jsonl and audio files"
        )
    if not indic.is_dir():
        raise FileNotFoundError(f"Indic Audio directory does not exist: {indic}")
    audio_extensions = {".wav", ".flac", ".ogg", ".mp3"}
    orpheus_files = [
        path for path in orpheus.rglob("*")
        if path.is_file() and path.suffix.lower() in audio_extensions
    ] if orpheus.is_dir() else []
    if not orpheus_files:
        raise FileNotFoundError(
            "Orpheus local input must be an extracted directory containing "
            "audio files (.wav/.flac/.ogg/.mp3), not only Parquet metadata"
        )
    return {
        "svarah_audio_files": sum(1 for _ in (svarah / "audio").rglob("*")),
        "indic_audio_files": sum(1 for path in indic.rglob("*") if path.is_file() and path.suffix.lower() in audio_extensions),
        "orpheus_audio_files": len(orpheus_files),
    }


def select_hf_split(
    dataset_name: str,
    available_splits: Iterable[str],
    requested_split: str | None = None,
) -> str:
    """Select a documented split without guessing among multiple choices."""
    available = sorted({str(split) for split in available_splits})
    if not available:
        raise ValueError(f"No splits are available for {dataset_name}")
    requested = requested_split or HF_REQUESTED_SPLITS[dataset_name]
    if requested in available:
        return requested
    if "validation" in available:
        return "validation"
    if len(available) == 1:
        return available[0]
    raise ValueError(
        f"Requested split {requested!r} is unavailable for {dataset_name}; "
        f"available splits are {available} and no deterministic fallback applies"
    )


def _load_bounded_indic_audio(
    limit: int,
    output_dir: Path,
) -> list[SourceExample]:
    """Fetch only metadata and at most ``limit`` Indic Audio files.

    The dataset builder currently resolves a large validation file set even in
    streaming mode. Direct Hub file access keeps bounded smoke tests lazy.
    """
    from huggingface_hub import hf_hub_download

    root = output_dir / "sources" / "indic_audio"
    metadata_path = hf_hub_download(
        repo_id=HF_REPOSITORIES["indic_audio"],
        repo_type="dataset",
        filename="metadata.jsonl",
        local_dir=root,
    )
    rows = [
        json.loads(line)
        for line in Path(metadata_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        voice = str(row.get("voice") or "unavailable")
        grouped.setdefault(INDIC_GROUPS.get(voice, voice), []).append((index, row))
    selected: list[tuple[int, dict[str, Any]]] = []
    group_names = sorted(grouped)
    while len(selected) < limit:
        progressed = False
        for group_name in group_names:
            if grouped[group_name]:
                selected.append(grouped[group_name].pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break

    examples: list[SourceExample] = []
    for index, row in selected:
        relative = row.get("audio")
        voice = str(row.get("voice") or "unavailable")
        if not relative or voice == "unavailable":
            continue
        audio_path = hf_hub_download(
            repo_id=HF_REPOSITORIES["indic_audio"],
            repo_type="dataset",
            filename=relative,
            local_dir=root,
        )
        duration, sample_rate, channels = _duration_and_shape(audio_path)
        source_id = f"indic_audio:{relative}"
        examples.append(SourceExample(
            dataset="indic_audio",
            audio_path=str(audio_path),
            label=1,
            source_id=source_id,
            speaker_id=INDIC_GROUPS.get(voice, voice),
            generator="Fish Audio S2 Pro",
            sample_id=source_id,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
        ))
    return examples


def _available_hf_splits(repo: str, dataset_name: str) -> list[str]:
    """Read split names from lightweight Hub metadata, with known fallbacks."""
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id=repo, repo_type="dataset")
        discovered = set()
        for filename in files:
            match = re.search(r"(?:^|/)(train|validation|test)(?:[-_/]|$)", filename)
            if match:
                discovered.add(match.group(1))
        if discovered:
            return sorted(discovered)
    except Exception:
        pass
    return list(HF_FALLBACK_SPLITS[dataset_name])


def load_hf_stream(
    dataset_name: str,
    limit: int | None,
    output_dir: Path,
    *,
    load_dataset_fn=None,
    get_dataset_split_names_fn=None,
    cast_audio_column_fn=None,
) -> list[SourceExample]:
    """Load a bounded HF stream and persist only the selected source clips."""
    if (
        dataset_name == "indic_audio"
        and limit is not None
        and load_dataset_fn is None
        and get_dataset_split_names_fn is None
    ):
        return _load_bounded_indic_audio(limit, output_dir)

    download_config = None
    audio_column = "audio_filepath" if dataset_name == "svarah" else "audio"
    if load_dataset_fn is None or get_dataset_split_names_fn is None:
        from datasets import DownloadConfig, load_dataset
        from datasets import Audio
        load_dataset_fn = load_dataset
        get_dataset_split_names_fn = lambda repo: _available_hf_splits(
            repo,
            dataset_name,
        )
        download_config = DownloadConfig(max_retries=0)
        if cast_audio_column_fn is None:
            cast_audio_column_fn = lambda stream: stream.cast_column(
                audio_column,
                Audio(decode=False),
            )
    repo = HF_REPOSITORIES[dataset_name]
    available_splits = get_dataset_split_names_fn(repo)
    split = select_hf_split(
        dataset_name,
        available_splits,
        HF_REQUESTED_SPLITS[dataset_name],
    )
    load_kwargs = {"split": split, "streaming": True}
    if download_config is not None:
        load_kwargs["download_config"] = download_config
    try:
        stream = load_dataset_fn(repo, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize lazy Hugging Face streaming for "
            f"{dataset_name} ({repo}, split={split}): {exc}"
        ) from exc
    if cast_audio_column_fn is not None:
        try:
            stream = cast_audio_column_fn(stream)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to disable automatic Hugging Face Audio decoding "
                f"for {dataset_name}: {exc}"
            ) from exc
    source_dir = output_dir / "sources" / dataset_name
    examples = []
    candidate_rows = []
    candidate_groups = set()
    stream_iterator = None
    try:
        stream_iterator = iter(stream)
        while True:
            index = len(candidate_rows)
            if limit is not None and dataset_name != "indic_audio" and index >= limit:
                break
            try:
                row = next(stream_iterator)
            except StopIteration:
                break
            candidate_rows.append((index, row))
            if dataset_name == "indic_audio":
                audio_for_group = row.get("audio") or {}
                path_for_group = str(audio_for_group.get("path") or "")
                candidate_groups.add(Path(path_for_group).parent.name)
                if limit is not None and len(candidate_rows) >= limit and len(candidate_groups) >= 3:
                    break
    except Exception as exc:
        raise RuntimeError(
            f"Unable to stream {dataset_name} ({repo}, split={split}) "
            "without materializing the dataset. The bounded smoke test "
            f"stopped after {len(candidate_rows)} row(s); no further retries "
            f"will be attempted: {exc}"
        ) from exc
    finally:
        close_iterator = getattr(stream_iterator, "close", None)
        if callable(close_iterator):
            close_iterator()
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            close_stream()
    if limit is not None and dataset_name == "indic_audio" and len(candidate_rows) > limit:
        grouped_rows = {}
        for index, row in candidate_rows:
            path_for_group = str((row.get("audio") or {}).get("path") or "")
            grouped_rows.setdefault(Path(path_for_group).parent.name, []).append((index, row))
        selected_rows = []
        group_names = sorted(grouped_rows)
        while len(selected_rows) < limit:
            progressed = False
            for group_name in group_names:
                if grouped_rows[group_name]:
                    selected_rows.append(grouped_rows[group_name].pop(0))
                    progressed = True
                    if len(selected_rows) >= limit:
                        break
            if not progressed:
                break
        candidate_rows = selected_rows
    for index, row in candidate_rows:
        audio = row.get(audio_column)
        if not audio:
            continue
        array, sample_rate = _decode_hf_audio_reference(
            audio,
            dataset_name,
            output_dir,
        )
        source_path = str(audio.get("path") or f"{index:08d}.wav")
        relative = f"{index:08d}_{_safe_name(Path(source_path).name)}.wav"
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        normalized = normalize_audio(array, sample_rate)
        write_wave(destination, normalized.waveform, SAMPLE_RATE)
        if dataset_name == "svarah":
            label, source_prefix = 0, "svarah"
            speaker, generator = "unavailable", "unavailable"
        elif dataset_name == "indic_audio":
            label, source_prefix = 1, "indic_audio"
            voice = str(row.get("voice") or "unavailable")
            speaker, generator = INDIC_GROUPS.get(voice, voice), "Fish Audio S2 Pro"
        else:
            label, source_prefix = 1, "orpheus"
            speaker, generator = "unavailable", "unavailable"
        if dataset_name == "indic_audio" and speaker == "unavailable":
            documented_voice = Path(source_path).parent.name
            speaker = INDIC_GROUPS.get(documented_voice, "unavailable")
            if speaker == "unavailable":
                raise ValueError(
                    "Indic Audio streamed record has no verified voice field "
                    f"and an unrecognized documented audio directory: {source_path}"
                )
        source_id = f"{source_prefix}:{index}:{source_path}"
        examples.append(SourceExample(
            dataset=dataset_name, audio_path=str(destination), label=label,
            source_id=source_id, speaker_id=speaker, generator=generator,
            sample_id=source_id, duration_seconds=len(normalized.waveform) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE, channels=1,
        ))
    return examples


def _decode_hf_audio_reference(
    audio: dict[str, Any],
    dataset_name: str,
    output_dir: Path,
) -> tuple[np.ndarray, int]:
    """Decode raw HF Audio bytes/reference without invoking Audio.decode_example."""
    import soundfile as sf

    if audio.get("bytes"):
        try:
            waveform, sample_rate = sf.read(
                BytesIO(audio["bytes"]),
                always_2d=False,
                dtype="float32",
            )
        except Exception as exc:
            raise RuntimeError(
                f"{dataset_name} streamed raw audio bytes that could not be "
                f"decoded without automatic HF Audio decoding: {exc}"
            ) from exc
        return np.asarray(waveform, dtype=np.float32), int(sample_rate)

    path = audio.get("path")
    if path and Path(path).is_file():
        loaded = load_audio(path)
        return loaded.waveform, loaded.sample_rate

    raise RuntimeError(
        f"{dataset_name} streamed a raw audio reference without decodable "
        "bytes or a local path; automatic Hugging Face Audio decoding is "
        "disabled and no heavyweight decoder fallback will be used"
    )


def _load_source(path: str, waveform: np.ndarray | None) -> np.ndarray:
    if waveform is not None:
        return normalize_audio(waveform, SAMPLE_RATE).waveform
    return load_audio(path).waveform


def _chunk_source(example: SourceExample, split: str, output_dir: Path) -> list[AudioRecord]:
    audio = _load_source(example.audio_path, example.waveform)
    hop = HOP_LENGTH * 100  # 160 samples * 100 = 0.5 seconds at 16 kHz.
    window = WINDOW_SAMPLES
    records = []
    for offset in range(0, len(audio), hop):
        raw = audio[offset:offset + window]
        if len(raw) == 0:
            continue
        padded = len(raw) < window
        chunk = np.pad(raw, (0, window - len(raw))) if padded else raw
        chunk_id = len(records)
        start = offset / SAMPLE_RATE
        end = start + len(raw) / SAMPLE_RATE
        relative = Path("audio") / example.dataset / split / f"{_safe_name(example.sample_id)}_chunk_{chunk_id:04d}.wav"
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_wave(destination, chunk, SAMPLE_RATE)
        records.append(AudioRecord(
            audio_path=relative.as_posix(), label=example.label,
            source_id=example.source_id, speaker_id=example.speaker_id,
            generator=example.generator, split=split,
            sample_id=f"{example.sample_id}:chunk:{chunk_id}",
            duration_seconds=len(raw) / SAMPLE_RATE, chunk_id=chunk_id,
            start=start, end=end, padded=padded, region_index=None,
            dataset=example.dataset,
        ))
        if padded:
            break
    return records


def _assert_unique(records: list[AudioRecord], field: str) -> None:
    values = [getattr(record, field) for record in records]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {field} values detected")


def prepare_examples(examples: list[SourceExample], output_dir: str | Path, seed: int = 0) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_dataset = {name: [example for example in examples if example.dataset == name] for name in DATASET_NAMES}
    split_examples: dict[str, list[SourceExample]] = {"train": [], "validation": [], "test": []}
    source_reports = {}
    for dataset_name, dataset_examples in by_dataset.items():
        if not dataset_examples:
            source_reports[dataset_name] = {"status": "not_available", "original_clip_count": 0}
            continue
        records = [_example_to_record(example) for example in dataset_examples]
        group_by = "speaker_id" if dataset_name == "indic_audio" else "source_id"
        splits = split_records(records, seed=seed, group_by=group_by)
        lookup = {record.sample_id: example for record, example in zip(records, dataset_examples)}
        source_reports[dataset_name] = {
            "status": "audited",
            "original_clip_count": len(dataset_examples),
            "source_count": len({example.source_id for example in dataset_examples}),
            "speaker_group_count": len({example.speaker_id for example in dataset_examples if example.speaker_id != "unavailable"}),
            "generator_count": len({example.generator for example in dataset_examples if example.generator != "unavailable"}),
            "duration_seconds": sum(example.duration_seconds or 0.0 for example in dataset_examples),
            "sample_rates": sorted({example.sample_rate for example in dataset_examples if example.sample_rate}),
            "channels": sorted({example.channels for example in dataset_examples if example.channels}),
            "label_counts": dict(Counter(example.label for example in dataset_examples)),
            "missing_metadata_counts": {
                "speaker_id": sum(example.speaker_id == "unavailable" for example in dataset_examples),
                "generator": sum(example.generator == "unavailable" for example in dataset_examples),
                "source_id": sum(not example.source_id for example in dataset_examples),
            },
        }
        for split, split_records_ in splits.items():
            split_examples[split].extend(lookup[record.sample_id] for record in split_records_)

    chunk_splits = {split: [] for split in split_examples}
    for split, source_items in split_examples.items():
        for example in source_items:
            chunk_splits[split].extend(_chunk_source(example, split, output_dir))

    _assert_unique([record for records in chunk_splits.values() for record in records], "sample_id")
    manifest_path = output_dir / "module2b_manifest.csv"
    write_manifest([record for records in chunk_splits.values() for record in records], manifest_path)
    split_paths = {}
    for split, records in chunk_splits.items():
        filename = "module2b_val.csv" if split == "validation" else f"module2b_{split}.csv"
        path = output_dir / filename
        write_manifest(records, path)
        split_paths[split] = str(path)

    split_dataset_counts = {}
    for split, records in chunk_splits.items():
        split_dataset_counts[split] = {}
        for dataset_name in DATASET_NAMES:
            selected = [record for record in records if record.dataset == dataset_name]
            split_dataset_counts[split][dataset_name] = {
                "label": DATASET_NAMES[dataset_name][0],
                "original_source_count": len({record.source_id for record in selected}),
                "chunk_count": len(selected),
                "duration_seconds": sum(record.duration_seconds or 0.0 for record in selected),
            }

    report = {
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": source_reports,
        "splits": split_integrity_report(chunk_splits)["splits"],
        "split_dataset_counts": split_dataset_counts,
        "overlap": split_integrity_report(chunk_splits)["overlap"],
        "chunk_count": {split: len(records) for split, records in chunk_splits.items()},
        "manifest": str(manifest_path),
        "split_manifests": split_paths,
        "rejected_files": [],
    }
    (output_dir / "module2b_dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def validate_manifests(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    paths = {"train": output_dir / "module2b_train.csv", "validation": output_dir / "module2b_val.csv", "test": output_dir / "module2b_test.csv"}
    splits = {split: read_manifest(path) for split, path in paths.items()}
    all_records = [record for records in splits.values() for record in records]
    _assert_unique(all_records, "sample_id")
    for record in all_records:
        if record.label not in (0, 1):
            raise ValueError(f"invalid label: {record.label}")
        if not Path(record.audio_path).exists():
            raise FileNotFoundError(record.audio_path)
        loaded = load_audio(record.audio_path)
        if not np.isfinite(loaded.waveform).all() or len(loaded.waveform) != WINDOW_SAMPLES:
            raise ValueError(f"invalid prepared chunk: {record.audio_path}")
    report = split_integrity_report(splits)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Svarah, Indic Audio, and Orpheus for Module 2B")
    parser.add_argument("--svarah-path")
    parser.add_argument("--indic-audio-path")
    parser.add_argument("--orpheus-path")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Require all three local dataset roots; never access Hugging Face",
    )
    parser.add_argument("--use-huggingface", action="store_true")
    parser.add_argument("--max-records", type=int, default=None, help="Bound HF streaming; omit only when full acquisition is intentional")
    parser.add_argument("--output-dir", default="data/module2b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_manifests(args.output_dir), indent=2))
        return
    if args.local_only:
        if not all((args.svarah_path, args.indic_audio_path, args.orpheus_path)):
            raise SystemExit(
                "--local-only requires --svarah-path, --indic-audio-path, "
                "and --orpheus-path"
            )
        local_counts = validate_local_dataset_paths(
            args.svarah_path,
            args.indic_audio_path,
            args.orpheus_path,
        )
        print(json.dumps({"local_inputs_validated": local_counts}, indent=2))
    examples: list[SourceExample] = []
    if args.svarah_path:
        examples.extend(load_svarah_export(args.svarah_path))
    if args.indic_audio_path:
        examples.extend(load_indic_audio(args.indic_audio_path))
    if args.orpheus_path:
        examples.extend(load_audio_directory(args.orpheus_path, "orpheus", 1))
    if args.use_huggingface:
        output_dir = Path(args.output_dir)
        for name in ("svarah", "indic_audio", "orpheus"):
            examples.extend(load_hf_stream(name, args.max_records, output_dir))
    if not examples:
        raise SystemExit("No dataset sources supplied. Provide local paths or --use-huggingface.")
    report = prepare_examples(examples, args.output_dir, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
