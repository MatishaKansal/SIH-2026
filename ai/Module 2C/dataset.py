"""Dataset utilities for Module 2B spectral tensors."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from augment import augment_waveform
from extractor import SpectralExtractor
from input import load_audio, prepare_chunk
from spectral_config import WINDOW_SAMPLES


@dataclass(frozen=True)
class AudioRecord:
    audio_path: str
    label: int
    source_id: str
    speaker_id: str = "unavailable"
    generator: str = "unavailable"
    split: str = ""
    sample_id: str = ""

    duration_seconds: float | None = None

    # Module 1 metadata
    chunk_id: int | None = None
    start: float | None = None
    end: float | None = None
    padded: bool = False
    region_index: int | None = None
    dataset: str = ""


def _safe_string(value: Any, default: str = "unavailable") -> str:
    """Convert missing/empty metadata into a collate-safe string."""
    if value is None:
        return default

    value = str(value).strip()

    if not value or value.lower() in {"none", "null", "nan"}:
        return default

    return value


def _safe_float(value: Any, default: float = -1.0) -> float:
    """Convert optional numeric metadata into a collate-safe float."""
    if value is None or value == "":
        return default

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    """Convert optional integer metadata into a collate-safe integer."""
    if value is None or value == "":
        return default

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def _label(value: Any) -> int:
    value = str(value).strip().lower()

    if value in {
        "real",
        "bona_fide",
        "bonafide",
        "genuine",
        "0",
    }:
        return 0

    if value in {
        "synthetic",
        "spoof",
        "fake",
        "1",
    }:
        return 1

    raise ValueError(f"unsupported label: {value}")


def read_manifest(path: str | Path) -> list[AudioRecord]:
    """Read a CSV dataset manifest."""

    path = Path(path)
    manifest_dir = path.resolve().parent

    records: list[AudioRecord] = []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            audio = (
                row.get("audio_path")
                or row.get("audio")
                or row.get("file")
            )

            if not audio:
                raise ValueError(
                    "manifest requires audio_path, audio, or file"
                )

            audio_path = Path(audio)

            if not audio_path.is_absolute():
                audio_path = manifest_dir / audio_path

            source_id = _safe_string(
                row.get("source_id")
                or row.get("source")
                or audio,
                default=str(audio),
            )

            sample_id = _safe_string(
                row.get("sample_id")
                or Path(audio).stem,
                default=Path(audio).stem,
            )

            records.append(
                AudioRecord(
                    audio_path=str(audio_path),
                    label=_label(row["label"]),
                    source_id=source_id,
                    speaker_id=_safe_string(
                        row.get("speaker_id")
                    ),
                    generator=_safe_string(
                        row.get("generator")
                    ),
                    split=_safe_string(
                        row.get("split"),
                        default="",
                    ),
                    sample_id=sample_id,
                    duration_seconds=(
                        _safe_float(
                            row.get("duration_seconds")
                        )
                        if row.get("duration_seconds")
                        else None
                    ),
                    chunk_id=(
                        _safe_int(row.get("chunk_id"))
                        if row.get("chunk_id")
                        else None
                    ),
                    start=(
                        _safe_float(row.get("start"))
                        if row.get("start")
                        else None
                    ),
                    end=(
                        _safe_float(row.get("end"))
                        if row.get("end")
                        else None
                    ),
                    padded=str(
                        row.get("padded", "false")
                    ).lower()
                    in {"true", "1", "yes"},
                    region_index=(
                        _safe_int(
                            row.get("region_index")
                        )
                        if row.get("region_index")
                        else None
                    ),
                    dataset=_safe_string(
                        row.get("dataset"),
                        default="",
                    ),
                )
            )

    return records


def read_module1_manifest(
    path: str | Path,
    label: int,
    source_id: str | None = None,
    speaker_id: str = "unavailable",
    generator: str = "unavailable",
) -> list[AudioRecord]:
    """Convert one Module 1 JSON manifest into Module 2B records."""

    path = Path(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    source = str(
        payload.get("source", path.stem)
    )

    resolved_source_id = (
        source_id
        or payload.get("source_id")
        or source
    )

    records: list[AudioRecord] = []

    for segment in payload.get("segments", []):
        relative_file = segment.get("file")

        if not relative_file:
            raise ValueError(
                "Module 1 segment is missing 'file'"
            )

        audio_path = str(
            path.parent / relative_file
        )

        chunk_id = segment.get("id")

        records.append(
            AudioRecord(
                audio_path=audio_path,
                label=label,
                source_id=_safe_string(
                    resolved_source_id,
                    default=source,
                ),
                speaker_id=_safe_string(
                    speaker_id
                ),
                generator=_safe_string(
                    generator
                ),
                sample_id=(
                    f"{Path(source).stem}"
                    f"_chunk_{chunk_id}"
                ),
                duration_seconds=(
                    _safe_float(
                        segment.get("duration")
                    )
                    if segment.get("duration") is not None
                    else None
                ),
                chunk_id=(
                    _safe_int(chunk_id)
                    if chunk_id is not None
                    else None
                ),
                start=(
                    _safe_float(segment.get("start"))
                    if segment.get("start") is not None
                    else None
                ),
                end=(
                    _safe_float(segment.get("end"))
                    if segment.get("end") is not None
                    else None
                ),
                padded=bool(
                    segment.get("padded", False)
                ),
                region_index=(
                    _safe_int(
                        segment.get("region_index")
                    )
                    if segment.get("region_index") is not None
                    else None
                ),
            )
        )

    return records


def discover_audio(
    root: str | Path,
) -> list[AudioRecord]:
    """Fallback discovery for independent audio datasets."""

    root = Path(root)

    records: list[AudioRecord] = []

    for label_name, label in (
        ("real", 0),
        ("synthetic", 1),
        ("bona_fide", 0),
        ("spoof", 1),
    ):
        directory = root / label_name

        if not directory.is_dir():
            continue

        for audio_path in sorted(
            directory.rglob("*")
        ):
            if audio_path.suffix.lower() not in {
                ".wav",
                ".flac",
                ".ogg",
                ".mp3",
            }:
                continue

            records.append(
                AudioRecord(
                    audio_path=str(audio_path),
                    label=label,
                    source_id=str(audio_path),
                    generator=label_name,
                    sample_id=audio_path.stem,
                )
            )

    return records


class SpectralDataset(Dataset):
    """
    Dataset returning:

        features: [F, T]
        label: scalar float
        metadata: dictionary

    Metadata is guaranteed to be collate-safe for PyTorch's
    default DataLoader collate function.
    """

    def __init__(
        self,
        records: list[AudioRecord],
        spectral_type: str = "logmel",
        training: bool = False,
        augment: dict | None = None,
        seed: int = 0,
    ) -> None:
        if not records:
            raise ValueError(
                "dataset is empty"
            )

        self.records = records

        self.extractor = SpectralExtractor(
            spectral_type
        )

        self.training = training
        self.augment = augment or {}
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ):
        record = self.records[index]

        loaded = load_audio(
            record.audio_path
        )

        audio = loaded.waveform

        if (
            len(audio) > WINDOW_SAMPLES
            and record.chunk_id is None
        ):
            raise ValueError(
                "generic/full-recording dataset records longer than "
                f"the Module 2B window ({WINDOW_SAMPLES} samples) "
                "require an explicit preparation/chunking step"
            )

        audio, padded_by_preparation = prepare_chunk(
            audio,
            WINDOW_SAMPLES,
        )

        # Never augment validation/test data.
        if (
            self.training
            and self.augment.get("enabled", False)
        ):
            rng = np.random.default_rng(
                self.seed + index
            )

            audio = augment_waveform(
                audio,
                rng,
                self.augment,
            )

        features = self.extractor(
            torch.from_numpy(audio)
        )

        # Convert every optional metadata field to a value that
        # PyTorch's default collate function can batch.
        metadata = {
            "audio_path": str(record.audio_path),
            "label": int(record.label),
            "source_id": _safe_string(
                record.source_id,
                default="unavailable",
            ),
            "speaker_id": _safe_string(
                record.speaker_id
            ),
            "generator": _safe_string(
                record.generator
            ),
            "split": _safe_string(
                record.split,
                default="",
            ),
            "sample_id": _safe_string(
                record.sample_id,
                default="unavailable",
            ),
            "duration_seconds": _safe_float(
                record.duration_seconds
            ),
            "chunk_id": _safe_int(
                record.chunk_id
            ),
            "start": _safe_float(
                record.start
            ),
            "end": _safe_float(
                record.end
            ),
            "padded": bool(
                record.padded
                or padded_by_preparation
            ),
            "region_index": _safe_int(
                record.region_index
            ),
            "dataset": _safe_string(
                record.dataset,
                default="",
            ),
        }

        return (
            features,
            torch.tensor(
                float(record.label),
                dtype=torch.float32,
            ),
            metadata,
        )


def write_manifest(
    records: list[AudioRecord],
    path: str | Path,
) -> None:
    """Write records as a CSV manifest."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(
        asdict(records[0]).keys()
    ) if records else [
        "audio_path",
        "label",
        "source_id",
        "speaker_id",
        "generator",
        "split",
        "sample_id",
        "duration_seconds",
        "chunk_id",
        "start",
        "end",
        "padded",
        "region_index",
        "dataset",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(record)
            )