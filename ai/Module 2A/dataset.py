"""Manifest-driven data access for local files and embedded Parquet audio."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import prepare_for_aasist, preprocess_audio, preprocess_audio_bytes


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    label: int  # bona-fide=0, spoof=1
    split: str
    group_id: str
    audio_path: str = ""
    parquet_path: str = ""
    row_group: int = -1
    row_in_group: int = -1
    audio_column: str = ""


def read_manifest(path: str | Path, *, split: str | None = None) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "label", "split", "group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest must contain columns: {sorted(required)}")
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"invalid label for {row['sample_id']}: {label}")
            record = ManifestRecord(
                sample_id=row["sample_id"], label=label, split=row["split"], group_id=row["group_id"],
                audio_path=row.get("audio_path", ""), parquet_path=row.get("parquet_path", ""),
                row_group=int(row.get("row_group", -1) or -1),
                row_in_group=int(row.get("row_in_group", -1) or -1), audio_column=row.get("audio_column", ""),
            )
            if not record.audio_path and not record.parquet_path:
                raise ValueError(f"{record.sample_id} has neither audio_path nor parquet_path")
            records.append(record)
    return records


class AudioManifestDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Loads audio lazily; validation/test stay deterministic and unaugmented."""

    def __init__(
        self,
        records: list[ManifestRecord],
        *,
        training: bool = False,
        augment: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        if not records:
            raise ValueError("dataset split is empty")
        self.records = records
        self.training = training
        self.augment = augment or {}
        self.seed = seed
        self.epoch = 0
        self._parquet_cache: dict[tuple[str, int], Any] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic augmentation/crop seed for the next epoch."""
        self.epoch = epoch

    def _embedded_audio(self, record: ManifestRecord) -> bytes:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow is required for embedded Parquet audio") from exc
        key = (record.parquet_path, record.row_group)
        if key not in self._parquet_cache:
            self._parquet_cache[key] = pq.ParquetFile(record.parquet_path).read_row_group(record.row_group)
        row = self._parquet_cache[key].slice(record.row_in_group, 1).to_pylist()[0]
        audio = row[record.audio_column]
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            raise ValueError(f"embedded audio missing for {record.sample_id}")
        return data

    def _augment(self, waveform: np.ndarray, index: int) -> np.ndarray:
        if not self.training or not self.augment.get("enabled", False):
            return waveform
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        audio = waveform.copy()
        if rng.random() < float(self.augment.get("gain_probability", 0.0)):
            db = rng.uniform(*self.augment.get("gain_db_range", [-4.0, 4.0]))
            audio *= np.float32(10 ** (db / 20))
        if rng.random() < float(self.augment.get("noise_probability", 0.0)):
            snr = rng.uniform(*self.augment.get("snr_db_range", [20.0, 40.0]))
            rms = max(float(np.sqrt(np.mean(audio**2))), 1e-7)
            noise_rms = rms / (10 ** (snr / 20))
            audio += rng.normal(0, noise_rms, size=audio.shape).astype(np.float32)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        if record.audio_path:
            processed = preprocess_audio(record.audio_path)
        else:
            processed = preprocess_audio_bytes(self._embedded_audio(record))
        waveform = self._augment(processed.waveform, index)
        prepared = prepare_for_aasist(
            waveform,
            random_crop=self.training,
            rng=np.random.default_rng(self.seed + self.epoch * 1_000_003 + index),
        )
        return torch.from_numpy(prepared), record.label, record.sample_id
