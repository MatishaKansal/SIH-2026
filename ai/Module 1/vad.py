"""In-memory speech segmentation and sliding-window chunking with Silero VAD.

This module only detects speech activity.  It does not classify voices or
make authenticity, spoofing, or risk decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

from config import (
    HOP_SIZE_SECONDS,
    MIN_SILENCE_DURATION_MS,
    MIN_SPEECH_DURATION_MS,
    SAMPLE_RATE,
    SPEECH_PAD_MS,
    VAD_THRESHOLD,
    WINDOW_SIZE_SECONDS,
)


def normalize_audio(
    source: str | Path | np.ndarray,
    source_sample_rate: int | None = None,
    target_sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return a mono, float32 waveform at ``target_sample_rate`` in memory.

    A path is decoded with SoundFile.  An array source requires its sample rate
    so callers can use the same preprocessing path for future microphones.
    """
    if isinstance(source, (str, Path)):
        waveform, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    else:
        if source_sample_rate is None:
            raise ValueError("source_sample_rate is required when source is an array")
        waveform = np.asarray(source, dtype=np.float32)
        sample_rate = source_sample_rate
        if waveform.ndim == 1:
            waveform = waveform[:, None]
        elif waveform.ndim != 2:
            raise ValueError("audio array must have shape (samples,) or (samples, channels)")

    if sample_rate <= 0:
        raise ValueError("source sample rate must be positive")

    # SoundFile uses (samples, channels); averaging avoids silently keeping a
    # single arbitrary channel for stereo recordings.
    mono = np.ascontiguousarray(waveform.mean(axis=1), dtype=np.float32)
    if sample_rate != target_sample_rate:
        divisor = gcd(int(sample_rate), int(target_sample_rate))
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor)

    return np.ascontiguousarray(mono, dtype=np.float32)


@dataclass(frozen=True)
class SpeechRegion:
    """A contiguous speech interval from Silero, expressed in seconds."""

    start: float
    end: float
    audio: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "audio": self.audio}


class SileroVAD:
    """CPU Silero VAD wrapper with separate region detection and chunking APIs."""

    def __init__(self, *, use_onnx: bool = False) -> None:
        # This is the official silero-vad package loader.  ONNX can be enabled
        # for CPU deployments that have onnxruntime installed.
        self.model = load_silero_vad(onnx=use_onnx)
        self.sample_rate = SAMPLE_RATE

    def detect_regions(
        self,
        waveform: np.ndarray,
        *,
        threshold: float = VAD_THRESHOLD,
        min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
        speech_pad_ms: int = SPEECH_PAD_MS,
    ) -> list[dict[str, Any]]:
        """Return speech regions containing timestamps in seconds and waveforms."""
        audio = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32).squeeze())
        if audio.ndim != 1:
            raise ValueError("waveform must be mono (one-dimensional)")
        if not len(audio):
            return []

        timestamps = get_speech_timestamps(
            torch.from_numpy(audio),
            self.model,
            sampling_rate=self.sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,
        )
        regions: list[dict[str, Any]] = []
        for item in timestamps:
            start_sample, end_sample = int(item["start"]), int(item["end"])
            regions.append(
                SpeechRegion(
                    start=start_sample / self.sample_rate,
                    end=end_sample / self.sample_rate,
                    audio=audio[start_sample:end_sample].copy(),
                ).to_dict()
            )
        return regions

    def chunk_regions(
        self,
        regions: Iterable[dict[str, Any]],
        *,
        window_size_seconds: float = WINDOW_SIZE_SECONDS,
        hop_size_seconds: float = HOP_SIZE_SECONDS,
        pad_final_window: bool = True,
    ) -> list[dict[str, Any]]:
        """Create overlapping model windows from speech regions only.

        The final short window is zero-padded by default so later fixed-window
        models receive a consistent size. ``end`` remains the true source end;
        ``padded`` records whether padding was added.
        """
        window_samples = round(window_size_seconds * self.sample_rate)
        hop_samples = round(hop_size_seconds * self.sample_rate)
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window_size_seconds and hop_size_seconds must be positive")

        chunks: list[dict[str, Any]] = []
        for region_index, region in enumerate(regions):
            region_audio = np.asarray(region["audio"], dtype=np.float32)
            if not len(region_audio):
                continue
            for offset in range(0, len(region_audio), hop_samples):
                raw = region_audio[offset : offset + window_samples]
                if not len(raw):
                    continue
                if len(raw) < window_samples and not pad_final_window:
                    break
                padded = len(raw) < window_samples
                audio = np.pad(raw, (0, window_samples - len(raw))) if padded else raw.copy()
                start = float(region["start"]) + offset / self.sample_rate
                end = start + len(raw) / self.sample_rate
                chunks.append(
                    {
                        "start": start,
                        "end": end,
                        "audio": np.asarray(audio, dtype=np.float32),
                        "padded": padded,
                        "region_index": region_index,
                    }
                )
                if padded:
                    break
        return chunks

    def process_file(self, source: str | Path, **vad_options: Any) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
        """Normalize a file, detect speech regions, and return model chunks."""
        waveform = normalize_audio(source, target_sample_rate=self.sample_rate)
        regions = self.detect_regions(waveform, **vad_options)
        chunks = self.chunk_regions(regions)
        return waveform, regions, chunks


# A descriptive alias for callers who prefer the longer API name.
SileroVADDetector = SileroVAD
