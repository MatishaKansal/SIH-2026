"""Audio validation and deterministic preparation for pretrained AASIST."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import SAMPLE_RATE, TARGET_SAMPLES


@dataclass(frozen=True)
class PreprocessedAudio:
    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.asarray(array.mean(axis=1), dtype=np.float32)
    raise ValueError("audio must have shape (samples,) or (samples, channels)")


def normalize_waveform(
    waveform: np.ndarray,
    source_sample_rate: int,
    *,
    target_sample_rate: int = SAMPLE_RATE,
) -> PreprocessedAudio:
    """Validate and normalize a decoded waveform without duplicating I/O code."""
    if source_sample_rate <= 0:
        raise ValueError("source sample rate must be positive")
    mono = _to_mono_float32(waveform)
    if mono.size == 0:
        raise ValueError("audio is empty")
    if not np.isfinite(mono).all():
        raise ValueError("audio contains NaN or Inf")
    if source_sample_rate != target_sample_rate:
        divisor = gcd(int(source_sample_rate), int(target_sample_rate))
        mono = resample_poly(mono, target_sample_rate // divisor, source_sample_rate // divisor)
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if mono.size == 0:
        raise ValueError("audio is empty after resampling")
    return PreprocessedAudio(mono, target_sample_rate, len(mono) / target_sample_rate)


def preprocess_audio(
    source: str | Path | np.ndarray,
    source_sample_rate: int | None = None,
    *,
    target_sample_rate: int = SAMPLE_RATE,
) -> PreprocessedAudio:
    """Load/normalize an audio source to finite mono float32 at 16 kHz."""
    if isinstance(source, (str, Path)):
        waveform, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    else:
        if source_sample_rate is None:
            raise ValueError("source_sample_rate is required for an array source")
        waveform, sample_rate = source, source_sample_rate

    return normalize_waveform(waveform, sample_rate, target_sample_rate=target_sample_rate)


def preprocess_audio_bytes(
    audio_bytes: bytes,
    *,
    target_sample_rate: int = SAMPLE_RATE,
) -> PreprocessedAudio:
    """Decode embedded Parquet audio and reuse the same normalization path."""
    if not audio_bytes:
        raise ValueError("embedded audio bytes are empty")
    waveform, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=True, dtype="float32")
    return normalize_waveform(waveform, sample_rate, target_sample_rate=target_sample_rate)


def prepare_for_aasist(
    waveform: np.ndarray,
    target_samples: int = TARGET_SAMPLES,
    *,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pad/repeat short audio and optionally random-crop long training examples."""
    audio = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32).squeeze())
    if audio.ndim != 1 or not len(audio):
        raise ValueError("AASIST requires a non-empty mono waveform")
    if len(audio) >= target_samples:
        if random_crop and len(audio) > target_samples:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, len(audio) - target_samples + 1))
            return audio[start : start + target_samples].copy()
        return audio[:target_samples].copy()
    repeats = (target_samples + len(audio) - 1) // len(audio)
    return np.ascontiguousarray(np.tile(audio, repeats)[:target_samples], dtype=np.float32)
