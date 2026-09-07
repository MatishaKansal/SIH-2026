"""Audio normalization shared by Module 2B datasets and inference."""
from __future__ import annotations
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any
import numpy as np
from scipy.signal import resample_poly

@dataclass(frozen=True)
class NormalizedAudio:
    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float

def _mono(waveform: np.ndarray) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.asarray(audio.mean(axis=1), dtype=np.float32)
    raise ValueError("audio must have shape (samples,) or (samples, channels)")

def normalize_audio(source: str | Path | np.ndarray, source_sample_rate: int | None = None, *, target_sample_rate: int = 16_000) -> NormalizedAudio:
    if isinstance(source, (str, Path)):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("soundfile is required to load audio paths") from exc
        waveform, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    else:
        if source_sample_rate is None:
            raise ValueError("source_sample_rate is required for an array")
        waveform, sample_rate = source, source_sample_rate
    if int(sample_rate) <= 0:
        raise ValueError("sample rate must be positive")
    mono = _mono(waveform)
    if not mono.size:
        raise ValueError("audio is empty")
    if not np.isfinite(mono).all():
        raise ValueError("audio contains NaN or Inf")
    if int(sample_rate) != target_sample_rate:
        divisor = gcd(int(sample_rate), target_sample_rate)
        mono = resample_poly(mono, target_sample_rate // divisor, int(sample_rate) // divisor)
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    return NormalizedAudio(mono, target_sample_rate, len(mono) / target_sample_rate)

def load_audio(source: str | Path) -> NormalizedAudio:
    return normalize_audio(source)

def pad_or_trim(waveform: np.ndarray, samples: int = 16_000) -> tuple[np.ndarray, bool]:
    audio = _mono(waveform)
    if not audio.size:
        raise ValueError("audio is empty")
    if len(audio) >= samples:
        return np.ascontiguousarray(audio[:samples], dtype=np.float32), False
    return np.pad(audio, (0, samples - len(audio))).astype(np.float32), True

def prepare_chunk(waveform: np.ndarray, samples: int = 16_000) -> tuple[np.ndarray, bool]:
    """Prepare one Module 1 window; reject whole recordings passed as chunks."""
    audio = _mono(waveform)
    if not audio.size:
        raise ValueError("chunk is empty")
    if len(audio) > samples:
        raise ValueError(f"Module 1 chunk must be at most {samples} samples, got {len(audio)}")
    if len(audio) == samples:
        return np.ascontiguousarray(audio, dtype=np.float32), False
    return np.pad(audio, (0, samples - len(audio))).astype(np.float32), True

def write_wave(path: str | Path, waveform: np.ndarray, sample_rate: int = 16_000) -> None:
    import soundfile as sf
    sf.write(str(path), np.asarray(waveform, dtype=np.float32), sample_rate)
