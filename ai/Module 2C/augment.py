"""Small training-only waveform augmentations."""
from __future__ import annotations
import numpy as np

def augment_waveform(waveform: np.ndarray, rng: np.random.Generator, config: dict) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32).copy()
    if rng.random() < float(config.get("gain_probability", 0.0)):
        audio *= np.float32(10 ** rng.uniform(-4.0, 4.0) / 20)
    if rng.random() < float(config.get("noise_probability", 0.0)):
        rms = max(float(np.sqrt(np.mean(audio ** 2))), 1e-7)
        snr = rng.uniform(20.0, 40.0)
        audio += rng.normal(0, rms / (10 ** (snr / 20)), audio.shape).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
