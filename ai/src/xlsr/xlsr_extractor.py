"""Frozen Wav2Vec2 XLS-R embedding extraction shared by Module 2B.

This module intentionally exposes speech representations, not a spoof score or
probability. A downstream, separately trained fusion model may consume its
1024-dimensional embeddings later.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoProcessor, Wav2Vec2Model

MODEL_ID = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16_000
EMBEDDING_DIM = 1_024


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve an explicit device, using CUDA only when it is available."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def _as_mono_float32(waveform: np.ndarray | torch.Tensor) -> np.ndarray:
    """Validate finite audio and downmix a 1-D or 2-D waveform to mono."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    audio = np.asarray(waveform)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        # Audio APIs commonly use either (samples, channels) or (channels,
        # samples). The small dimension is interpreted as channels.
        channel_axis = 1 if audio.shape[1] <= audio.shape[0] else 0
        mono = audio.mean(axis=channel_axis)
    else:
        raise ValueError("waveform must have shape (samples,) or a two-dimensional audio shape")
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if not len(mono):
        raise ValueError("waveform must not be empty")
    if not np.isfinite(mono).all():
        raise ValueError("waveform must contain only finite values")
    return mono


def prepare_waveform(
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    target_sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Safely prepare independently supplied audio for XLS-R.

    Module 1 already gives mono 16 kHz Float32 audio, so that normal path is a
    validation-only no-op. Independent callers are downmixed and resampled
    without gain, peak, or loudness normalization.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    mono = _as_mono_float32(waveform)
    if sample_rate != target_sample_rate:
        divisor = gcd(int(sample_rate), int(target_sample_rate))
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor)
        mono = np.ascontiguousarray(mono, dtype=np.float32)
    return mono


class XLSREmbeddingExtractor:
    """Load XLS-R once and extract frozen, mean-pooled 1024-D embeddings."""

    def __init__(
        self,
        model_dir: str | Path = "models/xlsr-300m",
        *,
        device: str | torch.device = "auto",
        local_files_only: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"XLS-R model directory not found: {self.model_dir}")
        if not (self.model_dir / "pytorch_model.bin").is_file():
            raise FileNotFoundError(
                f"XLS-R weights are absent from {self.model_dir}; use the controlled download command first"
            )
        self.device = resolve_device(device)
        self.sample_rate = SAMPLE_RATE
        # The local snapshot intentionally contains only the acoustic
        # preprocessor, not an ASR vocabulary/tokenizer. AutoProcessor tries to
        # construct that absent tokenizer in current Transformers releases, so
        # fall back to the official feature extractor needed for embeddings.
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=local_files_only)
            self.processor_name = type(self.processor).__name__
        except (OSError, TypeError):
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_dir, local_files_only=local_files_only)
            self.processor_name = type(self.processor).__name__
        self.model = Wav2Vec2Model.from_pretrained(self.model_dir, local_files_only=local_files_only).to(self.device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)
        if self.embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"expected XLS-R hidden size {EMBEDDING_DIM}, got {self.embedding_dim}")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _processor_inputs(self, waveforms: Sequence[np.ndarray], valid_lengths: Sequence[int] | None) -> dict[str, torch.Tensor]:
        inputs = self.processor(
            list(waveforms),
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if valid_lengths is not None:
            if len(valid_lengths) != len(waveforms):
                raise ValueError("valid_lengths must match the batch size")
            attention_mask = inputs["attention_mask"]
            for index, (valid_length, audio) in enumerate(zip(valid_lengths, waveforms, strict=True)):
                if not 0 < int(valid_length) <= len(audio):
                    raise ValueError("each valid length must be between 1 and its waveform length")
                attention_mask[index, int(valid_length) :] = 0
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool only temporal hidden states, excluding padded tail frames."""
        feature_mask = self.model._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        feature_mask = feature_mask.unsqueeze(-1).to(hidden_states.dtype)
        denominator = feature_mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden_states * feature_mask).sum(dim=1) / denominator
        if pooled.ndim != 2 or pooled.shape[1] != EMBEDDING_DIM:
            raise AssertionError(f"expected (batch, {EMBEDDING_DIM}) pooled embeddings, got {tuple(pooled.shape)}")
        return pooled

    def extract_batch(
        self,
        waveforms: Sequence[np.ndarray | torch.Tensor],
        sample_rates: Sequence[int] | int = SAMPLE_RATE,
        *,
        valid_lengths: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return one Float32 ``(1024,)`` embedding per waveform as ``(B, 1024)``.

        ``valid_lengths`` enables Module 1 callers to exclude known zero-padded
        tail samples from temporal pooling while retaining their chunk record.
        """
        if not waveforms:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates] * len(waveforms)
        if len(sample_rates) != len(waveforms):
            raise ValueError("sample_rates must match the batch size")
        raw_lengths = [len(_as_mono_float32(audio)) for audio in waveforms]
        prepared = [prepare_waveform(audio, rate) for audio, rate in zip(waveforms, sample_rates, strict=True)]
        prepared_valid_lengths: list[int] | None = None
        if valid_lengths is not None:
            if len(valid_lengths) != len(waveforms):
                raise ValueError("valid_lengths must match the batch size")
            prepared_valid_lengths = []
            for valid_length, raw_length, prepared_audio, rate in zip(valid_lengths, raw_lengths, prepared, sample_rates, strict=True):
                if not 0 < int(valid_length) <= raw_length:
                    raise ValueError("each valid length must be between 1 and its waveform length")
                scaled_length = round(int(valid_length) * self.sample_rate / int(rate))
                prepared_valid_lengths.append(min(len(prepared_audio), max(1, scaled_length)))
        inputs = self._processor_inputs(prepared, prepared_valid_lengths)
        with torch.no_grad():
            hidden_states = self.model(**inputs).last_hidden_state
            pooled = self._pool(hidden_states, inputs["attention_mask"])
        result = pooled.detach().cpu().to(torch.float32).numpy()
        if result.shape != (len(prepared), EMBEDDING_DIM):
            raise AssertionError(f"unexpected embedding batch shape: {result.shape}")
        if not np.isfinite(result).all():
            raise FloatingPointError("XLS-R produced non-finite embeddings")
        return result

    def extract(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
        *,
        valid_length: int | None = None,
    ) -> np.ndarray:
        """Return a single frozen XLS-R embedding with shape ``(1024,)``."""
        batch = self.extract_batch([waveform], sample_rate, valid_lengths=None if valid_length is None else [valid_length])
        embedding = batch[0]
        if embedding.shape != (EMBEDDING_DIM,):
            raise AssertionError(f"unexpected embedding shape: {embedding.shape}")
        return embedding

    def predict(self, waveform: np.ndarray | torch.Tensor, sample_rate: int = SAMPLE_RATE, *, valid_length: int | None = None) -> dict[str, Any]:
        """Public fusion-ready interface; deliberately contains no P(AI)."""
        embedding = self.extract(waveform, sample_rate, valid_length=valid_length)
        return {"embedding": embedding, "embedding_dim": EMBEDDING_DIM, "model": MODEL_ID, "device": str(self.device)}


# Compatibility name for the earlier project wrapper. It remains an embedding
# extractor and now shares all contract validation with Module 2B.
class XLSRFeatureExtractor(XLSREmbeddingExtractor):
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        return self.extract(audio, self.sample_rate)
