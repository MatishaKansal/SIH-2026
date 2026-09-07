"""Public chunk-level inference API for Module 2B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from spectral_config import DEVICE, WINDOW_SAMPLES
from extractor import SpectralExtractor
from input import normalize_audio, prepare_chunk
from model import SpectralCNN


class SpectralSpoofDetector:
    """
    Module 2B chunk-level spectral spoof detector.

    Input:
        one Module 1-compatible speech chunk

    Output:
        learned spectral embedding
        raw spoof logit
        uncalibrated sigmoid probability
    """

    FREQUENCY_BINS = {
        "stft": 257,
        "logmel": 64,
        "lfcc": 20,
    }

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        spectral_type: str = "logmel",
        embedding_dim: int = 64,
        device: str | None = None,
    ) -> None:
        if spectral_type not in self.FREQUENCY_BINS:
            raise ValueError(
                "spectral_type must be one of "
                f"{sorted(self.FREQUENCY_BINS)}"
            )

        self.device = torch.device(
            device or DEVICE
        )

        self.spectral_type = spectral_type

        self.extractor = SpectralExtractor(
            spectral_type
        )

        self.model = SpectralCNN(
            self.FREQUENCY_BINS[spectral_type],
            embedding_dim,
        )

        self.model_version = "untrained"

        if checkpoint is not None:
            checkpoint = Path(checkpoint)

            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"checkpoint not found: {checkpoint}"
                )

            payload = torch.load(
                checkpoint,
                map_location=self.device,
                weights_only=False,
            )

            state_dict = payload.get(
                "model",
                payload,
            )

            self.model.load_state_dict(
                state_dict
            )

            self.model_version = payload.get(
                "model_version",
                checkpoint.stem,
            )

        self.model.to(
            self.device
        )

        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        waveform,
        sample_rate: int,
    ) -> dict:
        """
        Predict spoof evidence for one chunk.

        `spoof_score` is the raw model logit.
        Larger values mean stronger spoof evidence.

        `uncalibrated_spoof_probability` is sigmoid(logit).
        It is NOT a calibrated probability.
        """

        normalized = normalize_audio(
            waveform,
            sample_rate,
        )

        audio, padded = prepare_chunk(
            normalized.waveform,
            WINDOW_SAMPLES,
        )

        features = self.extractor(
            torch.from_numpy(audio)
        ).unsqueeze(0)

        features = features.to(
            self.device
        )

        embedding, logits = self.model(
            features
        )

        logit = float(
            logits.reshape(-1)[0]
            .detach()
            .cpu()
            .item()
        )

        probability = float(
            torch.sigmoid(
                logits.reshape(-1)[0]
            )
            .detach()
            .cpu()
            .item()
        )

        return {
            "module": "2B",
            "spoof_logit": logit,
            "spoof_score": logit,
            "uncalibrated_spoof_probability": (
                probability
            ),
            "embedding": (
                embedding[0]
                .detach()
                .cpu()
                .tolist()
            ),
            "spectral_type": self.spectral_type,
            "model_version": self.model_version,
            "padded": padded,
        }

    def predict_manifest_record(
        self,
        manifest_path: str | Path,
        record: dict,
    ) -> dict:
        """
        Run inference on one Module 1 manifest segment
        while preserving its metadata.
        """

        manifest_path = Path(
            manifest_path
        )

        relative_file = record.get(
            "file"
        )

        if not relative_file:
            raise ValueError(
                "manifest record is missing 'file'"
            )

        chunk_path = (
            manifest_path.parent
            / relative_file
        )

        if not chunk_path.exists():
            raise FileNotFoundError(
                f"chunk not found: {chunk_path}"
            )

        waveform = _read_wave(
            chunk_path
        )

        result = self.predict(
            waveform,
            16_000,
        )

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        source = manifest.get(
            "source"
        )

        source_id = (
            manifest.get("source_id")
            or source
        )

        result.update(
            {
                "chunk_id": record.get(
                    "id"
                ),
                "source": source,
                "source_id": source_id,
                "start": record.get(
                    "start"
                ),
                "end": record.get(
                    "end"
                ),
                "duration": record.get(
                    "duration"
                ),
                "padded": record.get(
                    "padded",
                    result["padded"],
                ),
                "region_index": record.get(
                    "region_index"
                ),
            }
        )

        return result


def _read_wave(
    path: Path,
) -> np.ndarray:
    """Read audio as float32 while preserving channels."""

    import soundfile as sf

    waveform, _ = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    return waveform