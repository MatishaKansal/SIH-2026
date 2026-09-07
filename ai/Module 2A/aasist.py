"""Reusable CPU/CUDA inference wrapper for the immutable pretrained AASIST."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calibration import load_calibrator
from config import CHECKPOINT_PATH, MODEL_CONFIG_PATH, SAMPLE_RATE, TARGET_SAMPLES
from preprocessing import prepare_for_aasist, preprocess_audio
from scoring import scores_from_logits, sha256_file

# Reuse the repository's official AASIST architecture, rather than duplicating
# model code inside this baseline experiment.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.aasist.aasist_model import Model  # noqa: E402


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


class AASISTDetector:
    """Evaluation-only pretrained AASIST inference. No training state is created."""

    def __init__(
        self,
        model_path: str | Path = CHECKPOINT_PATH,
        *,
        config_path: str | Path = MODEL_CONFIG_PATH,
        device: str = "auto",
        calibration_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.config_path = Path(config_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Pretrained AASIST checkpoint not found: {self.model_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"AASIST model configuration not found: {self.config_path}")
        self.device = resolve_device(device)
        self.sample_rate = SAMPLE_RATE
        self.target_samples = TARGET_SAMPLES

        model_config = json.loads(self.config_path.read_text(encoding="utf-8"))["model_config"]
        self.model = Model(model_config).to(self.device)
        payload = torch.load(self.model_path, map_location=self.device, weights_only=True)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.calibrator = (
            load_calibrator(calibration_path, expected_checkpoint_sha256=sha256_file(self.model_path))
            if calibration_path is not None
            else None
        )

    def predict(self, waveform: np.ndarray) -> dict[str, float | None]:
        """Predict raw and spoof-direction evidence from a normalized waveform."""
        prepared = prepare_for_aasist(waveform, self.target_samples)
        batch = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            _, logits = self.model(batch)
        scores = scores_from_logits(logits[0])
        if self.calibrator is not None:
            scores["spoof_probability"] = float(self.calibrator.predict_proba([scores["spoof_score"]])[0])
        return scores

    def predict_file(self, audio_path: str | Path) -> tuple[dict[str, float | None], Any]:
        """Preprocess a file, then return score values and audio metadata."""
        audio = preprocess_audio(audio_path, target_sample_rate=self.sample_rate)
        return self.predict(audio.waveform), audio
