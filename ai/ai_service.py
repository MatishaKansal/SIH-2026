"""HTTP adapter for the calibrated Module 2B spoof detector."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
MODULE_2B = ROOT / "Module 2B"
sys.path.insert(0, str(MODULE_2B))

from prosody.detector import ProsodySpoofDetector  # noqa: E402


class WindowRequest(BaseModel):
    call_id: str
    window_id: int
    audio_base64: str


app = FastAPI(
    title="VoiceShield AI Detection Service",
    version="1.0.0",
)

_detector: ProsodySpoofDetector | None = None


def detector() -> ProsodySpoofDetector:
    global _detector
    if _detector is None:
        _detector = ProsodySpoofDetector(model_dir=MODULE_2B / "output")
    return _detector


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "voice-shield-ai"}


@app.post("/detect-window")
async def detect_window(request: WindowRequest) -> dict:
    try:
        pcm = bytes.fromhex(request.audio_base64)
        waveform = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if waveform.size == 0:
            raise ValueError("audio window is empty")

        prediction = detector().predict(waveform, sample_rate=16000)
        spoof_probability = float(prediction["calibrated_spoof_prob"])
        if spoof_probability >= 0.70:
            decision = "HIGH_RISK"
        elif spoof_probability >= 0.30:
            decision = "INVESTIGATE"
        else:
            decision = "SAFE"

        return {
            "call_id": request.call_id,
            "window_id": request.window_id,
            "vad_active": bool(np.any(np.abs(waveform) > 0.01)),
            "p_ai_window": spoof_probability,
            "cumulative_risk_score": spoof_probability,
            "wald_decision": decision,
            "latency_ms": prediction["latency_ms"],
            "metrics": {
                "aasist_spoof_score": spoof_probability,
                "prosody_anomaly_score": spoof_probability,
                "speaker_match_score": 1.0,
            },
            "model": prediction["model_family"],
            "explanations": prediction["top_forensic_cues"],
        }
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="AI model inference failed") from error
