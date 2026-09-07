"""User-facing entry point for one immutable pretrained AASIST baseline run."""

from __future__ import annotations

from pathlib import Path

from aasist import AASISTDetector
from config import BASELINE_OUTPUT_DIR, CALIBRATION_PATH, DEVICE, INPUT_AUDIO, SPOOF_THRESHOLD
from scoring import save_baseline_result, save_model_metadata


def classification_for_demo(spoof_score: float, threshold: float | None = SPOOF_THRESHOLD) -> str | None:
    """Return a label only when a validation-derived threshold is configured."""
    if threshold is None:
        return None
    return "SPOOF_LIKELY" if spoof_score >= threshold else "BONA_FIDE"


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        sources = [sys.argv[1]]
    else:
        # Default to available real and synthetic voice test files
        sources = [
            "../Module 1/input_audio/real_voice.wav",
            "../Module 1/input_audio/synthetic_voice.wav",
        ]

    detector = AASISTDetector(device=DEVICE, calibration_path=CALIBRATION_PATH)

    save_model_metadata(
        BASELINE_OUTPUT_DIR,
        checkpoint_path=detector.model_path,
        device=str(detector.device).upper(),
        sample_rate=detector.sample_rate,
        target_samples=detector.target_samples,
    )

    for audio_src in sources:
        source = Path(audio_src)
        if not source.is_absolute():
            source = Path(__file__).resolve().parent / source
        if not source.is_file():
            print(f"Audio source not found: {source}")
            continue

        scores, audio = detector.predict_file(source)
        calibrated_threshold = detector.calibrator.metadata.get("operating_threshold") if detector.calibrator else None
        if SPOOF_THRESHOLD is not None:
            label = classification_for_demo(float(scores["spoof_score"]), SPOOF_THRESHOLD)
        elif scores["spoof_probability"] is not None:
            label = classification_for_demo(float(scores["spoof_probability"]), calibrated_threshold)
        else:
            label = None

        result = {
            "module": "2A",
            "stage": "baseline",
            "model": "AASIST",
            "model_version": "pretrained",
            "input_file": str(source),
            "sample_rate": audio.sample_rate,
            "duration_seconds": audio.duration_seconds,
            **scores,
            "label": label,
        }
        result_path = save_baseline_result(BASELINE_OUTPUT_DIR, result)

        print("\n" + "=" * 50)
        print(f"MODULE 2A — AASIST BASELINE EVALUATION")
        print("=" * 50)
        print(f"Input File       : {source.name}")
        print(f"Sample Rate      : {audio.sample_rate} Hz")
        print(f"Duration         : {audio.duration_seconds:.2f} sec")
        print(f"Device           : {str(detector.device).upper()}")
        print("-" * 50)
        print("AASIST MODEL OUTPUTS:")
        print(f"  Raw Score (Bona-Fide logit) : {scores['raw_score']:+.6f}")
        print(f"  Spoof Score (Spoof - Real)  : {scores['spoof_score']:+.6f}")
        probability = scores["spoof_probability"]
        probability_text = "Uncalibrated baseline (logits provided)" if probability is None else f"{probability:.6f}"
        print(f"  Spoof Probability           : {probability_text}")
        print(f"  Classification              : {label if label else 'Not thresholded (Pretrained baseline)'}")
        print(f"Result Saved To  : {result_path}")
        print("=" * 50)


if __name__ == "__main__":
    main()
