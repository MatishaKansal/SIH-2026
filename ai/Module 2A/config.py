"""Configuration for Module 2A's immutable pretrained AASIST baseline."""

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

# Change only this value to evaluate a different Module 1 speech chunk.
INPUT_AUDIO = "../Module 1/output/real_voice/speech_000.wav"
DEVICE = "auto"  # "auto", "cpu", "cuda", or e.g. "cuda:0"
SPOOF_THRESHOLD = None  # No calibrated decision threshold exists for this baseline.
CALIBRATION_PATH = None  # Optional model-version-matched calibration artifact.

SAMPLE_RATE = 16_000
TARGET_SAMPLES = 64_600  # Official AASIST ASVspoof2019 evaluation input length.

# Keep the project's existing canonical checkpoint and architecture rather
# than creating a second copy of the 1.2 MB pretrained artifact.
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "aasist" / "AASIST.pth"
MODEL_CONFIG_PATH = PROJECT_ROOT / "src" / "aasist" / "AASIST.conf"
BASELINE_OUTPUT_DIR = MODULE_DIR / "output" / "baseline"
FINETUNED_OUTPUT_DIR = MODULE_DIR / "output" / "finetuned"
DATASET_OUTPUT_DIR = MODULE_DIR / "output" / "datasets"
