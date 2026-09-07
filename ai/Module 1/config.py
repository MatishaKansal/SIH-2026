"""Configuration for the standalone Silero VAD module."""

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent

# Audio passed to Silero is always normalized to this format.
SAMPLE_RATE = 16_000
CHANNELS = 1

# Silero speech-region detection settings.
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 100
SPEECH_PAD_MS = 30

# Windows emitted for later pipeline modules.  Change these (for example to
# 2.0 and 1.0) without changing the VAD implementation.
WINDOW_SIZE_SECONDS = 1.0
HOP_SIZE_SECONDS = 0.5

# Official Silero artifacts may be placed here for offline/reference use. The
# Python package remains the default loader because it is Silero's supported
# loading interface.
MODEL_DIR = MODULE_DIR / "models" / "silero_vad"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "output"
