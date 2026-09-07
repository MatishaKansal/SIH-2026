"""Configuration for the frozen XLS-R speech-embedding branch."""

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
MODEL_ID = "facebook/wav2vec2-xls-r-300m"
MODEL_DIR = PROJECT_ROOT / "models" / "xlsr-300m"
DEVICE = "auto"  # "auto", "cpu", "cuda", or e.g. "cuda:0"
SAMPLE_RATE = 16_000
EMBEDDING_DIM = 1_024
DEFAULT_OUTPUT_DIR = MODULE_DIR / "output" / "embeddings"
DEFAULT_MODULE1_DIR = PROJECT_ROOT / "Module 1" / "output" / "real_voice"
