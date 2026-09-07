"""Configuration for the independent Module 2B spectral branch."""

from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

# Audio
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE  # 1 second

# Spectral extraction
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160

N_MELS = 64
N_LFCC = 20

EPSILON = 1e-6

# Model
EMBEDDING_DIM = 64


def default_device() -> str:
    """Prefer CUDA, then Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = default_device()