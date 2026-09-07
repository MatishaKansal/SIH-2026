"""Shared AASIST construction/checkpoint loading for evaluation and training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from aasist import resolve_device
from config import CHECKPOINT_PATH, MODEL_CONFIG_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.aasist.aasist_model import Model  # noqa: E402


def load_aasist_model(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    *,
    device: str = "auto",
    config_path: str | Path = MODEL_CONFIG_PATH,
) -> tuple[Model, torch.device]:
    """Load either the original state dict or a Module 2A training checkpoint."""
    selected_device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path)
    model_config = json.loads(Path(config_path).read_text(encoding="utf-8"))["model_config"]
    payload: Any = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model = Model(model_config).to(selected_device)
    model.load_state_dict(state_dict, strict=True)
    return model, selected_device


def checkpoint_model_sha256(path: str | Path) -> str:
    from scoring import sha256_file
    return sha256_file(path)
