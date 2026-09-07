"""Explicit, opt-in downloader for the local XLS-R model cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import MODEL_DIR, MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XLS-R once into the project's model cache")
    parser.add_argument("--confirm", action="store_true", help="Required: this downloads approximately 1.27 GB")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    if not args.confirm:
        parser.error("refusing a large download without --confirm")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=args.model_dir, allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"])
    print(f"Downloaded {MODEL_ID} to {args.model_dir}")


if __name__ == "__main__":
    main()
