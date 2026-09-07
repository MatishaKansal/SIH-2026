"""Configurable Module 2B training entry point."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_config import DEVICE, EMBEDDING_DIM
from dataset import (
    SpectralDataset,
    read_manifest,
)
from metrics import evaluate_scores
from model import (
    SpectralCNN,
    count_parameters,
)


def set_seed(seed: int) -> None:
    """Set deterministic random seeds where practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    optimizer=None,
    device="cpu",
):
    """
    Run one training or evaluation epoch.

    Returns:
        loss,
        metrics,
        labels,
        logits
    """

    training = optimizer is not None

    model.train(training)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses = []
    labels = []
    logits = []

    for features, target, _ in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            _, output = model(features)

            output = output.reshape(-1)
            target = target.reshape(-1)

            loss = loss_fn(
                output,
                target,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        losses.append(
            float(loss.detach().cpu())
        )

        labels.extend(
            target.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        logits.extend(
            output.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    if not losses:
        raise RuntimeError(
            "epoch received an empty DataLoader"
        )

    metrics = evaluate_scores(
        labels,
        logits,
    )

    return (
        float(np.mean(losses)),
        metrics,
        labels,
        logits,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Module 2B spectral CNN."
        )
    )

    parser.add_argument(
        "--split-dir",
        default="data/splits",
    )

    parser.add_argument(
        "--output",
        default=(
            "output/checkpoints/"
            "spectral_cnn.pt"
        ),
    )

    parser.add_argument(
        "--spectral-type",
        choices=[
            "stft",
            "logmel",
            "lfcc",
        ],
        default="logmel",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Enable train-only audio augmentation."
        ),
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.lr <= 0:
        raise ValueError(
            "--lr must be > 0"
        )

    set_seed(args.seed)

    device = torch.device(DEVICE)

    split_dir = Path(
        args.split_dir
    )

    def manifest_path(split: str, generated_name: str) -> Path:
        generated = split_dir / generated_name
        return generated if generated.exists() else split_dir / f"{split}.csv"

    train_manifest = manifest_path("train", "module2b_train.csv")
    validation_manifest = manifest_path("validation", "module2b_val.csv")
    test_manifest = manifest_path("test", "module2b_test.csv")

    for manifest in (
        train_manifest,
        validation_manifest,
        test_manifest,
    ):
        if not manifest.exists():
            raise FileNotFoundError(
                f"missing split manifest: {manifest}"
            )

    train_records = read_manifest(
        train_manifest
    )

    validation_records = read_manifest(
        validation_manifest
    )

    test_records = read_manifest(
        test_manifest
    )

    train_dataset = SpectralDataset(
        train_records,
        spectral_type=args.spectral_type,
        training=True,
        augment={
            "enabled": args.augment,
        },
        seed=args.seed,
    )

    validation_dataset = SpectralDataset(
        validation_records,
        spectral_type=args.spectral_type,
        training=False,
        augment=None,
        seed=args.seed,
    )

    test_dataset = SpectralDataset(
        test_records,
        spectral_type=args.spectral_type,
        training=False,
        augment=None,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    frequency_bins = {
        "stft": 257,
        "logmel": 64,
        "lfcc": 20,
    }[args.spectral_type]

    model = SpectralCNN(
        in_channels=frequency_bins,
        embedding_dim=args.embedding_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_score = -float("inf")
    best_epoch = None
    stale_epochs = 0

    history = []

    print(
        f"device={device}"
    )

    print(
        f"spectral_type={args.spectral_type}"
    )

    print(
        f"parameters={count_parameters(model)}"
    )

    print(
        f"train={len(train_dataset)} "
        f"validation={len(validation_dataset)} "
        f"test={len(test_dataset)}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        started = time.perf_counter()

        train_loss, train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_loss, validation_metrics, _, _ = (
            run_epoch(
                model,
                validation_loader,
                None,
                device,
            )
        )

        roc_auc = validation_metrics.get(
            "roc_auc"
        )

        if roc_auc is not None:
            validation_score = roc_auc
        else:
            validation_score = validation_metrics[
                "f1"
            ]

        elapsed = (
            time.perf_counter()
            - started
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "seconds": elapsed,
        }

        history.append(epoch_record)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_roc_auc="
            f"{validation_metrics.get('roc_auc')} "
            f"validation_f1="
            f"{validation_metrics['f1']:.4f} "
            f"seconds={elapsed:.2f}"
        )

        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "spectral_type": args.spectral_type,
                    "embedding_dim": args.embedding_dim,
                    "model_version": (
                        "2B-spectral-cnn"
                    ),
                    "sample_rate": 16_000,
                    "n_fft": 512,
                    "win_length": 400,
                    "hop_length": 160,
                    "n_mels": 64,
                    "n_lfcc": 20,
                    "best_validation_score": (
                        best_score
                    ),
                    "best_epoch": best_epoch,
                    "seed": args.seed,
                },
                output,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(
                "Early stopping triggered."
            )
            break

    # Reload the best checkpoint before final test evaluation.
    if not output.exists():
        raise RuntimeError(
            "training finished without "
            "creating a checkpoint"
        )

    checkpoint = torch.load(
        output,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    test_loss, test_metrics, _, _ = run_epoch(
        model,
        test_loader,
        None,
        device,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "parameter_count": count_parameters(
            model
        ),
        "checkpoint": str(output),
        "device": str(device),
        "spectral_type": args.spectral_type,
        "embedding_dim": args.embedding_dim,
        "seed": args.seed,
        "augmentation_enabled_for_training": (
            args.augment
        ),
        "history": history,
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()