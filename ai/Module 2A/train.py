"""Reproducible, staged fine-tuning of pretrained AASIST on a frozen manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_PATH, FINETUNED_OUTPUT_DIR
from dataset import AudioManifestDataset, read_manifest
from evaluate import evaluate_model, write_predictions
from model_utils import checkpoint_model_sha256, load_aasist_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_training_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def class_weights(records: list[Any], device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency CE weights in AASIST output index order.

    Project labels: bonafide=0, spoof=1.
    AASIST output indices: spoof=0, bonafide=1.
    CrossEntropyLoss weight[i] applies to target class i (AASIST index),
    so we must return [weight_for_spoof, weight_for_bonafide].
    """
    # counts[0] = bonafide count, counts[1] = spoof count (project convention)
    counts = np.bincount([record.label for record in records], minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"training split must contain both classes; got {counts.tolist()}")
    # Inverse-frequency weights in project order: [w_bonafide, w_spoof]
    project_weights = counts.sum() / (2 * counts.astype(float))
    # Flip to AASIST output order: [w_spoof, w_bonafide]
    aasist_weights = project_weights[::-1].copy()
    return torch.tensor(aasist_weights, dtype=torch.float32, device=device)


def configure_trainable_parameters(model: torch.nn.Module, prefixes: list[str]) -> list[str]:
    """Allow a head-only stage before full-model fine-tuning without new code."""
    selected: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not prefixes or any(name.startswith(prefix) for prefix in prefixes)
        if parameter.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError("trainable_parameter_prefixes selected no AASIST parameters")
    return selected


def save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained AASIST; calibration/test splits remain untouched")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("training_config.json"))
    parser.add_argument("--output-dir", type=Path, default=FINETUNED_OUTPUT_DIR)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    set_seed(int(cfg["seed"]))

    print("="*60)
    print(" AASIST Fine-tuning")
    print("="*60)
    sys.stdout.flush()

    print("\n[1/5] Reading manifest splits...")
    sys.stdout.flush()
    t0 = time.time()
    train_records = read_manifest(args.manifest, split="train")
    validation_records = read_manifest(args.manifest, split="validation")
    # Load these now to fail early if the required frozen holdouts are absent;
    # neither split is used for model fitting in this command.
    calibration_records = read_manifest(args.manifest, split="calibration")
    test_records = read_manifest(args.manifest, split="test")
    print(f"  train={len(train_records):,}  val={len(validation_records):,}  "
          f"cal={len(calibration_records):,}  test={len(test_records):,}  "
          f"({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print("\n[2/5] Loading AASIST model...")
    sys.stdout.flush()
    t0 = time.time()
    start_checkpoint = args.resume or args.pretrained_checkpoint
    model, device = load_aasist_model(start_checkpoint, device=args.device)
    trainable_names = configure_trainable_parameters(model, list(cfg.get("trainable_parameter_prefixes", [])))
    print(f"  device={device}  trainable_params={len(trainable_names):,}  ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_records, device))
    use_amp = bool(cfg.get("mixed_precision", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("\n[3/5] Creating data loaders...")
    print(f"  batch_size={cfg['batch_size']}  num_workers={cfg['num_workers']}  augmentation={'ON' if cfg.get('augmentation', {}).get('enabled') else 'OFF'}")
    sys.stdout.flush()
    train_dataset = AudioManifestDataset(train_records, training=True, augment=cfg.get("augmentation"), seed=int(cfg["seed"]))
    validation_dataset = AudioManifestDataset(validation_records, training=False)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda")
    num_train_batches = len(train_loader)
    num_val_batches = len(validation_loader)
    print(f"  train_batches={num_train_batches:,}  val_batches={num_val_batches:,}")
    print("  (NOTE: First epoch may be slow while workers load & resample audio from Parquet)")
    sys.stdout.flush()

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    stale_epochs = 0
    initial_epoch = 0
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        initial_epoch = int(resume["epoch"]) + 1
        best_loss = float(resume.get("best_validation_loss", best_loss))

    total_epochs = int(cfg["num_epochs"])
    print(f"\n[4/5] Training  (epochs {initial_epoch+1}..{total_epochs}, patience={cfg['patience']}, lr={cfg['learning_rate']})")
    print("-"*60)
    sys.stdout.flush()

    for epoch in range(initial_epoch, total_epochs):
        epoch_start = time.time()
        model.train()
        train_dataset.set_epoch(epoch)
        batch_losses: list[float] = []

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]",
            unit="batch", leave=False, file=sys.stdout,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        for audio, labels, _ in train_bar:
            audio, labels = audio.to(device), labels.to(device)
            # Remap project labels (bonafide=0, spoof=1) to AASIST output
            # indices (spoof=0, bonafide=1) for CrossEntropyLoss targets.
            ce_labels = 1 - labels
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, logits = model(audio)
                loss = criterion(logits, ce_labels)
            scaler.scale(loss).backward()
            if cfg.get("gradient_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.item()))
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_bar = tqdm(
            validation_loader, desc=f"Epoch {epoch+1}/{total_epochs} [val]  ",
            unit="batch", leave=False, file=sys.stdout,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        # Wrap the validation DataLoader with tqdm for visibility
        model.eval()
        val_rows: list[dict[str, Any]] = []
        val_losses: list[float] = []
        with torch.inference_mode():
            for audio, labels_t, sample_ids in val_bar:
                audio = audio.to(device, non_blocking=True)
                labels_t = labels_t.to(device, non_blocking=True)
                # Remap for CE loss (same as training)
                ce_labels_val = 1 - labels_t
                _, logits = model(audio)
                val_loss_batch = criterion(logits, ce_labels_val)
                val_losses.append(float(val_loss_batch.item()))
                val_bar.set_postfix(loss=f"{val_loss_batch.item():.4f}")
                logits_np = logits.detach().cpu().numpy()
                # Use original project labels (not remapped) for metric computation
                labels_np = labels_t.detach().cpu().numpy()
                scores = logits_np[:, 0] - logits_np[:, 1]
                for idx_s, sid in enumerate(sample_ids):
                    val_rows.append({"label": int(labels_np[idx_s]), "spoof_score": float(scores[idx_s])})

        from metrics import binary_metrics
        val_labels = np.array([r["label"] for r in val_rows])
        val_scores = np.array([r["spoof_score"] for r in val_rows])
        validation_metrics = binary_metrics(val_labels, val_scores, threshold=None, probabilities=None)
        validation_metrics["mean_loss"] = float(np.mean(val_losses)) if val_losses else None

        validation_loss = float(validation_metrics["mean_loss"])
        scheduler.step(validation_loss)
        train_loss = float(np.mean(batch_losses))
        lr_now = float(optimizer.param_groups[0]["lr"])
        summary = {
            "epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
            "validation_roc_auc": validation_metrics["roc_auc"],
            "learning_rate": lr_now,
        }
        history.append(summary)
        payload = {
            "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "best_validation_loss": best_loss,
            "pretrained_checkpoint": str(args.pretrained_checkpoint.resolve()),
            "pretrained_checkpoint_sha256": checkpoint_model_sha256(args.pretrained_checkpoint),
            "training_config": cfg, "manifest": str(args.manifest.resolve()),
        }
        if (epoch + 1) % int(cfg["checkpoint_interval"]) == 0:
            save_checkpoint(checkpoint_dir / "last_model.pth", **payload)
        marker = "  "
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            payload["best_validation_loss"] = best_loss
            save_checkpoint(checkpoint_dir / "best_model.pth", **payload)
            marker = "✓ best"
        else:
            stale_epochs += 1
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1:>3}/{total_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={validation_loss:.4f}  "
              f"AUC={validation_metrics['roc_auc']:.4f}  "
              f"lr={lr_now:.2e}  {marker}  ({elapsed:.0f}s)")
        sys.stdout.flush()
        if stale_epochs >= int(cfg["patience"]):
            print(f"\n  ⏹ Early stopping after {epoch + 1} epochs (no improvement for {cfg['patience']} epochs).")
            break

    print("\n" + "="*60)
    print(" [5/5] Saving metadata")
    print("="*60)
    run_metadata = {
        "completed_at": datetime.now(timezone.utc).isoformat(), "device": str(device), "manifest": str(args.manifest.resolve()),
        "pretrained_checkpoint": str(args.pretrained_checkpoint.resolve()), "pretrained_checkpoint_sha256": checkpoint_model_sha256(args.pretrained_checkpoint),
        "split_sizes": {"train": len(train_records), "validation": len(validation_records), "calibration": len(calibration_records), "test": len(test_records)},
        "class_balance_train": {"bona_fide": sum(x.label == 0 for x in train_records), "spoof": sum(x.label == 1 for x in train_records)},
        "trainable_parameter_prefixes": cfg.get("trainable_parameter_prefixes", []), "trainable_parameter_count": len(trainable_names),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "best_checkpoint": str((checkpoint_dir / "best_model.pth").resolve()),
        "note": "Calibration and test data were not used by training. Run evaluate.py and calibrate.py separately after training.",
    }
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_metadata, indent=2))
    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
