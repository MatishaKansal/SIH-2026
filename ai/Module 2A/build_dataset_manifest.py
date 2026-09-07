"""Create a reproducible, non-overlapping train/calibration/test manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import DATASET_OUTPUT_DIR, PROJECT_ROOT

FIELDS = ["sample_id", "label", "split", "group_id", "audio_path", "parquet_path", "row_group", "row_in_group", "audio_column", "source_dataset"]


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _assign_grouped_splits(records: list[dict[str, Any]], fractions: tuple[float, float, float], seed: int) -> None:
    if not abs(sum(fractions) - 1.0) < 1e-8:
        raise ValueError("split fractions must sum to 1")
    names = ("train", "validation", "calibration", "test")
    for label in (0, 1):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["label"] == label:
                groups[record["group_id"]].append(record)
        if len(groups) < 4:
            raise ValueError(f"label {label} has fewer than four groups; cannot create leakage-safe four-way split")
        ordered = sorted(groups, key=lambda value: _stable_order(value, seed))
        target = [sum(len(groups[group]) for group in ordered) * fraction for fraction in fractions]
        totals = [0, 0, 0, 0]
        for group in ordered:
            index = min(range(4), key=lambda i: (totals[i] / max(target[i], 1), totals[i]))
            for record in groups[group]:
                record["split"] = names[index]
            totals[index] += len(groups[group])


def _rows_from_parquet(root: Path, *, label: int, dataset_name: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    records: list[dict[str, Any]] = []
    for parquet_path in sorted(root.glob("*.parquet")):
        file = pq.ParquetFile(parquet_path)
        audio_column = "audio_filepath" if label == 0 else "audio"
        metadata_columns = ["native_place_district", "primary_language", "gender", "age-group"] if label == 0 else ["source"]
        for row_group in range(file.num_row_groups):
            table = file.read_row_group(row_group, columns=metadata_columns)
            for row_index, row in enumerate(table.to_pylist()):
                sample_id = f"{dataset_name}:{parquet_path.name}:rg{row_group}:r{row_index}"
                if label == 0:
                    # Svarah has no speaker ID in its local schema. This groups
                    # by the strongest available demographic/location proxy.
                    group_id = "svarah:" + "|".join(str(row[key]) for key in metadata_columns)
                else:
                    # Orpheus exposes source, but only two values. Use a unique
                    # recording group to prevent any recording/chunk overlap;
                    # source-independent evaluation needs richer source IDs.
                    group_id = f"orpheus:recording:{sample_id}"
                records.append({
                    "sample_id": sample_id, "label": label, "split": "", "group_id": group_id,
                    "audio_path": "", "parquet_path": str(parquet_path.resolve()), "row_group": row_group,
                    "row_in_group": row_index, "audio_column": audio_column, "source_dataset": dataset_name,
                })
    return records


def _rows_from_indic_audio(root: Path, *, dataset_name: str = "IndicAudio") -> list[dict[str, Any]]:
    """Load indic-audio metadata.jsonl entries as spoof records with voice-based grouping.

    Each clip is synthesized by ElevenLabs persona design + Fish Audio S2 Pro
    voice cloning. Grouping by voice persona ensures no persona leaks across
    train/validation/calibration/test splits.
    """
    metadata_path = root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"indic-audio metadata not found: {metadata_path}")
    records: list[dict[str, Any]] = []
    with metadata_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            entry = json.loads(line)
            audio_rel = entry["audio"]  # e.g. "audio/bed_hindi/0012.wav"
            voice = entry.get("voice", "unknown")
            audio_abs = (root / audio_rel).resolve()
            if not audio_abs.is_file() or audio_abs.stat().st_size <= 1000:
                continue  # skip missing or un-downloaded LFS pointer files
            sample_id = f"{dataset_name}:{voice}:{index:05d}"
            # Group by synthetic persona for leakage-safe splitting: ensures
            # the same persona across languages (e.g. en_aman and hing_aman)
            # stays in the same split with zero cross-split persona leakage.
            persona = voice if voice == "bed_hindi" else (voice.split("_", 1)[1] if "_" in voice else voice)
            group_id = f"indic:{persona}"
            records.append({
                "sample_id": sample_id, "label": 1, "split": "", "group_id": group_id,
                "audio_path": str(audio_abs), "parquet_path": "", "row_group": -1,
                "row_in_group": -1, "audio_column": "", "source_dataset": dataset_name,
            })
    return records


def build_builtin_manifest(output_path: Path, *, seed: int = 1337, include_indic_audio: bool = True) -> dict[str, Any]:
    real = _rows_from_parquet(PROJECT_ROOT / "data" / "realvoice" / "Svarah" / "data", label=0, dataset_name="Svarah")
    spoof = _rows_from_parquet(PROJECT_ROOT / "data" / "syntheticvoice" / "orpheus_tts_english_indian_multispeaker" / "data", label=1, dataset_name="Orpheus")
    indic_count = 0
    if include_indic_audio:
        indic_root = PROJECT_ROOT / "data" / "syntheticvoice" / "indic-audio"
        if (indic_root / "metadata.jsonl").is_file():
            indic = _rows_from_indic_audio(indic_root, dataset_name="IndicAudio")
            indic_count = len(indic)
            spoof = spoof + indic
    records = real + spoof
    _assign_grouped_splits(records, (0.70, 0.10, 0.10, 0.10), seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    report = {
        "manifest": str(output_path.resolve()), "seed": seed, "total_records": len(records),
        "class_counts": {"bona_fide": len(real), "spoof": len(spoof)},
        "spoof_source_counts": {
            "Orpheus": len(spoof) - indic_count,
            "IndicAudio": indic_count,
        },
        "split_class_counts": {
            split: {"bona_fide": sum(x["split"] == split and x["label"] == 0 for x in records), "spoof": sum(x["split"] == split and x["label"] == 1 for x in records)}
            for split in ("train", "validation", "calibration", "test")
        },
        "group_counts": {"bona_fide": len({x["group_id"] for x in real}), "spoof": len({x["group_id"] for x in spoof})},
        "grouping_limitations": [
            "Svarah local Parquet schema has no speaker ID; geographic/demographic proxy groups are used.",
            "Orpheus local schema has only two source IDs, insufficient for a three-way source-held-out split; each recording is isolated but generation-source independence is not established.",
            "IndicAudio groups by voice persona (29 groups); all clips from the same ElevenLabs/Fish Audio persona stay in the same split.",
        ],
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_external_manifest(input_path: Path, output_path: Path, *, seed: int = 1337) -> dict[str, Any]:
    """Normalize a user CSV: audio_path,label[,group_id,split] into our schema."""
    records: list[dict[str, Any]] = []
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"audio_path", "label"}.issubset(reader.fieldnames):
            raise ValueError("external manifest requires audio_path,label columns (bona-fide=0, spoof=1)")
        for index, row in enumerate(reader):
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = (input_path.parent / audio_path).resolve()
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"row {index + 2} has invalid label {label}")
            sample_id = row.get("sample_id") or f"external:{index:07d}"
            records.append({
                "sample_id": sample_id, "label": label, "split": row.get("split", ""),
                "group_id": row.get("group_id") or f"recording:{audio_path}", "audio_path": str(audio_path),
                "parquet_path": "", "row_group": -1, "row_in_group": -1, "audio_column": "", "source_dataset": "external",
            })
    known_splits = {"train", "validation", "calibration", "test"}
    if not all(record["split"] in known_splits for record in records):
        _assign_grouped_splits(records, (0.70, 0.10, 0.10, 0.10), seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    report = {
        "manifest": str(output_path.resolve()), "input_manifest": str(input_path.resolve()), "seed": seed,
        "total_records": len(records), "class_counts": {"bona_fide": sum(x["label"] == 0 for x in records), "spoof": sum(x["label"] == 1 for x in records)},
        "split_class_counts": {split: {"bona_fide": sum(x["split"] == split and x["label"] == 0 for x in records), "spoof": sum(x["split"] == split and x["label"] == 1 for x in records)} for split in known_splits},
        "grouping_note": "Provided group_id values were preserved. Without them, each file is its own group; supply speaker/source/session IDs for stronger leakage control.",
    }
    output_path.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the target-domain AASIST training manifest")
    parser.add_argument("--output", type=Path, default=DATASET_OUTPUT_DIR / "target_domain_manifest.csv")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--input-manifest", type=Path, help="Optional CSV: audio_path,label[,group_id,split]")
    args = parser.parse_args()
    report = build_external_manifest(args.input_manifest, args.output, seed=args.seed) if args.input_manifest else build_builtin_manifest(args.output, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
