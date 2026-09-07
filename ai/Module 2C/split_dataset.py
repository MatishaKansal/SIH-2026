"""Leakage-safe dataset splitting for Module 2B."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Iterable

import numpy as np

from dataset import (
    AudioRecord,
    write_manifest,
)

VALID_SPLITS = {"train", "validation", "test"}
MISSING_METADATA = {"", "unavailable", "unknown", "not_available", "na"}


def _metadata_value(value: object) -> str | None:
    normalized = "" if value is None else str(value).strip().lower()
    return None if normalized in MISSING_METADATA else str(value).strip()


def split_records(
    records: list[AudioRecord],
    seed: int = 0,
    fractions: tuple[float, float, float] = (
        0.8,
        0.1,
        0.1,
    ),
    group_by: str = "source_id",
    preserve_existing_splits: bool = False,
    required_splits: Iterable[str] = VALID_SPLITS,
) -> dict[str, list[AudioRecord]]:
    """
    Split records without allowing one group to cross splits.

    Recommended:
        group_by="source_id"

    If speaker metadata is reliable:
        group_by="speaker_id"
    """

    if not records:
        raise ValueError(
            "records are empty"
        )

    required = set(required_splits)
    if not required.issubset(VALID_SPLITS):
        raise ValueError(
            f"required_splits must be drawn from {sorted(VALID_SPLITS)}"
        )

    if preserve_existing_splits:
        invalid = sorted(
            {
                str(record.split).strip().lower()
                for record in records
                if str(record.split).strip().lower() not in VALID_SPLITS
            }
        )
        if invalid:
            raise ValueError(
                "preserve_existing_splits=True requires every record to "
                f"have a valid split in {sorted(VALID_SPLITS)}; invalid or "
                f"missing values: {invalid}"
            )

        preserved = {
            split: [
                AudioRecord(
                    **{
                        **record.__dict__,
                        "split": split,
                    }
                )
                for record in records
                if record.split.strip().lower() == split
            ]
            for split in VALID_SPLITS
        }
        missing = sorted(
            split
            for split in required
            if not preserved[split]
        )
        if missing:
            raise ValueError(
                "preserved split assignments are missing required "
                f"splits: {missing}"
            )
        return preserved

    if len(fractions) != 3:
        raise ValueError(
            "fractions must contain train, "
            "validation and test fractions"
        )

    if any(
        fraction <= 0
        for fraction in fractions
    ):
        raise ValueError(
            "all split fractions must be positive"
        )

    total = sum(fractions)

    if not np.isclose(total, 1.0):
        raise ValueError(
            "split fractions must sum to 1.0"
        )

    if not hasattr(
        records[0],
        group_by,
    ):
        raise ValueError(
            f"unknown grouping field: {group_by}"
        )

    if group_by == "speaker_id":
        missing_speakers = sum(
            _metadata_value(record.speaker_id) is None
            for record in records
        )
        if missing_speakers:
            raise ValueError(
                "speaker-disjoint splitting cannot be performed safely: "
                f"{missing_speakers} record(s) have missing speaker_id "
                "metadata. Provide verified speaker IDs; they will not be "
                "inferred from filenames or demographics."
            )

    groups: dict[
        str,
        list[AudioRecord],
    ] = {}

    for record in records:
        group_value = getattr(
            record,
            group_by,
            None,
        )

        if not group_value:
            group_value = record.source_id

        groups.setdefault(
            str(group_value),
            [],
        ).append(record)

    group_keys = sorted(groups)

    if len(group_keys) < 3:
        raise ValueError(
            "at least three independent groups "
            "are required for train/validation/test"
        )

    rng = np.random.default_rng(seed)

    rng.shuffle(group_keys)

    n_groups = len(group_keys)

    train_count = max(
        1,
        int(round(
            n_groups * fractions[0]
        )),
    )

    validation_count = max(
        1,
        int(round(
            n_groups * fractions[1]
        )),
    )

    # Leave at least one group for test.
    if (
        train_count
        + validation_count
        >= n_groups
    ):
        validation_count = max(
            1,
            n_groups
            - train_count
            - 1,
        )

    test_count = (
        n_groups
        - train_count
        - validation_count
    )

    if test_count < 1:
        train_count -= 1
        test_count = 1

    train_groups = group_keys[
        :train_count
    ]

    validation_groups = group_keys[
        train_count:
        train_count + validation_count
    ]

    test_groups = group_keys[
        train_count + validation_count:
    ]

    result: dict[
        str,
        list[AudioRecord],
    ] = {}

    for split_name, selected_groups in (
        ("train", train_groups),
        ("validation", validation_groups),
        ("test", test_groups),
    ):
        split_records_: list[
            AudioRecord
        ] = []

        for group in selected_groups:
            for record in groups[group]:
                split_records_.append(
                    AudioRecord(
                        **{
                            **record.__dict__,
                            "split": split_name,
                        }
                    )
                )

        result[split_name] = split_records_

    return result


def create_split_manifests(
    records: list[AudioRecord],
    output_dir: str | Path,
    seed: int = 0,
    group_by: str = "source_id",
    preserve_existing_splits: bool = False,
    required_splits: Iterable[str] = VALID_SPLITS,
) -> dict[str, Path]:
    """Create train/validation/test CSV manifests."""

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = split_records(
        records,
        seed=seed,
        group_by=group_by,
        preserve_existing_splits=preserve_existing_splits,
        required_splits=required_splits,
    )

    paths: dict[str, Path] = {}

    for split_name, records_ in splits.items():
        path = (
            output_dir
            / f"{split_name}.csv"
        )

        write_manifest(
            records_,
            path,
        )

        paths[split_name] = path

    return paths


def validate_generator_disjoint(
    train_records: list[AudioRecord],
    test_records: list[AudioRecord],
) -> None:
    """Raise if known generator IDs overlap between train and test."""
    train_generators = {
        value
        for record in train_records
        if (value := _metadata_value(record.generator)) is not None
    }
    test_generators = {
        value
        for record in test_records
        if (value := _metadata_value(record.generator)) is not None
    }
    overlap = train_generators & test_generators
    if overlap:
        raise ValueError(
            f"generator overlap between train and test: {sorted(overlap)}"
        )


def split_by_generator_holdout(
    records: list[AudioRecord],
    held_out_generators: Iterable[str],
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
    group_by: str = "source_id",
) -> dict[str, list[AudioRecord]]:
    """Create train/validation splits plus a generator-disjoint test split."""
    held_out = {str(value).strip() for value in held_out_generators}
    if not held_out:
        raise ValueError("held_out_generators must not be empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    missing = [
        record.sample_id or record.audio_path
        for record in records
        if _metadata_value(record.generator) is None
    ]
    if missing:
        raise ValueError(
            "generator holdout requires a verified generator for every "
            f"record; missing metadata in {len(missing)} record(s)"
        )

    test_records = [
        record
        for record in records
        if str(record.generator).strip() in held_out
    ]
    development_records = [
        record
        for record in records
        if str(record.generator).strip() not in held_out
    ]
    if not test_records:
        raise ValueError("held_out_generators matched no records")
    if not development_records:
        raise ValueError("generator holdout leaves no training records")

    groups: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in development_records:
        value = getattr(record, group_by, None)
        if not value:
            raise ValueError(
                f"generator holdout requires non-missing {group_by} values"
            )
        groups[str(value)].append(record)

    group_keys = sorted(groups)
    if len(group_keys) < 2:
        raise ValueError(
            "at least two development groups are required for train/validation"
        )
    np.random.default_rng(seed).shuffle(group_keys)
    validation_count = max(1, round(len(group_keys) * validation_fraction))
    validation_keys = set(group_keys[:validation_count])

    result = {
        "train": [record for key in group_keys if key not in validation_keys for record in groups[key]],
        "validation": [record for key in group_keys if key in validation_keys for record in groups[key]],
        "test": [AudioRecord(**{**record.__dict__, "split": "test"}) for record in test_records],
    }
    validate_generator_disjoint(result["train"], result["test"])
    validate_generator_disjoint(result["validation"], result["test"])
    return {
        split: [
            AudioRecord(**{**record.__dict__, "split": split})
            for record in split_records_
        ]
        for split, split_records_ in result.items()
    }


def split_integrity_report(
    splits: dict[str, list[AudioRecord]],
) -> dict[str, object]:
    """Summarize counts and known metadata overlap for each split."""
    report: dict[str, object] = {"splits": {}, "overlap": {}}
    metadata_fields = ("source_id", "speaker_id", "generator")
    known_sets: dict[str, dict[str, set[str]]] = {}

    for split, records in splits.items():
        classes = {0: 0, 1: 0}
        for record in records:
            classes[record.label] = classes.get(record.label, 0) + 1
        split_report = {
            "record_count": len(records),
            "bona_fide_count": classes.get(0, 0),
            "spoof_count": classes.get(1, 0),
            "total_duration_seconds": sum(
                record.duration_seconds or 0.0
                for record in records
            ),
        }
        known_sets[split] = {
            field: {
                value
                for record in records
                if (value := _metadata_value(getattr(record, field))) is not None
            }
            for field in metadata_fields
        }
        for field in metadata_fields:
            if field == "speaker_id":
                split_report["unique_speaker_count"] = len(known_sets[split][field])
            elif field == "source_id":
                split_report["unique_source_count"] = len(known_sets[split][field])
            else:
                split_report["unique_generator_count"] = len(known_sets[split][field])
        report["splits"][split] = split_report

    for left_name, left_sets in known_sets.items():
        for right_name, right_sets in known_sets.items():
            if left_name >= right_name:
                continue
            report["overlap"][f"{left_name}:{right_name}"] = {
                field: sorted(left_sets[field] & right_sets[field])
                for field in metadata_fields
            }
    return report