import csv
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_datasets import (
    SourceExample,
    load_hf_stream,
    prepare_examples,
    select_hf_split,
    validate_local_dataset_paths,
    validate_manifests,
)


def _write_fixture(path: Path, samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(samples, dtype=np.float32), 16000)


def test_prepare_datasets_writes_schema_labels_chunks_and_report(tmp_path):
    examples = []
    for dataset, label, speaker in (
        ("svarah", 0, "unavailable"),
        ("indic_audio", 1, "persona_a"),
        ("orpheus", 1, "unavailable"),
    ):
        for index in range(3):
            source = tmp_path / "inputs" / dataset / f"clip_{index}.wav"
            _write_fixture(source, 24000 if index == 0 else 8000)
            examples.append(
                SourceExample(
                    dataset=dataset,
                    audio_path=str(source),
                    label=label,
                    source_id=f"{dataset}:source:{index}",
                    speaker_id=speaker if dataset != "indic_audio" else f"persona_{index}",
                    generator="Fish Audio S2 Pro" if dataset == "indic_audio" else "unavailable",
                    sample_id=f"{dataset}:clip:{index}",
                    duration_seconds=(24000 if index == 0 else 8000) / 16000,
                    sample_rate=16000,
                    channels=1,
                )
            )

    output_dir = tmp_path / "prepared"
    report = prepare_examples(examples, output_dir, seed=7)

    required = {
        "dataset", "audio_path", "label", "source_id", "speaker_id",
        "generator", "split", "sample_id", "duration_seconds", "chunk_id",
        "start", "end", "padded", "region_index",
    }
    manifest_rows = list(csv.DictReader((output_dir / "module2b_manifest.csv").open()))
    assert required.issubset(manifest_rows[0])
    assert {int(row["label"]) for row in manifest_rows} == {0, 1}
    assert len({row["sample_id"] for row in manifest_rows}) == len(manifest_rows)
    assert any(row["padded"] == "True" for row in manifest_rows)
    assert report["seed"] == 7
    assert report["chunk_count"]["train"] > 0
    assert (output_dir / "module2b_dataset_report.json").exists()
    assert (output_dir / "module2b_train.csv").exists()
    assert (output_dir / "module2b_val.csv").exists()
    assert (output_dir / "module2b_test.csv").exists()


def test_prepared_manifests_validate_without_network(tmp_path):
    examples = []
    for dataset, label in (("svarah", 0), ("indic_audio", 1), ("orpheus", 1)):
        for index in range(3):
            source = tmp_path / "inputs" / dataset / f"clip_{index}.wav"
            _write_fixture(source, 8000)
            examples.append(
                SourceExample(
                    dataset=dataset,
                    audio_path=str(source),
                    label=label,
                    source_id=f"{dataset}:source:{index}",
                    speaker_id=f"persona_{index}" if dataset == "indic_audio" else "unavailable",
                    sample_id=f"{dataset}:clip:{index}",
                    duration_seconds=0.5,
                    sample_rate=16000,
                    channels=1,
                )
            )
    output_dir = tmp_path / "prepared"
    prepare_examples(examples, output_dir, seed=3)
    result = validate_manifests(output_dir)
    assert result["splits"]["train"]["record_count"] > 0


def test_hf_validation_split_is_selected_when_train_is_unavailable():
    selected = select_hf_split("indic_audio", ["validation"], "train")
    assert selected == "validation"

    calls = []

    def fake_split_names(repo):
        return ["validation"]

    def fake_load_dataset(repo, *, split, streaming, **kwargs):
        calls.append((repo, split, streaming, kwargs))
        return []

    result = load_hf_stream(
        "indic_audio",
        limit=100,
        output_dir=Path("/tmp/module2b-test-hf"),
        load_dataset_fn=fake_load_dataset,
        get_dataset_split_names_fn=fake_split_names,
    )
    assert result == []
    assert calls[0][1:3] == ("validation", True)
    assert "download_config" not in calls[0][3]


def test_hf_stream_fails_cleanly_when_orpheus_shard_errors():
    def fake_split_names(repo):
        return ["train"]

    def failing_stream():
        raise OSError(9, "Bad file descriptor")
        yield  # pragma: no cover

    def fake_load_dataset(repo, *, split, streaming, **kwargs):
        assert split == "train"
        assert streaming is True
        return failing_stream()

    import pytest

    with pytest.raises(RuntimeError, match="Unable to stream orpheus"):
        load_hf_stream(
            "orpheus",
            limit=100,
            output_dir=Path("/tmp/module2b-test-orpheus"),
            load_dataset_fn=fake_load_dataset,
            get_dataset_split_names_fn=fake_split_names,
        )


def test_hf_stream_casts_audio_to_raw_references_without_decoding(tmp_path):
    calls = []

    class FakeStream:
        def cast_column(self, column, feature):
            calls.append((column, feature.decode))
            return self

        def __iter__(self):
            yield {
                "audio": {
                    "bytes": b"not-a-real-audio-file",
                    "path": "clip.wav",
                },
                "text": "sample",
                "source": 1,
            }

    def fake_split_names(repo):
        return ["train"]

    def fake_load_dataset(repo, *, split, streaming, **kwargs):
        assert streaming is True
        return FakeStream()

    import pytest

    def cast_audio_column(stream):
        calls.append(("audio", False))
        return stream

    with pytest.raises(RuntimeError, match="could not be decoded"):
        load_hf_stream(
            "orpheus",
            limit=1,
            output_dir=tmp_path,
            load_dataset_fn=fake_load_dataset,
            get_dataset_split_names_fn=fake_split_names,
            cast_audio_column_fn=cast_audio_column,
        )
    assert calls == [("audio", False)]


def test_bounded_hf_stream_closes_iterator_after_limit(tmp_path):
    class ClosingIterator:
        def __init__(self):
            self.index = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= 3:
                raise StopIteration
            self.index += 1
            return {
                "audio": {
                    "bytes": b"invalid",
                    "path": f"clip_{self.index}.wav",
                },
                "text": "sample",
                "source": 1,
            }

        def close(self):
            self.closed = True

    iterator = ClosingIterator()

    def fake_split_names(repo):
        return ["train"]

    def fake_load_dataset(repo, *, split, streaming, **kwargs):
        return iterator

    def cast_audio_column(stream):
        return stream

    import pytest

    with pytest.raises(RuntimeError, match="could not be decoded"):
        load_hf_stream(
            "orpheus",
            limit=1,
            output_dir=tmp_path,
            load_dataset_fn=fake_load_dataset,
            get_dataset_split_names_fn=fake_split_names,
            cast_audio_column_fn=cast_audio_column,
        )
    assert iterator.closed is True


def test_local_dataset_paths_require_all_three_audio_inputs(tmp_path):
    svarah = tmp_path / "svarah"
    (svarah / "audio").mkdir(parents=True)
    (svarah / "metadata.csv").write_text("filename,duration\n", encoding="utf-8")
    indic = tmp_path / "indic"
    indic.mkdir()
    (indic / "metadata.jsonl").write_text("", encoding="utf-8")
    orpheus = tmp_path / "orpheus"
    orpheus.mkdir()
    import pytest

    with pytest.raises(FileNotFoundError, match="Orpheus"):
        validate_local_dataset_paths(svarah, indic, orpheus)

    _write_fixture(orpheus / "clip.wav", 16000)
    counts = validate_local_dataset_paths(svarah, indic, orpheus)
    assert counts["orpheus_audio_files"] == 1
