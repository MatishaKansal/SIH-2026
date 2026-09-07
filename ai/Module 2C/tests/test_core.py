from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor import SpectralExtractor
from input import normalize_audio, pad_or_trim
from model import SpectralCNN
from split_dataset import split_records
from dataset import AudioRecord, SpectralDataset, read_manifest
from inference import SpectralSpoofDetector
from split_dataset import (
    split_by_generator_holdout,
    split_integrity_report,
    validate_generator_disjoint,
)

def test_stereo_resamples_to_mono_16khz():
    audio=normalize_audio(np.ones((8000,2),dtype=np.float32),8000)
    assert audio.sample_rate==16000 and audio.waveform.ndim==1 and len(audio.waveform)==16000

def test_short_audio_padding_and_finite_features():
    padded, was_padded=pad_or_trim(np.ones(100,dtype=np.float32)); features=SpectralExtractor("logmel")(torch.from_numpy(padded))
    assert was_padded and features.shape[0]==64 and torch.isfinite(features).all()

def test_lfcc_uses_linear_filterbank_and_has_expected_shape():
    features=SpectralExtractor("lfcc")(torch.zeros(16000))
    assert features.shape[0]==20 and features.shape[1]==101 and torch.isfinite(features).all()

def test_model_uses_frequency_as_channels_and_embedding_dimension():
    features=SpectralExtractor("logmel")(torch.zeros(16000)).unsqueeze(0); model=SpectralCNN(64,32); embedding,logits=model(features)
    assert embedding.shape==(1,32) and logits.shape==(1,)

def test_detector_is_deterministic():
    torch.manual_seed(1); detector=SpectralSpoofDetector(); waveform=np.sin(np.linspace(0,100,16000)).astype(np.float32)
    first=detector.predict(waveform,16000); second=detector.predict(waveform,16000)
    assert first["embedding"]==second["embedding"] and len(first["embedding"])==64

def test_detector_rejects_whole_recording_longer_than_module1_window():
    detector=SpectralSpoofDetector()
    with __import__("pytest").raises(ValueError, match="Module 1 chunk"):
        detector.predict(np.zeros(16001,dtype=np.float32),16000)

def test_split_keeps_source_groups_together():
    records=[AudioRecord(f"{i}.wav",i%2,str(i//2)) for i in range(12)]; splits=split_records(records,seed=2)
    seen={record.source_id:split for split,items in splits.items() for record in items}
    assert len(seen)==6
    for split,items in splits.items(): assert all(seen[item.source_id]==split for item in items)

def test_normalization_rejects_nan():
    import pytest

    waveform = np.zeros(16000, dtype=np.float32)
    waveform[100] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_audio(waveform, 16000)


def test_normalization_rejects_inf():
    import pytest

    waveform = np.zeros(16000, dtype=np.float32)
    waveform[100] = np.inf

    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_audio(waveform, 16000)


def test_prepare_chunk_pads_short_chunk():
    from input import prepare_chunk

    waveform = np.ones(8000, dtype=np.float32)

    prepared, padded = prepare_chunk(waveform)

    assert prepared.shape == (16000,)
    assert padded is True
    assert np.allclose(prepared[:8000], 1.0)
    assert np.allclose(prepared[8000:], 0.0)


def test_prepare_chunk_rejects_oversized_chunk():
    import pytest
    from input import prepare_chunk

    waveform = np.zeros(16001, dtype=np.float32)

    with pytest.raises(ValueError, match="Module 1 chunk"):
        prepare_chunk(waveform)


def test_stft_has_expected_shape():
    extractor = SpectralExtractor("stft")
    features = extractor(torch.zeros(16000))

    assert features.shape == (257, 101)
    assert torch.isfinite(features).all()


def test_logmel_has_expected_shape():
    extractor = SpectralExtractor("logmel")
    features = extractor(torch.zeros(16000))

    assert features.shape == (64, 101)
    assert torch.isfinite(features).all()


def test_lfcc_has_expected_shape():
    extractor = SpectralExtractor("lfcc")
    features = extractor(torch.zeros(16000))

    assert features.shape == (20, 101)
    assert torch.isfinite(features).all()


def test_all_representations_are_finite():
    waveform = torch.randn(16000)

    for spectral_type in ["stft", "logmel", "lfcc"]:
        features = SpectralExtractor(spectral_type)(waveform)

        assert torch.isfinite(features).all()
        assert features.ndim == 2
        assert features.shape[1] == 101


def test_model_forward_is_finite():
    waveform = torch.randn(16000)

    for spectral_type in ["stft", "logmel", "lfcc"]:
        features = SpectralExtractor(spectral_type)(waveform).unsqueeze(0)

        model = SpectralCNN(
            in_channels=features.shape[1],
            embedding_dim=64,
        )

        embedding, logits = model(features)

        assert torch.isfinite(embedding).all()
        assert torch.isfinite(logits).all()
        assert embedding.shape == (1, 64)
        assert logits.shape == (1,)


def test_unavailable_speaker_id_cannot_be_used_for_grouping():
    import pytest

    records = [
        AudioRecord(f"{index}.wav", index % 2, f"source-{index}")
        for index in range(3)
    ]

    with pytest.raises(ValueError, match="speaker-disjoint"):
        split_records(records, group_by="speaker_id")


def test_speaker_grouping_keeps_each_speaker_together():
    records = [
        AudioRecord(f"{index}.wav", index % 2, f"source-{index}", f"speaker-{index // 2}")
        for index in range(12)
    ]
    splits = split_records(records, seed=2, group_by="speaker_id")
    assignments = {
        record.speaker_id: split
        for split, items in splits.items()
        for record in items
    }
    assert all(
        assignments[record.speaker_id] == split
        for split, items in splits.items()
        for record in items
    )


def test_explicit_split_preservation_and_validation():
    records = [
        AudioRecord(f"{split}.wav", index % 2, f"source-{split}", f"speaker-{split}", split=split)
        for index, split in enumerate(("train", "validation", "test"))
    ]
    preserved = split_records(records, preserve_existing_splits=True)
    assert [record.split for record in preserved["train"]] == ["train"]
    assert [record.split for record in preserved["validation"]] == ["validation"]
    assert [record.split for record in preserved["test"]] == ["test"]

    import pytest
    invalid = [records[0].__class__(**{**records[0].__dict__, "split": "dev"})] + records[1:]
    with pytest.raises(ValueError, match="valid split"):
        split_records(invalid, preserve_existing_splits=True)


def test_generator_holdout_is_disjoint_and_overlap_is_detected():
    records = [
        AudioRecord(
            f"{generator}-{index}.wav",
            index % 2,
            f"source-{generator}-{index}",
            f"speaker-{index}",
            generator=generator,
        )
        for generator in ("g1", "g2", "g3")
        for index in range(2)
    ]
    splits = split_by_generator_holdout(records, ["g3"], validation_fraction=0.5, seed=1)
    assert {record.generator for record in splits["test"]} == {"g3"}
    validate_generator_disjoint(splits["train"], splits["test"])
    import pytest
    with pytest.raises(ValueError, match="generator overlap"):
        validate_generator_disjoint(splits["train"], [splits["train"][0]])


def test_split_integrity_report_counts_known_metadata():
    splits = {
        "train": [
            AudioRecord("a.wav", 0, "source-a", "speaker-a", "g1", duration_seconds=1.5),
            AudioRecord("b.wav", 1, "source-b", "unavailable", "g2", duration_seconds=2.0),
        ],
        "test": [
            AudioRecord("c.wav", 1, "source-a", "speaker-c", "g3", duration_seconds=1.0),
        ],
    }
    report = split_integrity_report(splits)
    assert report["splits"]["train"]["record_count"] == 2
    assert report["splits"]["train"]["bona_fide_count"] == 1
    assert report["splits"]["train"]["spoof_count"] == 1
    assert report["splits"]["train"]["total_duration_seconds"] == 3.5
    assert report["splits"]["train"]["unique_speaker_count"] == 1
    assert report["overlap"]["test:train"]["source_id"] == ["source-a"]


def test_relative_manifest_path_resolves_from_manifest_directory(tmp_path):
    import csv
    import soundfile as sf

    audio_path = tmp_path / "audio" / "foo.wav"
    audio_path.parent.mkdir()
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "label", "source_id"])
        writer.writeheader()
        writer.writerow({"audio_path": "audio/foo.wav", "label": "0", "source_id": "source-1"})
    record = read_manifest(manifest)[0]
    assert record.audio_path == str(audio_path.resolve())


def test_full_recordings_are_rejected_by_spectral_dataset(tmp_path):
    import soundfile as sf
    import pytest

    audio_path = tmp_path / "long.wav"
    sf.write(audio_path, np.zeros(16001, dtype=np.float32), 16000)
    record = AudioRecord(str(audio_path), 0, "source-1")
    with pytest.raises(ValueError, match="explicit preparation/chunking"):
        SpectralDataset([record])[0]