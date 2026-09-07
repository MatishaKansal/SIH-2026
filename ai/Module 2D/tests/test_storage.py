from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_embeddings  # noqa: E402


class _FakeExtractor:
    device = "cpu"
    processor_name = "FakeProcessor"

    def __init__(self, *args, **kwargs):
        pass

    def extract_batch(self, waveforms, sample_rate, *, valid_lengths=None):
        assert sample_rate == 16_000
        assert valid_lengths == [12_000]
        return np.ones((len(waveforms), 1_024), dtype=np.float32)


class TestStorage(unittest.TestCase):
    def test_module1_manifest_metadata_and_embedding_index_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sf.write(root / "speech_000.wav", np.zeros(16_000, dtype=np.float32), 16_000)
            (root / "manifest.json").write_text(json.dumps({
                "source": "input_audio/source.wav", "sample_rate": 16_000, "channels": 1,
                "segments": [{"id": 7, "start": 2.0, "end": 2.75, "duration": 0.75, "file": "speech_000.wav", "padded": True, "region_index": 3}],
            }))
            with patch.object(extract_embeddings, "XLSREmbeddingExtractor", _FakeExtractor):
                summary = extract_embeddings.extract_module1_embeddings(root, root / "out", dataset="demo", label=1)
            self.assertEqual(summary["count"], 1)
            matrix = np.load(root / "out" / "embeddings.npy")
            self.assertEqual(matrix.shape, (1, 1_024))
            with (root / "out" / "metadata.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["chunk_id"], "7")
            self.assertEqual(row["source_id"], "source")
            self.assertEqual(row["padded"], "True")
            self.assertEqual(row["embedding_index"], "0")
