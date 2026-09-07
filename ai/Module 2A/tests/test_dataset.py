from pathlib import Path
import csv
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import AudioManifestDataset, read_manifest  # noqa: E402


class TestManifestDataset(unittest.TestCase):
    def test_external_wav_manifest_yields_aasist_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            sf.write(audio, np.zeros(8000), 8000)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "split", "group_id", "audio_path"])
                writer.writeheader()
                writer.writerow({"sample_id": "x", "label": 1, "split": "train", "group_id": "g", "audio_path": str(audio)})
            dataset = AudioManifestDataset(read_manifest(manifest))
            tensor, label, sample_id = dataset[0]
        self.assertEqual(tensor.shape[0], 64600)
        self.assertEqual(label, 1)
        self.assertEqual(sample_id, "x")
