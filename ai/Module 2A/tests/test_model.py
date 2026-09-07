from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aasist import AASISTDetector  # noqa: E402
from config import CHECKPOINT_PATH  # noqa: E402
from scoring import sha256_file  # noqa: E402


@unittest.skipUnless(CHECKPOINT_PATH.is_file(), "pretrained checkpoint is unavailable")
class TestPretrainedModel(unittest.TestCase):
    def test_cpu_load_predict_and_checkpoint_is_unchanged(self) -> None:
        before = sha256_file(CHECKPOINT_PATH)
        detector = AASISTDetector(device="cpu")
        result = detector.predict(np.zeros(16_000, dtype=np.float32))
        self.assertEqual(str(detector.device), "cpu")
        self.assertIsInstance(result["raw_score"], float)
        self.assertIsInstance(result["spoof_score"], float)
        self.assertIsNone(result["spoof_probability"])
        self.assertEqual(before, sha256_file(CHECKPOINT_PATH))
