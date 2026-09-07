"""Tests for preprocessing and VAD-window boundary behavior (no model download)."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad import SileroVAD, normalize_audio  # noqa: E402


class TestVADUtilities(unittest.TestCase):
    def test_normalize_array_downmixes_and_resamples(self) -> None:
        stereo = np.column_stack((np.ones(8_000), -np.ones(8_000))).astype(np.float32)
        output = normalize_audio(stereo, source_sample_rate=8_000)
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(output.ndim, 1)
        self.assertEqual(len(output), 16_000)
        self.assertTrue(np.allclose(output, 0.0))

    def test_chunker_uses_overlap_and_pads_final_window(self) -> None:
        detector = object.__new__(SileroVAD)
        detector.sample_rate = 16_000
        region = {"start": 2.0, "end": 3.25, "audio": np.ones(20_000, dtype=np.float32)}
        chunks = detector.chunk_regions([region], window_size_seconds=1.0, hop_size_seconds=0.5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual((chunks[0]["start"], chunks[0]["end"]), (2.0, 3.0))
        self.assertEqual((chunks[1]["start"], chunks[1]["end"]), (2.5, 3.25))
        self.assertTrue(chunks[1]["padded"])
        self.assertEqual(len(chunks[1]["audio"]), 16_000)


if __name__ == "__main__":
    unittest.main()
