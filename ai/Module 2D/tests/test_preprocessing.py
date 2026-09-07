from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xlsr import prepare_waveform  # noqa: E402


class TestPreprocessing(unittest.TestCase):
    def test_native_module1_contract_is_a_float32_noop(self) -> None:
        audio = np.linspace(-0.1, 0.1, 16_000, dtype=np.float32)
        prepared = prepare_waveform(audio, 16_000)
        self.assertEqual(prepared.dtype, np.float32)
        self.assertEqual(prepared.ndim, 1)
        np.testing.assert_array_equal(prepared, audio)

    def test_stereo_8khz_is_downmixed_and_resampled(self) -> None:
        stereo = np.column_stack((np.ones(8_000), np.zeros(8_000)))
        prepared = prepare_waveform(stereo, 8_000)
        self.assertEqual(prepared.dtype, np.float32)
        self.assertEqual(prepared.shape, (16_000,))

    def test_nan_and_empty_audio_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prepare_waveform(np.array([0.0, np.nan]), 16_000)
        with self.assertRaises(ValueError):
            prepare_waveform(np.array([], dtype=np.float32), 16_000)
