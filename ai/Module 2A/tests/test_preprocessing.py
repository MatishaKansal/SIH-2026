from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import preprocess_audio, prepare_for_aasist  # noqa: E402


class TestPreprocessing(unittest.TestCase):
    def test_stereo_8khz_file_is_mono_16khz_float32(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stereo.wav"
            sf.write(path, np.column_stack((np.ones(8000), np.zeros(8000))), 8000)
            audio = preprocess_audio(path)
        self.assertEqual(audio.sample_rate, 16000)
        self.assertEqual(audio.waveform.dtype, np.float32)
        self.assertEqual(audio.waveform.ndim, 1)
        self.assertEqual(len(audio.waveform), 16000)

    def test_prepare_repeats_short_input_to_official_length(self) -> None:
        prepared = prepare_for_aasist(np.array([1.0, 2.0], dtype=np.float32), 7)
        np.testing.assert_array_equal(prepared, [1, 2, 1, 2, 1, 2, 1])

    def test_nan_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_audio(np.array([0.0, np.nan]), 16000)
