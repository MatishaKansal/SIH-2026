from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import fit_platt_calibrator, load_calibrator, save_calibrator  # noqa: E402


class TestCalibration(unittest.TestCase):
    def test_platt_calibration_serializes_and_returns_spoof_probability(self) -> None:
        scores = np.r_[np.linspace(-4, -1, 25), np.linspace(1, 4, 25)]
        labels = np.r_[np.zeros(25, dtype=int), np.ones(25, dtype=int)]
        calibrator, report = fit_platt_calibrator(scores, labels, model_checkpoint_sha256="abc", model_identifier="test")
        self.assertGreater(calibrator.predict_proba([3.0])[0], calibrator.predict_proba([-3.0])[0])
        self.assertIn("roc_auc", report["fit_diagnostics"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_calibrator(calibrator, Path(temp_dir) / "calibration_model.joblib")
            loaded = load_calibrator(path, expected_checkpoint_sha256="abc")
            self.assertAlmostEqual(loaded.predict_proba([1.0])[0], calibrator.predict_proba([1.0])[0])

    def test_rejects_insufficient_class_support(self) -> None:
        with self.assertRaises(ValueError):
            fit_platt_calibrator([0.0, 1.0], [0, 1], model_checkpoint_sha256="a", model_identifier="x")
