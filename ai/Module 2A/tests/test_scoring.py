from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoring import save_baseline_result, scores_from_logits  # noqa: E402


class TestScoring(unittest.TestCase):
    def test_spoof_margin_direction(self) -> None:
        result = scores_from_logits(np.array([3.0, 1.0]))
        self.assertEqual(result["raw_score"], 1.0)
        self.assertEqual(result["spoof_score"], 2.0)
        self.assertIsNone(result["spoof_probability"])

    def test_json_and_csv_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = {
                "input_file": "speech_000.wav", "duration_seconds": 1.0,
                "raw_score": 1.0, "spoof_score": 2.0, "spoof_probability": None,
                "label": None, "model": "AASIST", "stage": "baseline",
            }
            path = save_baseline_result(temp_dir, result)
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text())["spoof_score"], 2.0)
            with (Path(temp_dir) / "baseline_results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
