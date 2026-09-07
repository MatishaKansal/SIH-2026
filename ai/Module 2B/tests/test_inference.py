import unittest
from pathlib import Path
import numpy as np

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[1]
        model_file = base / 'output' / 'models' / 'best_boosting_model.joblib'
        if not model_file.exists():
            raise unittest.SkipTest('Module 2B model artifacts not found. Run training first.')
        import sys
        sys.path.insert(0, str(base))
        from prosody.detector import ProsodySpoofDetector
        cls.detector = ProsodySpoofDetector(base / 'output')

    def test_schema_and_types(self):
        y = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000).astype(np.float32)
        r = self.detector.predict(y)
        expected_keys = {'model_family', 'raw_score', 'calibrated_spoof_prob', 'decision', 'feature_vector', 'top_forensic_cues', 'latency_ms'}
        self.assertTrue(expected_keys.issubset(r.keys()))
        self.assertIn(r['decision'], ('spoof', 'bonafide'))
        self.assertTrue(0.0 <= r['calibrated_spoof_prob'] <= 1.0)
        self.assertEqual(len(r['feature_vector']), 56)
        self.assertIsInstance(r['top_forensic_cues'], list)
        self.assertGreater(len(r['top_forensic_cues']), 0)
        self.assertIsInstance(r['top_forensic_cues'][0], str)

    def test_silence_input(self):
        r = self.detector.predict(np.zeros(16000, dtype=np.float32))
        self.assertIn(r['decision'], ('spoof', 'bonafide'))
        self.assertTrue(0.0 <= r['calibrated_spoof_prob'] <= 1.0)

    def test_latency_constraint(self):
        y = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000).astype(np.float32)
        # Warmup
        self.detector.predict(y)
        latencies = [self.detector.predict(y)['latency_ms'] for _ in range(10)]
        mean_latency = float(np.mean(latencies))
        self.assertLess(mean_latency, 50.0, f"Mean latency {mean_latency:.2f}ms exceeds 50ms budget")

