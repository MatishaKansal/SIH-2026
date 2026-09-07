import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prosody.feature_extractor import ProsodyFeatureExtractor
class TestExtractor(unittest.TestCase):
 def setUp(self): self.x=ProsodyFeatureExtractor();self.y=np.sin(2*np.pi*180*np.arange(16000)/16000).astype('float32')
 def test_shape_determinism(self): self.assertEqual(self.x.extract(self.y).shape,(56,));np.testing.assert_allclose(self.x.extract(self.y),self.x.extract(self.y),equal_nan=True)
 def test_silence(self): self.assertEqual(self.x.extract(np.zeros(16000)).shape,(56,))
 def test_short(self): self.assertEqual(self.x.extract(np.random.randn(100).astype('float32')).shape,(56,))
