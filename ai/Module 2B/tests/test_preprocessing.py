import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prosody.preprocessing import FeaturePreprocessor
class TestPreprocess(unittest.TestCase):
 def test_impute_and_drop_constant(self):
  x=np.array([[1,np.nan,4],[2,3,4],[np.inf,4,4]],float);p=FeaturePreprocessor().fit(x,['a','b','constant']);z=p.transform(x);self.assertEqual(z.shape,(3,2));self.assertTrue(np.isfinite(z).all())
