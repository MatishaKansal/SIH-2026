import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prosody.calibration import PlattCalibrator
class TestCalibration(unittest.TestCase):
 def test_probability_and_threshold(self):
  y=np.array([0]*20+[1]*20);s=np.r_[np.linspace(0,.4,20),np.linspace(.6,1,20)];c=PlattCalibrator().fit(s,y);p=c.predict_proba(s);self.assertTrue(((p>=0)&(p<=1)).all());self.assertTrue(0<=c.threshold_metrics(s,y)['threshold']<=1)
