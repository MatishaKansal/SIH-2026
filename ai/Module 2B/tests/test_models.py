import unittest
class TestModels(unittest.TestCase):
 def test_dependencies_or_trainers(self):
  try: import lightgbm,xgboost,catboost
  except ImportError: self.skipTest('optional GBDT dependencies not installed')
  import numpy as np,sys
  from pathlib import Path
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from training import train_lightgbm,train_xgboost,train_catboost
  X=np.random.default_rng(1337).normal(size=(100,6));y=(X[:,0]>0).astype(int)
  for fn in (train_lightgbm.train,train_xgboost.train,train_catboost.train): self.assertEqual(len(fn(X[:80],y[:80],X[80:],y[80:]).predict(X[80:])),20)
