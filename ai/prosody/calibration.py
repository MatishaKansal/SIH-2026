import json
from pathlib import Path
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve

class PlattCalibrator:
    def fit(self, scores, labels):
        self.model = LogisticRegression(random_state=1337).fit(np.asarray(scores).reshape(-1,1), labels)
        return self
    def predict_proba(self, scores): return self.model.predict_proba(np.asarray(scores).reshape(-1,1))[:,1]
    def threshold_metrics(self, scores, labels):
        probs=self.predict_proba(scores); fpr,tpr,thr=roc_curve(labels,probs); idx=np.argmax(tpr-fpr)
        return {"threshold":float(thr[idx]),"brier_score":float(brier_score_loss(labels,probs)),"ece":float(expected_calibration_error(labels,probs))}
    def save(self, path): joblib.dump(self.model,path)
    @classmethod
    def load(cls,path): obj=cls(); obj.model=joblib.load(path); return obj

def expected_calibration_error(y, p, bins=10):
    y,p=np.asarray(y),np.asarray(p); result=0.
    for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(1/bins,1,bins)):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if mask.any(): result += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return result
