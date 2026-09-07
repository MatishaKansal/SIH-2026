import argparse,json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prosody.preprocessing import FeaturePreprocessor
from prosody.calibration import PlattCalibrator
def main(feature_dir='output/features',output_dir='output'):
 root=Path(__file__).resolve().parent; f=root/feature_dir;o=root/output_dir;X=np.load(f/'cal_features.npy');y=np.load(f/'cal_labels.npy');prep=FeaturePreprocessor.load(o/'models');m=joblib.load(o/'models'/'best_boosting_model.joblib');s=m.predict_proba(prep.transform(X))[:,1];c=PlattCalibrator().fit(s,y);d=o/'calibration';d.mkdir(parents=True,exist_ok=True);(o/'plots').mkdir(parents=True,exist_ok=True);c.save(d/'calibration_model.joblib');stats=c.threshold_metrics(s,y);(d/'thresholds.json').write_text(json.dumps(stats,indent=2));p=c.predict_proba(s);a,b=calibration_curve(y,p,n_bins=10);plt.figure();plt.plot(b,a,'o-');plt.plot([0,1],[0,1],'--');plt.xlabel('Mean predicted probability');plt.ylabel('Fraction spoof');plt.savefig(o/'plots'/'calibration_curve.png',bbox_inches='tight');plt.close();return stats
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--feature-dir',default='output/features');p.add_argument('--output-dir',default='output');a=p.parse_args();print(main(**vars(a)))
