import argparse,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prosody.preprocessing import FeaturePreprocessor
def main(feature_dir='output/features',output_dir='output',limit=200):
 import shap
 root=Path(__file__).resolve().parents[1];f=root/feature_dir;o=root/output_dir;(o/'plots').mkdir(parents=True,exist_ok=True); X=np.load(f/'test_features.npy')[:limit]; prep=FeaturePreprocessor.load(o/'models');X=prep.transform(X);m=joblib.load(o/'models'/'best_boosting_model.joblib'); explainer=shap.TreeExplainer(m); values=explainer.shap_values(X); values=values[1] if isinstance(values,list) else values; imp=np.abs(values).mean(axis=0);df=pd.DataFrame({'feature':prep.feature_names,'mean_abs_shap':imp}).sort_values('mean_abs_shap',ascending=False);df.head(20).to_csv(o/'plots'/'feature_importance.csv',index=False);plt.figure(figsize=(8,6));plt.barh(df.head(20).feature[::-1],df.head(20).mean_abs_shap[::-1]);plt.tight_layout();plt.savefig(o/'plots'/'feature_importance.png');plt.close();shap.summary_plot(values,X,feature_names=prep.feature_names,show=False);plt.tight_layout();plt.savefig(o/'plots'/'shap_summary.png');plt.close();return df
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--feature-dir',default='output/features');p.add_argument('--output-dir',default='output');p.add_argument('--limit',type=int,default=200);a=p.parse_args();main(**vars(a))
