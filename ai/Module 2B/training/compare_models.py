"""Train three identical-split GBDT candidates and persist the validation winner."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prosody.preprocessing import FeaturePreprocessor
from prosody.feature_extractor import ProsodyFeatureExtractor
from training.model_utils import scores,metrics,latency_ms
from training import train_lightgbm,train_xgboost,train_catboost

def main(feature_dir='output/features', output_dir='output'):
    root=Path(__file__).resolve().parents[1]; feature_dir=root/feature_dir; output=root/output_dir; model_dir=output/'models'; metric_dir=output/'metrics'; plot_dir=output/'plots'; model_dir.mkdir(parents=True,exist_ok=True); metric_dir.mkdir(parents=True,exist_ok=True); plot_dir.mkdir(parents=True,exist_ok=True)
    X,y=np.load(feature_dir/'train_features.npy'),np.load(feature_dir/'train_labels.npy'); Xv,yv=np.load(feature_dir/'val_features.npy'),np.load(feature_dir/'val_labels.npy')
    prep=FeaturePreprocessor().fit(X,ProsodyFeatureExtractor().feature_names); X,Xv=prep.transform(X),prep.transform(Xv); prep.save(model_dir)
    # Correlation analysis and feature statistics (spec §5)
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(12,10)); im=ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1); ax.set_xticks(range(len(prep.feature_names))); ax.set_yticks(range(len(prep.feature_names))); ax.set_xticklabels(prep.feature_names, rotation=90, fontsize=5); ax.set_yticklabels(prep.feature_names, fontsize=5); fig.colorbar(im); fig.tight_layout(); fig.savefig(plot_dir/'correlation_matrix.png', dpi=150); plt.close(fig)
    feat_stats = {name: {'mean': float(np.mean(X[:,i])), 'std': float(np.std(X[:,i])), 'min': float(np.min(X[:,i])), 'max': float(np.max(X[:,i]))} for i, name in enumerate(prep.feature_names)}
    (metric_dir/'feature_statistics.json').write_text(json.dumps(feat_stats, indent=2))
    contenders={"LightGBM":train_lightgbm.train,"XGBoost":train_xgboost.train,"CatBoost":train_catboost.train}; results={}; trained={}
    for name,fn in contenders.items():
        try:
            m=fn(X,y,Xv,yv); record=metrics(yv,scores(m,Xv)); record['latency_ms']=latency_ms(m,Xv); results[name]=record; trained[name]=m
        except ImportError as exc: results[name]={"error":str(exc)}
    valid=[n for n in trained];
    if not valid: raise RuntimeError('Install at least one GBDT dependency from requirements.txt')
    winner=sorted(valid,key=lambda n:(-results[n]['roc_auc'],results[n]['eer'],results[n]['latency_ms']))[0]
    joblib.dump(trained[winner],model_dir/'best_boosting_model.joblib'); (model_dir/'model_metadata.json').write_text(json.dumps({'model_family':winner},indent=2)); results['winner']=winner; (metric_dir/'model_comparison.json').write_text(json.dumps(results,indent=2)); return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--feature-dir',default='output/features');p.add_argument('--output-dir',default='output');args=p.parse_args();print(json.dumps(main(**vars(args)),indent=2))
