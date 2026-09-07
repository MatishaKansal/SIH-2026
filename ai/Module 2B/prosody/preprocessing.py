import json
from pathlib import Path
import joblib, numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

class FeaturePreprocessor:
    def fit(self, X, feature_names):
        X = np.asarray(X, dtype=float); X[~np.isfinite(X)] = np.nan
        self.imputer = SimpleImputer(strategy="median").fit(X)
        imputed = self.imputer.transform(X)
        self.selector = VarianceThreshold(0.).fit(imputed)
        self.feature_names = [n for n, keep in zip(feature_names, self.selector.get_support()) if keep]
        return self
    def transform(self, X):
        X=np.asarray(X,dtype=float); X[~np.isfinite(X)] = np.nan
        return self.selector.transform(self.imputer.transform(X)).astype(np.float32)
    def save(self, model_dir):
        model_dir=Path(model_dir); model_dir.mkdir(parents=True,exist_ok=True)
        joblib.dump(self.imputer, model_dir/'feature_imputer.joblib'); joblib.dump(self.selector, model_dir/'feature_selector.joblib')
        (model_dir/'feature_names.json').write_text(json.dumps(self.feature_names, indent=2))
    @classmethod
    def load(cls, model_dir):
        obj=cls(); model_dir=Path(model_dir); obj.imputer=joblib.load(model_dir/'feature_imputer.joblib'); obj.selector=joblib.load(model_dir/'feature_selector.joblib'); obj.feature_names=json.loads((model_dir/'feature_names.json').read_text())
        if not hasattr(obj.imputer, '_fill_dtype') and hasattr(obj.imputer, '_fit_dtype'):
            obj.imputer._fill_dtype = obj.imputer._fit_dtype
        return obj
