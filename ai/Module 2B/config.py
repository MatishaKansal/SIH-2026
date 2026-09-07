from pathlib import Path

SEED = 1337
SAMPLE_RATE = 16000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_ROOT / "output"
FEATURE_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODEL_PARAMS = {
    "LightGBM": dict(n_estimators=400, learning_rate=.03, num_leaves=31, subsample=.8, colsample_bytree=.8, random_state=SEED),
    "XGBoost": dict(n_estimators=400, learning_rate=.03, max_depth=6, subsample=.8, colsample_bytree=.8, random_state=SEED),
    "CatBoost": dict(iterations=400, learning_rate=.03, depth=6, random_seed=SEED, verbose=False),
}
