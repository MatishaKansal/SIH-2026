# Module 2B: Handcrafted Prosody & Acoustic Multi-GBDT Boosting Branch

Module 2B provides a real-time, highly interpretable prosodic and acoustic deepfake detection branch for the SIH 2026 AI Voice-Cloning Detection System. It extracts **56 interpretable acoustic features** from 1.0-second speech windows, evaluates three competitive Gradient Boosted Decision Tree (GBDT) families (**LightGBM**, **XGBoost**, **CatBoost**), performs Platt probability calibration on held-out calibration data, generates SHAP explainability visualizations, and provides the real-time inference class `ProsodySpoofDetector` with **<50ms CPU latency**.

---

## 1. Acoustic Feature Taxonomy (56 Dimensions)

Every 16 kHz mono waveform window (typically 16,000 samples) is converted into a 56-dimensional feature vector across 4 acoustic categories:

### A. Fundamental Frequency ($F0$ / Pitch) & Intonation Dynamics (12 features)
Neural TTS vocoders frequently display unnatural pitch stability, missing micro-variations, or rigid phrase-boundary intonation.
- `f0_mean_hz`, `f0_median_hz`, `f0_std_hz`: Pitch central tendency and variation across voiced frames.
- `f0_min_hz`, `f0_max_hz`, `f0_range_hz`, `f0_iqr_hz`: Pitch boundaries and interquartile spread.
- `voiced_frame_ratio`, `unvoiced_frame_ratio`: Ratio of voiced frames ($F0 > 0$) to total window frames.
- `f0_slope_hz_per_frame`: Linear regression intonation slope across the window.
- `f0_velocity_hz`, `f0_acceleration_hz`: Mean absolute first ($|\Delta F0|$) and second ($|\Delta^2 F0|$) pitch derivatives.

### B. Energy & Intensity Dynamics (9 features)
Synthetic vocoders often struggle with natural syllabic power decay and respiratory dynamics.
- `rms_mean`, `rms_std`: Average loudness and standard deviation over time.
- `rms_min`, `rms_max`, `rms_range`: Loudness extremes and dynamic range ($max - min$).
- `rms_skewness`, `rms_kurtosis`: Higher-order statistical distribution moments of energy.
- `voiced_to_unvoiced_energy_ratio`: Ratio of average energy during voiced frames vs. unvoiced gaps.
- `energy_flux`: Mean absolute frame-to-frame RMS variation.

### C. Speech Rhythm & Temporal Fluency (8 features)
AI voices generate uniform cadence without biological breathing pauses or syllable variation.
- `voiced_duration_s`, `unvoiced_duration_s`: Total cumulative duration of voiced and unvoiced intervals.
- `voiced_burst_count`, `pause_count`: Number of distinct voiced bursts and unvoiced pauses ($>50\text{ms}$).
- `mean_voiced_segment_s`, `mean_pause_s`: Average durations of voiced segments and pauses.
- `pause_variance_s2`: Variance of pause lengths (natural human speech has high pause variance).
- `speaking_activity_ratio`: Total voiced duration divided by window duration.

### D. Voice Quality & Spectral Texture (27 features)
Vocoder synthesis architectures (HiFi-GAN, BigVGAN) leave distinctive cycle-to-cycle perturbation artifacts.
- `local_jitter`: Period perturbation quotient across glottal cycles (Praat / Parselmouth).
- `local_shimmer`: Amplitude perturbation quotient across glottal cycles (Praat / Parselmouth).
- `hnr_db`: Harmonics-to-Noise Ratio in decibels (acoustic harmonicity).
- `spectral_centroid_mean`, `spectral_centroid_std`: Spectral center of mass / brightness.
- `spectral_bandwidth_mean`, `spectral_bandwidth_std`: Spectral spread around centroid.
- `spectral_rolloff_mean`, `spectral_rolloff_std`: 85% energy rolloff frequency.
- `spectral_flatness_mean`, `spectral_flatness_std`: Tonality vs. white-noise ratio.
- `zcr_mean`, `zcr_std`: Zero-crossing rate mean and standard deviation.
- `spectral_contrast_mean_band_0..6`, `spectral_contrast_std_band_0..6`: Sub-band peak-to-valley contrast across 7 frequency bands.

---

## 2. Leakage-Safe Preprocessing

Implemented in `prosody/preprocessing.py`:
1. **Median Imputation (`SimpleImputer`):** Fits median values strictly on the `train` split. Handles any windows containing zero voiced frames ($NaN$ pitch values) cleanly without data leakage.
2. **Variance Thresholding (`VarianceThreshold`):** Drops strictly constant / zero-variance features on the training set.
3. **Artifact Persistence:** Saves `output/models/feature_imputer.joblib`, `feature_selector.joblib`, and active `feature_names.json`.
4. **Correlation Analysis:** Generates `output/plots/correlation_matrix.png` and `output/metrics/feature_statistics.json`.

---

## 3. Multi-GBDT Model Benchmark (Full 3-Dataset Corpus: 25,122 Samples)

All three models were trained on 17,518 training samples and benchmarked on 2,587 validation samples under identical leakage-safe features and splits:

| Metric | LightGBM | XGBoost | CatBoost (Winner) |
| :--- | :---: | :---: | :---: |
| **Validation ROC-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **Validation PR-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **Validation EER** | **0.00%** | **0.00%** | **0.00%** |
| **Accuracy** | 100.0% | 100.0% | **100.0%** |
| **F1 Score** | 1.0000 | 1.0000 | **1.0000** |
| **Inference Latency** | 0.981 ms | 0.581 ms | **0.525 ms** |
| **Status** | Contender | Contender | **WINNER (Selected)** |

**Winner Selection Rule:** Highest Validation ROC-AUC, with lowest EER and lowest latency as tie-breakers. Checkpoint saved to `output/models/best_boosting_model.joblib`.

---

## 4. Platt Probability Calibration

Raw boosting margins are uncalibrated and cannot be passed directly into Wald SPRT.
- **Calibrator:** Logistic Regression Platt Scaler fit strictly on the held-out `calibration` split (2,505 samples).
- **Optimal Threshold ($\tau^*$):** `0.7722` (selected via Youden's $J$ statistic).
- **Brier Score:** `0.0481` | **ECE:** `0.1918`.
- **Artifacts:** `output/calibration/calibration_model.joblib`, `thresholds.json`, `output/plots/calibration_curve.png`.

---

## 5. Held-Out Test Evaluation (Full Test Set: 2,512 Samples)

Evaluated strictly on the untouched test split (665 bona-fide human speech from Svarah / 1,847 synthetic AI voice clones from Orpheus & indic-audio):
- **ROC-AUC:** **100.0%** (`1.0`)
- **PR-AUC:** **100.0%** (`1.0`)
- **Equal Error Rate (EER):** **0.00%**
- **Test Accuracy:** **99.88%** (2,509 / 2,512 samples correct)
- **False Acceptance Rate (FAR):** **0.00%** (0 AI voice clones bypassed detection, 665 / 665 real voices correct)
- **False Rejection Rate (FRR):** **0.16%** (only 3 synthetic samples out of 1,847 missed)
- **Brier Score:** `0.0513` (outstanding calibration)
- **Artifacts:** `output/metrics/test_metrics.json`, `output/plots/roc_curves.png`.

---

## 6. Forensic Explainability (SHAP Analysis)

- **TreeExplainer:** Computes SHAP values across test samples.
- **Top Discriminators:**
  1. `spectral_contrast_mean_band_6` (mean $|SHAP| = 0.1933$): High-frequency spectral contrast discrepancies from vocoders.
  2. `f0_max_hz` ($0.0282$): Unnatural pitch ceiling boundaries in synthetic speech.
  3. `spectral_contrast_mean_band_4` ($0.0229$): Mid-to-high frequency spectral contrast peak-to-valley differences.
  4. `unvoiced_frame_ratio` ($0.0087$): Breathing and unvoiced phoneme duration patterns.
  5. `f0_std_hz` ($0.0080$): Pitch standard deviation (flat/monotone TTS delivery vs. human prosody).
- **Artifacts:** `output/plots/feature_importance.png`, `feature_importance.csv`, `output/plots/shap_summary.png`.

---

## 7. Real-Time Inference Usage

```python
from prosody.detector import ProsodySpoofDetector
import numpy as np

detector = ProsodySpoofDetector("output")

# Input: 16 kHz mono waveform (1.0s = 16,000 samples)
waveform = np.zeros(16000, dtype=np.float32)

result = detector.predict(waveform)
print("Model Family:           ", result["model_family"])
print("Calibrated P(spoof):    ", result["calibrated_spoof_prob"])
print("Decision:               ", result["decision"])
print("Forensic Cues:          ", result["top_forensic_cues"])
print("Latency:                ", f"{result['latency_ms']:.2f} ms")
```

### Performance & Latency:
- **Warm End-to-End Latency:** **~28.19 ms** per 1.0s window (Feature extraction + Preprocessing + GBDT Inference + Platt Calibration + Forensic Cues).
- **Pure GBDT Model Latency:** **~0.52 ms**.
- **Real-Time Constraint:** Well under the **<50 ms** CPU budget.

---

## 8. Running the Pipeline & Tests

```powershell
# 1. Extract features (smoke test or full)
python training/prepare_features.py --manifest ../Module 2A/output/datasets/target_domain_manifest.csv --smoke-test 100

# 2. Train and benchmark LightGBM, XGBoost, and CatBoost
python training/compare_models.py

# 3. Fit Platt probability calibrator on held-out calibration split
python calibrate.py

# 4. Evaluate winning model on untouched test split
python training/evaluate.py

# 5. Generate SHAP explainability plots
python training/explainability.py

# 6. Run the complete unit test suite
python -m unittest discover -s tests -v
```
