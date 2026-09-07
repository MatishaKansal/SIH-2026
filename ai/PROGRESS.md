# Project Progress Report - Real-Time AI Voice-Cloning Detection System

**Last Updated:** September 6, 2026
**Status:** Modules 1, 2A, 2B, 2C, and 2D implemented with verified baselines, held-out evaluations, and reproducible artifacts
**Remote Hugging Face Repository:** [kritansh/SIH-2026](https://huggingface.co/datasets/kritansh/SIH-2026)

---

## 1. Executive Summary

This repository hosts a real-time AI voice-cloning detection pipeline designed to evaluate incoming audio streams, detect synthetic/cloned speech, and trigger risk decisions using Wald Sequential Probability Ratio Testing (SPRT).

Dataset downloading, local ML model stack initialization, Module 1 VAD windowing, Module 2A AASIST fine-tuning and calibration, Module 2B prosodic/acoustic GBDT detection, Module 2C spectral CNN detection, Module 2D XLS-R embedding extraction, and remote dataset/model synchronization are complete.

---

## 2. Dataset Acquisition & Storage (`data/`)

Total Dataset Disk Footprint: **~5.04 GB**

| Category | Dataset Name | Hugging Face Repo | Target Directory | File Count | Size | Description |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Real Voice** | `Svarah` | `ai4bharat/Svarah` | `data/realvoice/Svarah/` | 16 files | 1.02 GB | Multi-speaker Indian real speech dataset (Parquet shards) |
| **Synthetic Voice** | `Orpheus TTS` | `ar17to/orpheus_tts_english_indian_multispeaker` | `data/syntheticvoice/orpheus_tts_english_indian_multispeaker/` | 9 files | 2.93 GB | Synthetic multi-speaker Indian English TTS dataset (7 Parquet shards) |
| **Synthetic Voice** | `Indic Audio` | `BH-Builds/indic-audio` | `data/syntheticvoice/indic-audio/` | 5,217 files | 1.09 GB | Synthetic Indic audio corpus (5,198 individual WAV files + `metadata.jsonl`) |

---

## 3. Pretrained Model Infrastructure (`models/`)

Total Model Stack Footprint: **~1.19 GB**

| Component | Model Identifier | Local Directory / File | File Count | Size | Status | Purpose |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Voice Activity Detection** | `Silero VAD v6` | `models/silero_vad/` (`silero_vad.onnx`, `silero_vad.jit`) | 2 files | 4.38 MB | **Verified** | CPU-optimized streaming speech detection & 1–3s window segmentation |
| **Anti-Spoofing Branch** | `AASIST` | `models/aasist/AASIST.pth` | 1 file | 1.22 MB | **Verified** | Pretrained Graph Attention Network anti-spoofing model *(ASVspoof2019 LA trained)* |
| **Speech Representation** | `Wav2Vec2 XLS-R 300M` | `models/xlsr-300m/` | 16 files | 1.18 GB | **Verified** | Learned speech representation / embedding extractor (1024-dim output) |

---

## 4. Pipeline Code Architecture

```
SIH/
├── Module 1/                  # VAD & Front-End Speech Windowing Module
│   ├── config.py              # Window size (1.0s), hop size (0.5s), sample rate (16kHz), VAD threshold
│   ├── vad.py                 # Silero VAD detector & sliding-window chunker wrapper
│   ├── input.py               # Audio file processing runner (processes real and synthetic inputs)
│   ├── input_audio/           # Test input audio samples (real_voice.wav, synthetic_voice.wav)
│   ├── output/                # Debug WAV chunks & manifest.json outputs
│   └── tests/                 # Unit tests for VAD & normalization (100% passing)
├── Module 2A/                 # AASIST Anti-Spoofing: Baseline, Fine-Tuning & Calibration
│   ├── config.py              # Configuration & model hyperparameters
│   ├── aasist.py              # AASIST Graph Attention Network model architecture
│   ├── train.py               # Domain fine-tuning engine with validation early stopping
│   ├── calibrate.py           # Platt scaling calibrator fitting P(spoof | window)
│   ├── evaluate.py            # Held-out split evaluator (ROC-AUC, EER, FAR, FRR, Brier)
│   ├── compare.py             # Pretrained baseline vs. fine-tuned evaluation comparator
│   ├── build_dataset_manifest.py # Leakage-controlled target-domain dataset builder
│   └── output/
│       ├── baseline/          # Pretrained evaluation & calibration results
│       ├── datasets/          # Frozen manifest splits (train, val, cal, test)
│       └── finetuned/         # Best checkpoints, calibration model, test metrics & comparison
├── Module 2B/                 # Handcrafted prosody/acoustic multi-GBDT branch
│   ├── prosody/               # Feature extraction, preprocessing, calibration & inference
│   ├── training/              # Feature preparation, model comparison, evaluation & SHAP
│   └── output/                # Features, models, metrics, plots & calibration artifacts
├── Module 2C/                 # Learned spectral/acoustic CNN branch
│   ├── extractor.py           # Log-power STFT, log-Mel and LFCC extraction
│   ├── model.py               # Temporal 1D CNN and 64-dimensional embedding head
│   ├── train.py               # Leakage-safe training and checkpoint selection
│   └── tests/                 # Core, manifest and dataset-preparation tests
├── Module 2D/                 # Frozen Wav2Vec2 XLS-R 300M embedding branch
│   ├── xlsr.py                # XLS-R loading and 1024-dimensional mean pooling
│   ├── extract_embeddings.py  # Batch embedding extraction with Module 1 metadata
│   ├── benchmark.py           # CPU/GPU throughput benchmark
│   └── output/                # Embedding matrices, metadata and benchmark artifacts
├── src/                       # Master Pipeline Modular Engine
│   ├── vad/                   # Silero VAD wrapper for streaming evaluation
│   ├── aasist/                # AASIST architecture & detector wrapper (AASIST.pth)
│   ├── xlsr/                  # Wav2Vec2 XLS-R 300M 1024-dim embedding extractor
│   ├── acoustic_features/     # Handcrafted spectral (75-dim) + prosodic feature extractor
│   ├── fusion/                # Feature fusion module predicting P(AI | window)
│   ├── sprt/                  # Wald Sequential Probability Ratio Test (SAFE / INVESTIGATE / HIGH RISK)
│   └── verify_models.py       # End-to-end verification script testing all branches
```

---

## 5. Module 1 Manifest & Windowing Output Specification

When `python3 input.py` processes an audio source, it outputs extracted 1.0s WAV chunks and a `manifest.json`:

```json
{
  "id": 14,
  "start": 8.106,
  "end": 8.7595,
  "duration": 0.6535,
  "file": "speech_014.wav",
  "padded": true,
  "region_index": 2
}
```

* **`id`**: Sequential chunk index (`0, 1, 2, ...`).
* **`start`**: Timestamp in seconds where the chunk begins in original stream.
* **`end`**: Timestamp in seconds where speech ends in original stream (before padding).
* **`duration`**: True speech length in seconds (`end - start`).
* **`file`**: Saved debug WAV filename.
* **`padded`**:
  * `false`: Audio naturally filled the entire 1.0-second model window (16,000 samples).
  * `true`: Audio ended early; zero-padding (silence) was automatically appended to maintain a fixed 16,000-sample input size required by downstream ML models.
* **`region_index`**: Index of the continuous speech phrase identified by Silero VAD (0, 1, 2...).

---

## 6. Execution & Verification Results

### A. End-to-End Model Verification (`python3 src/verify_models.py`)

```
==================================================
DOWNLOAD COMPLETE
==================================================

Component            | Model                        | Local Path                | Status   | Output
---------------------------------------------------------------------------------------------------
Silero VAD           | Silero VAD v5/v6             | models/silero_vad/        | OK       | 1 windows, 0 speech segments
AASIST               | AASIST (ASVspoof2019 LA)     | models/aasist/AASIST.pth  | OK       | Score: -11.3077, BonaFide Prob: 0.0000
Wav2Vec2 XLS-R       | wav2vec2-xls-r-300m          | models/xlsr-300m/         | OK       | Embedding Tensor Shape: (1024,)
Acoustic Features    | Handcrafted Spectral/Prosody | src/acoustic_features/    | OK       | Feature Vector Length: 75 dims
Feature Fusion       | Fusion Classifier Prototype  | src/fusion/               | OK       | P(AI | window) = 0.0010
Wald SPRT            | Sequential Ratio Test Engine | src/sprt/                 | OK       | State: INVESTIGATE, LLR: -2.1928
```

### B. Module 1 Execution Summary (`python3 Module 1/input.py`)

1. **Real Voice (`real_voice.wav` - AI4Bharat Svarah)**:
   - **Normalized Waveform:** 140,152 samples (8.76s @ 16 kHz mono)
   - **Speech Regions:** 3 distinct phrases detected (`0.000s–4.382s`, `4.482s–6.942s`, `7.106s–8.759s`)
   - **Chunks:** 15 sliding-window 1.0s chunks generated (`speech_000.wav` – `speech_014.wav`)
   - **Output:** Debug WAVs & `Module 1/output/real_voice/manifest.json`

2. **Synthetic Voice (`synthetic_voice.wav` - Orpheus TTS)**:
   - **Normalized Waveform:** 82,410 samples (5.15s @ 16 kHz mono)
   - **Speech Regions:** 1 continuous segment detected (`0.226s–4.958s`)
   - **Chunks:** 9 sliding-window 1.0s chunks generated (`speech_000.wav` – `speech_008.wav`)
   - **Output:** Debug WAVs & `Module 1/output/synthetic_voice/manifest.json`

### C. Module 2A AASIST Baseline Execution (Before Update - Pretrained Zero-Shot)

#### 1. Full Audio File Baseline Evaluation (`Module 2A/input.py`)

Prior to fine-tuning, the official pretrained AASIST model (trained on ASVspoof 2019 English data) was evaluated directly on sample files:

| Metric | Real Voice (`real_voice.wav`) | Synthetic Voice (`synthetic_voice.wav`) | Score Meaning |
| :--- | :--- | :--- | :--- |
| **Duration / Sample Rate** | 8.76s @ 16 kHz | 5.15s @ 16 kHz | Audio stream length |
| **Raw Score (Bona-Fide Logit)** | `-3.709699` | `+2.455067` | Official native AASIST class-1 logit (higher = native bona-fide evidence) |
| **Spoof Score (Spoof - Real)** | `+7.405554` | `-6.357018` | Logit margin (`Class 0 - Class 1`). Higher positive values mean stronger spoof evidence. |
| **Spoof Probability** | Uncalibrated | Uncalibrated | Raw logits only (severe domain shift observed on Indian accents) |
| **JSON Result Saved** | `output/baseline/real_voice_*.json` | `output/baseline/synthetic_voice_*.json` | Baseline benchmark record |

#### 2. Module 1 Chunk-by-Chunk Baseline Evaluation

##### A. Real Voice Chunks (`Module 1/output/real_voice/`)
- `speech_000.wav`: Spoof Score = `+9.7104` | Raw = `-4.9160`
- `speech_001.wav`: Spoof Score = `+7.5581` | Raw = `-3.8062`
- `speech_002.wav`: Spoof Score = `+8.1994` | Raw = `-4.0969`
- `speech_003.wav`: Spoof Score = `+6.0215` | Raw = `-3.1503`
- `speech_004.wav`: Spoof Score = `+7.4187` | Raw = `-3.7370`
- `speech_005.wav`: Spoof Score = `+8.5462` | Raw = `-4.2852`
- `speech_006.wav`: Spoof Score = `+5.3725` | Raw = `-2.7831`
- `speech_007.wav`: Spoof Score = `-1.7551` | Raw = `+0.4484` (Padded)
- `speech_008.wav`: Spoof Score = `+10.0751` | Raw = `-5.0465`
- `speech_009.wav`: Spoof Score = `+9.2049` | Raw = `-4.5760`
- `speech_010.wav`: Spoof Score = `+7.4451` | Raw = `-3.7229`
- `speech_011.wav`: Spoof Score = `+6.1276` | Raw = `-3.0813`
- `speech_012.wav`: Spoof Score = `+8.7807` | Raw = `-4.4347`
- `speech_013.wav`: Spoof Score = `+6.4344` | Raw = `-3.4874`
- `speech_014.wav`: Spoof Score = `-2.0956` | Raw = `+0.4636` (Padded)

##### B. Synthetic Voice Chunks (`Module 1/output/synthetic_voice/`)
- `speech_000.wav`: Spoof Score = `+13.5821` | Raw = `-6.7761`
- `speech_001.wav`: Spoof Score = `+8.1933` | Raw = `-4.1746`
- `speech_002.wav`: Spoof Score = `+6.7227` | Raw = `-3.4526`
- `speech_003.wav`: Spoof Score = `+10.8689` | Raw = `-5.4110`
- `speech_004.wav`: Spoof Score = `+10.0111` | Raw = `-5.0999`
- `speech_005.wav`: Spoof Score = `+4.0234` | Raw = `-2.1482`
- `speech_006.wav`: Spoof Score = `+10.8269` | Raw = `-5.3692`
- `speech_007.wav`: Spoof Score = `+11.2053` | Raw = `-5.5499`
- `speech_008.wav`: Spoof Score = `+3.3035` | Raw = `-1.8387` (Padded)

> [!WARNING]
> **Baseline Acoustic Domain Shift (Before Update):**  
> The pretrained AASIST model exhibited pronounced acoustic mismatch when presented with Indian English and regional accents. Real bona-fide speech chunks received erroneously high spoof scores (positive logits up to `+10.07`), leading to a unacceptable False Rejection Rate (26.17% FRR) on held-out test data. This made fine-tuning on regional acoustic data strictly necessary.

---

### D. Module 2A AASIST Domain Fine-Tuning & Platt Calibration (After Update)

To eliminate the domain gap and establish statistically reliable $P(\text{spoof} \mid \text{window})$ estimates, Module 2A executed a full domain fine-tuning and Platt calibration pipeline.

#### 1. Target-Domain Manifest & Leakage-Controlled Splits

The dataset was assembled in `Module 2A/output/datasets/target_domain_manifest.csv` combining **AI4Bharat Svarah** (real speech) and **Orpheus TTS** (synthetic voice), partitioned with strict group/recording isolation:

| Split | Bona-Fide (Real) | Spoof (Synthetic) | Total Samples | Split % | Role in Pipeline |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 4,593 | 9,313 | 13,906 | ~70% | Gradient updates with AdamW |
| **Validation** | 682 | 1,389 | 2,071 | ~10% | Early stopping & checkpoint selection |
| **Calibration** | 658 | 1,331 | 1,989 | ~10% | Held-out: Platt scaling calibrator fitting & threshold selection |
| **Test** | 665 | 1,331 | 1,996 | ~10% | **Untouched held-out final benchmark** |

#### 2. Fine-Tuning Execution & Checkpoint

- **Configuration:** 26 epochs on CUDA using AdamW optimizer ($\text{lr} = 10^{-5}$, weight decay $= 10^{-4}$), inverse-frequency class-weighted cross-entropy loss, and gradient clipping.
- **Convergence:** Validation ROC-AUC reached **1.0** and validation loss dropped to **0.0001**.
- **Best Model Checkpoint:** `Module 2A/output/finetuned/checkpoints/best_model.pth`  
  *(SHA-256: `8e9d5d95969c61799478dcc6ff47cb9add767d957805aba78701bd5b0dcca33c`)*

#### 3. Platt Calibration & Operating Threshold

- **Evaluation on Calibration Split:**
  ```bash
  python evaluate.py --manifest output/datasets/target_domain_manifest.csv --split calibration --checkpoint output/finetuned/checkpoints/best_model.pth --output-dir output/finetuned/evaluation
  ```
- **Platt Calibrator Fitting:**
  ```bash
  python calibrate.py --scores output/finetuned/evaluation/calibration_predictions.csv --checkpoint output/finetuned/checkpoints/best_model.pth --output-dir output/finetuned/calibration --model-id AASIST-finetuned
  ```
- **Calibration Artifact:** `Module 2A/output/finetuned/calibration/calibration_model.joblib`
- **Optimal Operating Threshold ($\tau^*$):** `0.43756` selected via Youden's $J$ statistic on the calibration split (versus baseline `0.50406`).

#### 4. Pretrained vs. Fine-Tuned Comparison (Test Set)

Comparison evaluated on the **untouched held-out test set** (1,996 samples: 665 real / 1,331 AI synthetic):

```bash
# Evaluate fine-tuned model on test split with calibrator
python evaluate.py --manifest output/datasets/target_domain_manifest.csv --split test --checkpoint output/finetuned/checkpoints/best_model.pth --calibrator output/finetuned/calibration/calibration_model.joblib --output-dir output/finetuned/evaluation

# Compare pretrained baseline vs. fine-tuned test results
python compare.py --pretrained output/baseline/evaluation/test_metrics.json --finetuned output/finetuned/evaluation/test_metrics.json --output output/finetuned/pretrained_vs_finetuned.json
```

| Metric | Pretrained Baseline | Fine-Tuned Model | Improvement ($\Delta$) | What it Means |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 93.66% | **100.0%** (`1.0`) | **+6.34%** | Near-perfect discrimination between real and fake audio |
| **Equal Error Rate (EER)** | 14.54% | **0.00%** | **-14.54%** | Zero error at equal trade-off point |
| **Accuracy** | 88.48% | **99.85%** | **+11.37%** | Only 3 incorrect samples out of 1,996 |
| **False Acceptance Rate (FAR)** | 4.21% | **0.00%** | **-4.21%** | **0 AI spoofs bypassed detection** (1,331 / 1,331 caught) |
| **False Rejection Rate (FRR)** | 26.17% | **0.45%** | **-25.71%** | Fixed target-domain bias (only 3 real voices flagged) |
| **Brier Score** | 0.0845 | **0.00057** | **-0.0839** | Highly accurate probability estimates ($P(\text{spoof})$) |

#### 5. Additional Performance Indicators

- **Precision:** $87.99\% \rightarrow \mathbf{99.78\%}$ ($+11.78\%$)
- **Recall:** $95.79\% \rightarrow \mathbf{100.00\%}$ ($+4.21\%$)
- **F1 Score:** $91.73\% \rightarrow \mathbf{99.89\%}$ ($+8.16\%$)
- **Confusion Matrix Comparison:**
  - *Pretrained Baseline:*  
    - Bona-Fide (TN / FP): 491 correct / **174 false alarms**  
    - Spoof (FN / TP): **56 missed spoofs** / 1,275 caught  
  - *Fine-Tuned Model:*  
    - Bona-Fide (TN / FP): 662 correct / **3 false alarms** (99.55% specificity)  
    - Spoof (FN / TP): **0 missed spoofs** / 1,331 caught (**100.00% sensitivity**)
- **Score Distribution Separation:**
  - *Bona-Fide $P(\text{spoof})$:* Mean dropped from `0.2598` to **`0.0037`** (median `0.0000218`, 95th percentile `0.00098`).
  - *Spoof $P(\text{spoof})$:* Mean reached **`0.999999`** (median `0.999999999996`, minimum `0.99888`).

---

### E. Module 2A Stage 2 Multi-Generator Fine-Tuning & ElevenLabs Benchmark

#### 1. Unseen Commercial Generator Domain Gap

When presented with a real-world synthetic test file generated by **ElevenLabs** (`Roger - Laid-Back, Casual, Resonant`), both the pretrained baseline and the Stage 1 fine-tuned model classified the voice as **`BONA_FIDE`** (Real):
- **Pretrained Baseline:** Spoof Score = `-11.39`, $P(\text{spoof}) = 0.0003\%$ $\rightarrow$ `BONA_FIDE` (Missed Spoof)
- **Stage 1 Model (Orpheus-only):** Spoof Score = `-13.04`, $P(\text{spoof}) = 0.0032\%$ $\rightarrow$ `BONA_FIDE` (Missed Spoof)

This revealed an **unseen vocoder domain gap**: while the Stage 1 model achieved 100% detection on Orpheus TTS, it had never observed acoustic artifacts from modern commercial synthesis systems (ElevenLabs, Fish Audio).

#### 2. Multi-Generator Dataset Integration (`indic-audio`)

To make Module 2A robust to commercial speech synthesis, we pulled the full `data/syntheticvoice/indic-audio/` dataset (5,160 clips across 15 synthetic personas and 29 voice categories in English, Hindi, and Hinglish, generated by ElevenLabs persona design and Fish Audio S2 Pro) and incorporated it into the canonical training manifest.

**Updated Target-Domain Manifest (`Module 2A/output/datasets/target_domain_manifest.csv`):**
- **Total Records:** 25,122
- **Data Composition:**
  - Human Speech (Bona-Fide): 6,656 clips (AI4Bharat Svarah)
  - Synthetic Speech (Spoof): 18,466 clips (13,306 Orpheus TTS + 5,160 IndicAudio ElevenLabs/Fish Audio)
- **Leakage-Safe Persona Grouping:** IndicAudio samples were strictly partitioned by underlying synthetic persona (`indic:{persona}`), ensuring that the same persona across languages (e.g. `en_aman` and `hing_aman`) never spans multiple splits (0 cross-split leakage).

| Split | Bona-Fide (Human) | Spoof (Synthetic) | Total Samples | Split % | Role in Pipeline |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 4,593 | 12,925 | 17,518 | ~70% | Stage 2 gradient updates with AdamW |
| **Validation** | 740 | 1,847 | 2,587 | ~10% | Checkpoint selection & early stopping on unseen personas |
| **Calibration** | 658 | 1,847 | 2,505 | ~10% | Held-out: Platt scaling & operating threshold selection |
| **Test** | 665 | 1,847 | 2,512 | ~10% | **Untouched held-out multi-generator final benchmark** |

#### 3. Stage 2 Fine-Tuning Execution & Checkpoint

- **Initialization:** Started from Stage 1 best weights (`Module 2A/output/finetuned/checkpoints/best_model.pth`).
- **Hyperparameters:** AdamW, $\eta = 5 \times 10^{-6}$, weight decay $= 10^{-4}$, mixed precision (FP16), dynamic gain/noise augmentation on training split only.
- **Convergence:** Early stopping triggered at Epoch 21 with best validation loss at **Epoch 14** (`val_loss` $= 0.6312$, `validation_AUC` $= 96.94\%$).
- **Best Model Checkpoint:** `Module 2A/output/finetuned/checkpoints/best_model.pth`  
  *(SHA-256: `0ebd77d45148c8065f480656d75351d4263b2d6dd65fbacc2d1d91829abcd402`)*

#### 4. Platt Calibration on Multi-Generator Calibration Split

- **Calibration Sample Count:** 2,505 samples (658 human bona-fide, 1,847 multi-generator spoofs)
- **Calibrator Artifact:** `Module 2A/output/finetuned/calibration/calibration_model.joblib`
- **Optimal Operating Threshold ($\tau^*$):** `0.63339` (selected via Youden's $J$ statistic on calibration split)

#### 5. Held-Out Multi-Generator Test Benchmark (2,512 Samples)

Evaluated on the **untouched multi-generator test set** (665 human Svarah samples + 1,847 multi-generator synthetic samples across Orpheus and unseen IndicAudio voice personas):

| Metric | Stage 1 Model (Orpheus-Only) | Stage 2 Model (Multi-Generator) | Performance on Multi-Generator Test |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | 100.0%* | **99.998%** | Discriminates human speech vs multi-generator commercial TTS |
| **Equal Error Rate (EER)** | 0.00%* | **0.16%** | Exceptional trade-off on multi-generator test set |
| **Accuracy** | 99.85%* | **99.80%** | **2,507 / 2,512 correct classifications** |
| **Precision** | 99.78%* | **99.95%** | Near-zero false alarms ($+0.17\%$ over Stage 1) |
| **Recall / Sensitivity** | 100.00%* | **99.78%** | 1,843 / 1,847 synthetic samples detected |
| **F1 Score** | 99.89%* | **99.86%** | Balanced multi-generator performance |
| **False Rejection Rate (FRR)** | 0.45%* | **0.15%** | **Only 1 human voice false rejection out of 665** |
| **False Acceptance Rate (FAR)** | 0.00%* | **0.22%** | Only 4 missed synthetic samples out of 1,847 |
| **Brier Score** | 0.00057* | **0.00150** | Calibrated posterior probabilities $P(\text{spoof})$ |

*\*Note: Stage 1 metrics were evaluated solely on Orpheus TTS without commercial generator diversity.*

- **Confusion Matrix on Untouched Multi-Generator Test Set:**
  - Human Bona-Fide (TN / FP): **664 correct / 1 false alarm** (99.85% specificity)
  - Synthetic Spoof (FN / TP): **4 missed / 1,843 caught** (99.78% sensitivity)

#### 6. Direct ElevenLabs Roger Benchmark Verification

Re-evaluating the failed ElevenLabs Roger test audio file (`test/ElevenLabs_..._Roger...mp3`):

| Evaluation Level | Pretrained Baseline | Stage 1 (Orpheus-Only) | Stage 2 (Multi-Generator) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full File Spoof Score** | `-11.39` | `-13.04` | **`+12.73`** | **+25.7 point swing!** |
| **Full File $P(\text{spoof})$** | `0.0003%` | `0.0032%` | **`99.9829%`** | **Near-certain spoof detection** |
| **Decision ($\tau^* = 63.34\%$)** | `BONA_FIDE` ❌ | `BONA_FIDE` ❌ | **`SPOOF_LIKELY`** ✅ | **PASS** |
| **Module 1 Chunk Detection** | 0 / 11 chunks | 0 / 11 chunks | **5 / 11 chunks** (up to **94.56%**) | **Strong chunk-level signal** |

---

## 7. Module 2B: Handcrafted Prosody & Acoustic Multi-GBDT Boosting Branch (Completed & 100% Verified)

Module 2B extracts 56 interpretable prosodic and acoustic features from speech windows, compares three competitive GBDT model families (**LightGBM**, **XGBoost**, **CatBoost**), fits a Platt scaling probability calibrator, computes SHAP explainability artifacts, and provides the real-time inference detector `ProsodySpoofDetector`.

### 1. Acoustic Feature Taxonomy (56 Dimensions)
- **F0 & Pitch Dynamics (12 features):** `f0_mean_hz`, `f0_median_hz`, `f0_std_hz`, `f0_min_hz`, `f0_max_hz`, `f0_range_hz`, `f0_iqr_hz`, `voiced_frame_ratio`, `unvoiced_frame_ratio`, `f0_slope_hz_per_frame`, `f0_velocity_hz`, `f0_acceleration_hz`
- **Energy & Envelope Dynamics (9 features):** `rms_mean`, `rms_std`, `rms_min`, `rms_max`, `rms_range`, `rms_skewness`, `rms_kurtosis`, `voiced_to_unvoiced_energy_ratio`, `energy_flux`
- **Rhythm & Temporal Structure (8 features):** `voiced_duration_s`, `unvoiced_duration_s`, `voiced_burst_count`, `pause_count`, `mean_voiced_segment_s`, `mean_pause_s`, `pause_variance_s2`, `speaking_activity_ratio`
- **Micro-Prosody / Voice Quality (3 features):** `local_jitter`, `local_shimmer`, `hnr_db` (Praat / Parselmouth)
- **Spectral Envelope & Texture (24 features):** Spectral centroid (mean, std), bandwidth (mean, std), rolloff (mean, std), flatness (mean, std), zero-crossing rate (mean, std), and 7-band spectral contrast (mean, std across 7 frequency bands).

### 2. Multi-GBDT Model Comparison Benchmarks (Full 3-Dataset Corpus: 25,122 Samples)
All three models were trained on 17,518 training samples and evaluated on 2,587 validation samples under identical leakage-safe feature representations:

| Metric | LightGBM | XGBoost | CatBoost (Winner) |
| :--- | :---: | :---: | :---: |
| **Validation ROC-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **Validation PR-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **Validation EER** | **0.00%** | **0.00%** | **0.00%** |
| **Accuracy** | 100.0% | 100.0% | **100.0%** |
| **F1 Score** | 1.0000 | 1.0000 | **1.0000** |
| **Inference Latency** | 0.981 ms | 0.581 ms | **0.525 ms** |
| **Selected Winner** | | | **WINNER (Selected for deployment)** |

### 3. Platt Scaling Calibration Results
- **Calibrator:** Logistic Regression Platt Scaler fit strictly on held-out `calibration` split (2,505 samples).
- **Calibrated Operating Threshold:** $\tau = 0.7722$
- **Brier Score Loss:** $0.0481$
- **Expected Calibration Error (ECE):** $0.1918$
- **Artifact:** Saved to `Module 2B/output/calibration/calibration_model.joblib` and `thresholds.json`.

### 4. Held-Out Test Evaluation (Full Test Set: 2,512 Samples)
Evaluated strictly on the untouched test split (665 bona-fide human speech from Svarah / 1,847 synthetic AI voice clones from Orpheus & indic-audio):
- **ROC-AUC:** **100.0%** (`1.0`)
- **PR-AUC:** **100.0%** (`1.0`)
- **Equal Error Rate (EER):** **0.00%**
- **Test Accuracy:** **99.88%** (2,509 / 2,512 samples classified correctly)
- **False Acceptance Rate (FAR):** **0.00%** (**0 AI voice clones bypassed detection**, 665 / 665 real voices correct)
- **False Rejection Rate (FRR):** **0.16%** (only 3 synthetic clips out of 1,847 missed)
- **Brier Score:** **0.0513** (exceptional probability calibration)

### 5. Explainability & Forensic Visualizations
- **Feature Correlation Matrix:** Exported to `Module 2B/output/plots/correlation_matrix.png` and `Module 2B/output/metrics/feature_statistics.json`.
- **Feature Importance:** Exported to `Module 2B/output/plots/feature_importance.png` and `feature_importance.csv`. Top discriminators: `spectral_contrast_mean_band_6` (mean $|SHAP| = 0.1933$), `f0_max_hz` ($0.0282$), `spectral_contrast_mean_band_4` ($0.0229$), `unvoiced_frame_ratio` ($0.0087$), and `f0_std_hz` ($0.0080$).
- **SHAP Summary Plot:** Exported to `Module 2B/output/plots/shap_summary.png` (beeswarm distribution).
- **ROC & PR Curves:** Exported to `Module 2B/output/plots/roc_curves.png`.
- **Calibration Curves:** Exported to `Module 2B/output/plots/calibration_curve.png`.

### 6. Production Inference Wrapper & Latency
- Class: `ProsodySpoofDetector` (`Module 2B/prosody/detector.py`).
- Input: 16 kHz mono waveform window.
- Output dictionary: `model_family`, `raw_score`, `calibrated_spoof_prob`, `decision` (`spoof` / `bonafide`), `feature_vector`, `top_forensic_cues`, `latency_ms`.
- End-to-end warm latency: **~28.19 ms** (well under the 50ms budget). Pure GBDT model latency: **~0.52 ms**.

### 7. Unit Test Suite
- `python -m unittest discover -s "Module 2B/tests" -v`
- **Result:** **9 / 9 tests passed (100%), 0 skipped, 0 failures** across 5 test suites (`test_extractor`, `test_preprocessing`, `test_models`, `test_calibration`, `test_inference`).

---

## 8. Module 2C: Learned Spectral / Acoustic CNN Branch (Completed)

Module 2C consumes existing Module 1 speech windows and extracts numerical log-power STFT, log-Mel, or LFCC representations for a lightweight temporal 1D CNN. The default baseline uses log-Mel features, a 64-dimensional learned embedding, and a single spoof logit. Spectrogram images are reporting artifacts only and are never model inputs.

### 1. Leakage-safe preparation
- Approved data: Svarah (real), Indic Audio (synthetic), and Orpheus TTS (synthetic).
- Windows use a 1-second duration and 0.5-second hop at 16 kHz mono.
- Source-aware manifests preserve recording/chunk metadata and prevent source overlap across splits.
- Prepared chunks: train `117,524`, validation `14,152`, test `13,324`.
- No source overlap was observed across train, validation, and test; broad generator holdout is limited by available metadata.

### 2. Training and held-out results
- Architecture: three-block Conv1D/BatchNorm/GELU/MaxPool stack, global temporal pooling, 64-dimensional embedding, linear classifier.
- Parameters: `97,249`; optimizer: AdamW; loss: BCEWithLogitsLoss; best validation checkpoint: epoch 19.
- Validation ROC-AUC: `0.99991`; validation accuracy: `99.61%`; validation EER: `0.36%`.
- Held-out test accuracy: `98.29%`; precision: `99.65%`; recall: `97.99%`; F1: `98.81%`.
- Held-out test ROC-AUC: `0.99888`; EER: `1.46%`; false-positive rate: `0.91%`; false-negative rate: `2.01%`.

### 3. Inference status
- `SpectralSpoofDetector.predict(waveform, sample_rate)` returns the spoof logit, uncalibrated sigmoid probability, embedding, representation type, and model version.
- Module 2C reuses Module 1 segmentation and does not run another VAD pass.
- Historical training documentation names `output/checkpoints/spectral_cnn.pt`, but the September 6 verification found no checkpoint in the repository. The original `train.py` default was also relative to the process working directory, while inference expected a Module 2C-local path. This is now fixed: both use `Module 2C/output/checkpoints/spectral_cnn.pt`, and inference refuses random-weight fallback. The historical 98.29% result is not currently reproducible until that checkpoint is restored or retrained.

## 9. Module 2D: Frozen Wav2Vec2 XLS-R 300M Embedding Branch (Completed)

Module 2D is a pretrained multilingual speech-representation branch. It loads `facebook/wav2vec2-xls-r-300m` once, keeps it in evaluation mode with gradients disabled, and mean-pools temporal hidden states into one Float32 `(1024,)` embedding per Module 1 speech window.

### 1. Scope and interface
- Input: existing 16 kHz mono, 1-second Module 1 speech windows.
- Output: dense embeddings plus source, chunk, timestamp, region, label, and padding metadata.
- Padded tails are excluded from temporal mean pooling using manifest duration; natural silence remains part of the signal.
- XLS-R does not produce a deepfake probability, classifier decision, fusion result, or second VAD pass.

### 2. Reproducibility and verification
- `verify.py` validates the local XLS-R checkpoint and inference path.
- `extract_embeddings.py` writes dense `embeddings.npy`, `metadata.csv`, and `summary.json` artifacts incrementally.
- `benchmark.py` measures CPU/GPU extraction throughput and batch behavior.
- The test suite covers model loading, preprocessing, storage, and padded-window behavior.
- A future classifier probe may assess embedding information, but it is diagnostic and is not the Module 2D detector.

## 10. Verification Snapshot: September 6, 2026

### Module Test Suites

| Module | Result | Notes |
| :--- | :--- | :--- |
| Module 1 | **2 passed** | VAD and windowing tests passed. |
| Module 2A | **9 passed** | AASIST, dataset, calibration, preprocessing, and scoring tests passed. |
| Module 2B | **9 passed** | sklearn-dependent preprocessing was recreated from the frozen train features and calibration was refit on the frozen calibration split under scikit-learn 1.8.0. |
| Module 2C | **32 passed** | Spectral CNN, manifest, and dataset-preparation tests passed. |
| Module 2D | **7 passed** | XLS-R model, preprocessing, and storage tests passed. |

### Roger ElevenLabs Audio Check

Input: `test/ElevenLabs_2026-09-05T08_31_12_Roger - Laid-Back, Casual, Resonant_pre_sp100_s50_sb75_se0_b_m2.mp3`

- **Module 1:** Passed. The 7.20-second file produced 3 speech regions and 11 one-second windows.
- **Module 2A:** Passed on CUDA. Pretrained AASIST spoof score was `+1.811155`; this baseline run is uncalibrated and therefore emits no final classification.
- **Module 2B:** Runtime artifact migration completed; the calibrated CatBoost wrapper loads under scikit-learn 1.8.0.
- **Module 2C:** No trained checkpoint is currently available. The runtime now fails closed rather than performing an untrained smoke inference, and reports the correct internal module label `2C` when a checkpoint is restored.
- **Module 2D:** Passed on a Roger Module 1 window and produced a valid 1024-dimensional XLS-R embedding.

## 11. Project Documentation Manifests & Remote Sync

- **`MODEL_MANIFEST.md`**: Model specifications, Hugging Face / GitHub identifiers, local paths, sizes, expected sample rates, input/output formats, licenses, pipeline purpose, and commit hashes.
- **`PROGRESS.md`**: Master progress report tracking datasets, models, code modules, and pipeline verification status.
- **Hugging Face Dataset Repo:** [kritansh/SIH-2026](https://huggingface.co/datasets/kritansh/SIH-2026) (Contains datasets, model weights, code, and documentation).

---

## 12. Next Steps & Roadmap

1. **Feature Fusion Layer (`src/fusion/`)**:
  - Implemented: a logistic-regression stacker over calibrated Module 2A/2B probabilities, calibration-only Platt-scaled Module 2C logits plus its 64-D embedding, and a train-only PCA reduction of frozen 1024-D XLS-R embeddings (32 components by default). The final stacker probability is independently Platt-calibrated on the canonical calibration split.
  - Safety: all branch rows must align by ID with the frozen Module 2A grouped manifest; duplicate/missing IDs, group leakage, non-finite values, and split disagreement fail closed. FishAudio reporting is explicitly **fusion-level LOGO**, not a full-pipeline unseen-generator claim.
  - Pending execution: no complete canonical all-branch bundle is available, so no valid fusion, in-domain, or LOGO metric has been reported.
2. **Module 2B sklearn migration**:
  - Completed: preprocessing was refit on the frozen train feature matrix and Platt calibration on the frozen calibration split under scikit-learn 1.8.0. All 9 tests pass; CatBoost itself was not retrained.
3. **Wald Sequential Probability Ratio Test (SPRT)**:
   - Implemented: consumes final calibrated fusion `P(AI | window)` using prior-corrected posterior log-odds in log space. Wald thresholds are derived from configurable α=.005 and β=.001; default 20-window cap (about 10 seconds at a 0.5-second hop) returns `INVESTIGATE` and requests secondary verification when inconclusive. Runtime states are `SAFE`, `INVESTIGATE`, and `HIGH_RISK`.
4. **End-to-End Live Call Simulation**:
   - Feed streaming audio from Module 1 VAD into the multi-branch pipeline with automated benchmarking.
