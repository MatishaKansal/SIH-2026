# Module 2C: Spectral / Acoustic Deepfake Detection

Module 2C is an independent learned spectral-evidence branch for audio deepfake detection. Its input is one existing Module 1 speech window; it does not perform a second VAD pass.

```text
Module 1 Silero VAD
        ↓
existing speech window
        ↓
numerical spectral representation
        ↓
temporal 1D CNN
        ↓
learned spectral embedding + spoof logit
        ↓
future fusion
```

The branch does not modify Module 1, Module 2A, Module 2B, XLS-R, AASIST, feature fusion, SPRT, frontend, WebRTC, SIP, or blocking logic.

---

## Representations

Module 2C supports three numerical time-frequency representations:

* **Log-power STFT** — short-time Fourier transform followed by logarithmic power compression.
* **Log-Mel** — STFT power projected through a Mel filterbank followed by logarithmic compression.
* **LFCC** — linear-frequency filterbank energies followed by logarithmic compression and a DCT-II transform.

These representations preserve time-frequency evidence that can be useful for detecting synthetic speech, including spectral discontinuities, unstable harmonics, spectral holes, and vocoder-related artifacts.

The representations are stored and processed as **numerical tensors**. PNG/JPEG spectrogram images are never used as model inputs.

This is distinct from:

```text
src/acoustic_features/feature_extractor.py
```

which produces handcrafted scalar acoustic features including MFCC, Mel statistics, spectral centroid, bandwidth, rolloff, onset, ZCR, spectral contrast, F0, jitter, and shimmer.

### Default configuration

```text
Sample rate:      16 kHz
Channels:         mono
n_fft:            512
win_length:       400
hop_length:       160
Mel bands:        64
LFCC coefficients: 20
Window length:    1 second
Embedding dim:    64
```

The CNN treats frequency bins as input channels and performs its convolutions across the temporal dimension.

Model interface:

```python
embedding, logits = model(features)
```

The classifier uses:

```text
BCEWithLogitsLoss
```

The output is a single spoof logit:

```text
spoof_score = raw spoof logit
```

Higher values indicate stronger spoof evidence.

The inference API also reports:

```text
uncalibrated_spoof_probability = sigmoid(spoof_logit)
```

This probability is **not calibrated**.

---

## Model Architecture

The spectral branch uses a lightweight temporal 1D CNN:

```text
Input spectral tensor
        ↓
Conv1D
        ↓
BatchNorm
        ↓
GELU
        ↓
MaxPool
        ↓
Conv1D
        ↓
BatchNorm
        ↓
GELU
        ↓
MaxPool
        ↓
Conv1D
        ↓
BatchNorm
        ↓
GELU
        ↓
Global temporal pooling
        ↓
64-dimensional embedding
        ↓
linear classifier
        ↓
spoof logit
```

The trained baseline contains **97,249 parameters**.

The architecture is intentionally lightweight so that spectral evidence can eventually be used as one branch of a larger real-time multimodal/deepfake detection system.

---

## Data

Module 2C uses the three datasets approved for this project:

1. **Svarah**
2. **BH-Builds/indic-audio**
3. **Orpheus TTS English Indian Multispeaker**

No additional training dataset was introduced.

### Svarah

Source:

```text
ai4bharat/Svarah
```

Verified local dataset:

```text
Clips:       6,656
Duration:    ~9.61 hours
Sample rate: 16 kHz
Channels:    mono
Label:       real (0)
```

The currently verified Svarah metadata does not provide a reliable filename-to-speaker-ID mapping. Therefore speaker identity is **not inferred from filenames** and speaker grouping is not claimed where the mapping is unavailable.

### BH-Builds/indic-audio

Source:

```text
BH-Builds/indic-audio
```

Verified local dataset:

```text
Audio files: 5,160
Duration:    ~7.13 hours
Sample rate: 44.1 kHz
Channels:    mono
Label:       synthetic/spoof (1)
```

The dataset contains 15 verified persona/grouping units. The dataset documentation identifies Fish Audio S2 Pro as the synthesis system used for bulk generation.

### Orpheus TTS English Indian Multispeaker

Source:

```text
ar17to/orpheus_tts_english_indian_multispeaker
```

Verified local dataset:

```text
Audio files: 13,306
Duration:    ~20.07 hours
Sample rate: 24 kHz
Channels:    mono
Label:       synthetic/spoof (1)
```

The available metadata does not establish a reliable speaker identity or generator field. The `source` field is therefore not promoted to speaker or generator metadata.

---

## Dataset Preparation

`prepare_datasets.py` prepares the three approved datasets for Module 2C.

The preparation pipeline:

1. Validates the local datasets.
2. Assigns original recordings to splits.
3. Preserves source-level grouping.
4. Creates one-second windows.
5. Uses a 0.5-second hop.
6. Resamples audio to the model's 16 kHz mono format during loading/extraction.
7. Preserves source and chunk metadata.
8. Writes deterministic train/validation/test manifests.

The resulting manifests are:

```text
data/module2b/module2b_manifest.csv
data/module2b/module2b_train.csv
data/module2b/module2b_val.csv
data/module2b/module2b_test.csv
data/module2b/module2b_dataset_report.json
```

The directory name `module2b` is retained for compatibility with the existing dataset-preparation artifacts; the implementation itself is **Module 2C**.

Prepared split sizes:

| Split      | Total chunks |   Real |  Spoof |
| ---------- | -----------: | -----: | -----: |
| Train      |      117,524 | 30,342 | 87,182 |
| Validation |       14,152 |  3,907 | 10,245 |
| Test       |       13,324 |  3,628 |  9,696 |

The preparation produced:

```text
Train sources:       20,340
Validation sources:   2,497
Test sources:         2,285
```

No `source_id` overlap was observed across train/validation/test.

Known speaker-group overlap was also absent where speaker metadata was available.

A generator holdout could not be meaningfully performed across all datasets because only the Indic Audio generator is explicitly verified in the available metadata.

---

## Leakage-Safe Splitting

`split_dataset.py` provides source-aware splitting.

The primary grouping key is:

```text
source_id
```

This prevents overlapping chunks originating from the same recording from being distributed across different splits.

When verified speaker metadata exists:

```text
speaker_id
```

can be used for speaker-grouped splitting.

Missing values such as:

```text
unavailable
unknown
```

are not treated as valid speaker identities.

The implementation does not infer speaker identity from:

* filenames
* directory names unless explicitly defined as verified metadata
* demographic attributes
* arbitrary metadata fields

Generator holdout is supported through generator-aware records and validates generator disjointness when sufficient metadata exists.

`split_integrity_report()` reports:

* sample counts
* duration
* source counts
* speaker-group counts
* generator counts
* cross-split source overlap
* cross-split speaker overlap
* cross-split generator overlap

---

## Training

Training is performed with:

```text
Optimizer:       AdamW
Loss:            BCEWithLogitsLoss
Representation:  log-Mel
Device:          Apple Silicon MPS
Model params:    97,249
```

Training uses deterministic seed configuration and saves the best checkpoint based on validation performance.

Training augmentation is optional and applied only to the training dataset.

Validation and test data remain unaugmented.

### Training command

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/train.py" \
--split-dir data/module2b \
--spectral-type logmel
```

The completed baseline training run reached 20 epochs.

The best validation checkpoint was obtained at **epoch 19**.

---

## Training Results

The best validation checkpoint achieved:

```text
Validation accuracy:       99.61%
Validation precision:      99.72%
Validation recall:         99.75%
Validation F1:             99.73%
Validation real recall:    99.26%
Validation spoof recall:   99.75%
Validation FPR:             0.74%
Validation FNR:             0.25%
Validation ROC-AUC:         0.99991
Validation EER:             0.36%
```

The validation confusion matrix was:

```text
                  Predicted
                Real   Spoof

Actual Real     3878     29
Actual Spoof      26  10219
```

The validation results are used for model selection; they should not be interpreted as external generalization performance.

---

## Held-Out Test Evaluation

The selected checkpoint was evaluated on the held-out test manifest:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/evaluate.py" \
output/checkpoints/spectral_cnn.pt \
--manifest data/module2b/module2b_test.csv
```

### Test results

```text
Accuracy:          98.29%
Precision:         99.65%
Recall:            97.99%
F1:                98.81%

Real recall:       99.09%
Spoof recall:      97.99%

False positive rate: 0.91%
False negative rate: 2.01%

ROC-AUC:           0.99888
EER:               1.46%
```

Test confusion matrix:

```text
                  Predicted
                Real   Spoof

Actual Real     3595     33
Actual Spoof     195   9501
```

The test set therefore contained:

```text
Real:   3,628
Spoof:  9,696
Total: 13,324
```

These are held-out results from the prepared project dataset. They are **not** external In-The-Wild or unseen-generator generalization results.

---

## Checkpoint

The trained checkpoint is:

```text
output/checkpoints/spectral_cnn.pt
```

Best checkpoint:

```text
Best validation epoch: 19
Representation:        log-Mel
Parameters:             97,249
```

The checkpoint is currently a local training artifact and is not included in the Module 2C source-code commit.

---

## Inference API

The main inference interface is:

```python
SpectralSpoofDetector.predict(waveform, sample_rate)
```

The API returns:

```text
spoof_logit
spoof_score
uncalibrated_spoof_probability
embedding
spectral_type
model_version
```

Manifest-level inference is also supported through:

```python
predict_manifest_record(...)
```

Module 1 metadata is preserved where available, including:

```text
source
source_id
chunk_id
start
end
padded
region_index
```

This allows the spectral branch to produce outputs that can later be aligned with other detection branches during fusion.

Module 2C does **not** perform another VAD pass during inference.

---

## Module 1 Integration

The integration path accepts an existing Module 1 manifest:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/integration.py" \
"../Module 1/output/real_voice/manifest.json" \
--checkpoint output/checkpoints/spectral_cnn.pt
```

The integration layer consumes the already-created Module 1 speech windows.

It does not:

* rerun Silero VAD
* create a second speech segmentation
* modify Module 1
* modify Module 2A
* modify Module 2B

---

## Visualization

`visualize_spectra.py` provides reporting visualizations for comparing real and synthetic spectral representations.

Example:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/visualize_spectra.py" sample.wav
```

The generated images are for visualization/reporting only.

They are not used as CNN training inputs.

---

## Dataset Acquisition and Verification

To acquire the three approved datasets locally:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/acquire_local_datasets.py" \
--output-root data/raw
```

Verify the downloaded data:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/verify_local_datasets.py" \
--svarah-path data/raw/svarah \
--indic-audio-path data/raw/indic-audio \
--orpheus-path data/raw/orpheus
```

The acquisition pipeline does not create training chunks.

Preparation is performed separately through:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/prepare_datasets.py" \
--local-only \
--svarah-path data/raw/svarah \
--indic-audio-path data/raw/indic-audio \
--orpheus-path data/raw/orpheus \
--output-dir data/module2b \
--seed 42
```

The recommended workflow is `--local-only`; it does not use Hugging Face streaming.

---

## Evaluation Commands

Evaluate the trained checkpoint:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/evaluate.py" \
output/checkpoints/spectral_cnn.pt \
--manifest data/module2b/module2b_test.csv
```

Benchmark inference:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/benchmark.py" \
sample.wav \
--checkpoint output/checkpoints/spectral_cnn.pt
```

Compare representations:

```bash
PYTHONPATH="Module 2C" .venv/bin/python \
"Module 2C/compare.py"
```

The STFT/log-Mel/LFCC comparison experiment has **not yet been run**, so no representation winner is claimed.

---

## Testing

The Module 2C test suite covers:

* 16 kHz normalization
* stereo-to-mono conversion
* tensor dimensions
* NaN/Inf protection
* short audio
* silence
* padded windows
* model forward pass
* embedding dimensions
* dataset loading
* label handling
* source-grouped splitting
* Module 1 manifest compatibility
* checkpoint save/load
* deterministic inference

Current local test result:

```text
32 passed
```

---

## Augmentation

Training augmentation is opt-in.

Supported lightweight augmentations include:

* additive noise
* gain variation
* light reverberation
* resampling perturbation
* codec-style perturbation where practical

Augmentation is applied only to training samples.

Validation and test samples are never augmented.

---

## Benchmarking

`benchmark.py` is provided to measure:

* spectral extraction time
* CNN inference time
* total per-chunk latency
* parameter count
* checkpoint size
* CPU/GPU latency where available

A complete real-time latency benchmark has **not yet been recorded**, so no latency claim is made.

---

## Scientific Integrity

Results are reported only when the corresponding experiment has actually been executed.

The following have been completed:

```text
Dataset verification                 DONE
Dataset preparation                  DONE
Leakage-safe source splitting        DONE
Log-Mel training                     DONE
Best checkpoint selection            DONE
Held-out test evaluation             DONE
Unit/integration tests               DONE
```

The following remain pending:

```text
STFT vs log-Mel vs LFCC comparison   NOT RUN
Unseen-generator evaluation          NOT RUN
External In-The-Wild evaluation      NOT RUN
Real-time latency benchmark          NOT RUN
Probability calibration              NOT RUN
```

No unexecuted metric, checkpoint, latency measurement, or external generalization result is claimed.

---

## Current Status

| Component                          | Status                               |
| ---------------------------------- | ------------------------------------ |
| Module 1 integration               | Coded; compatible                    |
| Audio preprocessing                | Coded; tested                        |
| STFT / log-Mel / LFCC extraction   | Coded; log-Mel used for baseline     |
| Real/synthetic visualization       | Coded; available                     |
| Dataset acquisition                | Completed locally                    |
| Dataset verification               | Completed                            |
| Dataset preparation                | Completed                            |
| Leakage-safe splitting             | Completed                            |
| Spectral 1D CNN                    | Completed                            |
| Training                           | Completed                            |
| Best checkpoint                    | `output/checkpoints/spectral_cnn.pt` |
| Held-out test evaluation           | Completed                            |
| STFT vs log-Mel vs LFCC comparison | NOT RUN                              |
| Unseen-generator evaluation        | NOT RUN                              |
| External In-The-Wild evaluation    | NOT RUN                              |
| Probability calibration            | NOT RUN                              |
| Real-time benchmark                | NOT RUN                              |
| Future multimodal fusion           | Not implemented                      |

---

## Scope Boundaries

Module 2C intentionally does **not** implement:

```text
XLS-R
ECAPA
AASIST
Prosody branch
Feature fusion
SPRT
Frontend
WebRTC
SIP
Blocking logic
```

Its sole responsibility is learned spectral/acoustic evidence from existing Module 1 speech windows.

The intended future interface is:

```text
Module 1
   ↓
speech window
   ↓
Module 2A ──┐
Module 2B ──┼──→ future fusion
Module 2C ──┘
```

The spectral branch provides:

```text
learned spectral embedding
+
spoof logit
+
chunk-level metadata
```

for downstream fusion and decision-making.
