# MODEL MANIFEST

This document specifies the exact ML models, local paths, versions, configurations, licenses, and input/output formats for the real-time AI voice-cloning detection prototype.

---

## 1. Silero VAD (Voice Activity Detection)
* **Model Name:** Silero VAD v5 / v6
* **Source URL:** [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)
* **Exact Identifier:** `snakers4/silero-vad` (package `silero-vad==6.2.1`)
* **Local Path:** `models/silero_vad/silero_vad.onnx` / `models/silero_vad/silero_vad.jit`
* **Model Size:** ~4.7 MB (ONNX/JIT)
* **Expected Sample Rate:** 16,000 Hz (16 kHz)
* **Input Format:** 1D Float32 Audio Tensor/Array `(samples,)` or 2D `(1, chunk_size)`
* **Output Format:** Float Speech Probability $\in [0.0, 1.0]$ / List of timestamp dicts `[{"start": int, "end": int}]`
* **License:** MIT License
* **Purpose in Pipeline:** Real-time CPU-optimized voice activity detection, streaming audio segmentation, and speech window generation (1–3 second overlapping windows).
* **Pretrained or Local:** Pretrained (Official Release)
* **Revision / Version:** `v6.2.1`

---

## 2. AASIST Anti-Spoofing Branch
* **Model Name:** AASIST (Audio Anti-Spoofing with Integrated Spectro-Temporal Graph Attention Networks)
* **Source URL:** [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist)
* **Exact Identifier:** `clovaai/aasist` (`AASIST.pth`)
* **Local Path:** `models/aasist/AASIST.pth`
* **Model Size:** 1.22 MB
* **Expected Sample Rate:** 16,000 Hz (16 kHz)
* **Input Format:** 1D Float32 Tensor `(1, 64600)` (~4 seconds audio window)
* **Output Format:**
  * Raw Logits: `(1, 2)` (spoof vs. bona-fide logits)
  * Raw Anti-Spoofing Score: $S = \text{logit}_{\text{bonafide}} - \text{logit}_{\text{spoof}}$
  * Last Hidden Representation: `(160,)` Float32 Vector
* **License:** MIT License
* **Purpose in Pipeline:** Pretrained anti-spoofing branch providing deep spectral-temporal graph representation and raw anti-spoofing evidence scores.
* **Pretrained or Local:** Pretrained on ASVspoof2019 Logical Access (LA) dataset.
* **Note:** *Pretrained on ASVspoof2019 LA; not specifically trained for Indian English.*
* **Revision / Commit:** `main` branch commit `b68019b787595ca1db9be9635b719430c1f28b4c`

---

## 3. Wav2Vec2 XLS-R 300M (Speech Representation Extractor)
* **Model Name:** Wav2Vec2 XLS-R 300M
* **Source URL:** [https://huggingface.co/facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
* **Exact Identifier:** `facebook/wav2vec2-xls-r-300m`
* **Local Path:** `models/xlsr-300m/`
* **Model Size:** ~1.27 GB (`pytorch_model.bin`)
* **Expected Sample Rate:** 16,000 Hz (16 kHz)
* **Input Format:** 1D Float32 Audio Waveform `(samples,)` processed via `AutoFeatureExtractor`
* **Output Format:** 1024-dimensional Float32 Mean-Pooled Hidden Embedding Vector `(1024,)`
* **License:** Apache License 2.0
* **Purpose in Pipeline:** Self-supervised cross-lingual speech representation feature extractor. *Used strictly as an embedding extractor feeding into the feature fusion classifier, NOT directly connected to final P(AI).*
* **Pretrained or Local:** Pretrained (Meta / Facebook AI)
* **Revision / Commit:** `fce4599a0996843468f3a3a71b12b591b65554f6`

---

## 4. Handcrafted Spectral + Prosodic Features
* **Model Name:** Handcrafted Acoustic Extractor
* **Source URL:** Local Implementation (`src/acoustic_features/feature_extractor.py`)
* **Local Path:** `src/acoustic_features/`
* **Model Size:** N/A (Algorithmic / Signal Processing)
* **Expected Sample Rate:** 16,000 Hz (16 kHz)
* **Input Format:** 1D Float32 Audio Waveform `(samples,)`
* **Output Format:** Fixed-length 75-dimensional Float32 Feature Vector:
  * **Spectral (66 dims):** MFCC (40), Mel Spectrogram Stats (2), Spectral Centroid (2), Bandwidth (2), Rolloff (2), Flux (2), ZCR (2), Spectral Contrast (14)
  * **Prosodic (9 dims):** RMS Energy Stats (2), F0 Mean/Std/Min/Max (4), Voicing Ratio (1), Jitter (1), Shimmer (1)
* **License:** Project Code
* **Purpose in Pipeline:** Domain-informed acoustic and physiological voice perturbation feature extraction.
* **Pretrained or Local:** Local algorithmic implementation using `librosa`, `scipy`, and `parselmouth` (Praat).

---

## 5. Feature Fusion Classifier
* **Model Name:** Feature Fusion Classifier Prototype
* **Local Path:** `src/fusion/fusion_classifier.py`
* **Purpose:** Fuses AASIST score/vector (160 dims), XLS-R embedding (1024 dims), and Handcrafted Acoustic Vector (75 dims) into a unified representation and predicts window-level $P(\text{AI} \mid \text{window}) \in (0, 1)$.

---

## 6. Wald Sequential Probability Ratio Test (SPRT)
* **Model Name:** Wald SPRT Decision Engine
* **Local Path:** `src/sprt/wald_sprt.py`
* **Purpose:** Sequential evidence aggregator operating on $P(\text{AI} \mid \text{window})$ window sequence. Outputs decision states: `SAFE`, `INVESTIGATE`, or `HIGH RISK`.
