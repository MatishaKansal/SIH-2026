# Data Manifest — Real-Time AI Voice-Cloning Detection System

**Last Updated:** September 4, 2026  
**Total Dataset Disk Footprint:** **~5.04 GB**  
**Remote Hugging Face Repository Mirror:** [kritansh/SIH-2026](https://huggingface.co/datasets/kritansh/SIH-2026)

---

## 1. Executive Summary

This manifest documents all real human speech (bona-fide) and synthetic AI voice-cloned (spoof) datasets acquired, configured, and integrated into the real-time AI voice-cloning detection system.

The dataset corpus provides:
1. **Diverse Real Human Speech**: Multi-speaker Indian accented English and regional language speech across 117 speakers, 19 states, and 65 districts in India.
2. **Multi-Speaker Synthetic Speech**: Advanced text-to-speech (TTS) voice-cloned speech generated across multi-speaker models (Orpheus TTS, ElevenLabs, Fish Audio S2 Pro) spanning Hindi, Indian English, and Hinglish.

---

## 2. Master Dataset Summary

| Category | Dataset Name | Hugging Face Hub Repository | Local Storage Directory | Format & Storage | Sample Rate | Utterances / Clips | Disk Size | Primary Languages / Accents | License |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Real Voice** | `Svarah` | [`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah) | `data/realvoice/Svarah/` | Parquet (3 shards) | 16,000 Hz | 6,656 clips (9.61 hrs) | **1.02 GB** | Indian Accented English & 19 regional Indic languages | `CC-BY-4.0` |
| **Synthetic Voice** | `Orpheus TTS` | [`ar17to/orpheus_tts_english_indian_multispeaker`](https://huggingface.co/datasets/ar17to/orpheus_tts_english_indian_multispeaker) | `data/syntheticvoice/orpheus_tts_english_indian_multispeaker/` | Parquet (7 shards) | 24,000 Hz | 13,306 clips | **2.93 GB** | Multi-speaker Indian English TTS | `Apache 2.0` |
| **Synthetic Voice** | `Indic Audio` | [`BH-Builds/indic-audio`](https://huggingface.co/datasets/BH-Builds/indic-audio) | `data/syntheticvoice/indic-audio/` | Raw WAV / MP3 + `metadata.jsonl` | 44,100 Hz / 16,000 Hz | 5,198 clips (7.13 hrs) | **1.09 GB** | Hindi (Devanagari), Indian English, Hinglish (Romanized) | Open Access / Synthetic Terms |

---

## 3. Detailed Dataset Specifications

### A. Real Voice Dataset: `ai4bharat/Svarah`

* **Repository URL:** [https://huggingface.co/datasets/ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
* **Creators:** AI4Bharat (IIT Madras), funded by Bhashini, MeitY, and Nilekani Philanthropies.
* **Paper / Citation:** *Svarah: Evaluating English ASR Systems on Indian Accents* (INTERSPEECH 2023).
* **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
* **Local Storage Path:** `data/realvoice/Svarah/`

#### Key Characteristics & Demographics
* **Speech Volume:** **9.61 hours** of transcribed audio across **6,656 audio clips**.
* **Speaker Diversity:** 117 speakers from **65 districts across 19 states of India**.
* **Language Representation:** Covers **19 of the 22 constitutionally recognized languages of India** across 4 language families.
* **Domain Coverage:** History, culture, tourism, government services, digital payments, grocery ordering, and spontaneous conversational speech.

#### File Structure & Schema
```
data/realvoice/Svarah/
├── README.md
└── data/
    ├── test-00000-of-00003.parquet
    ├── test-00001-of-00003.parquet
    └── test-00002-of-00003.parquet
```

#### Metadata Field Definitions
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `audio_filepath` | `struct` | Struct containing raw WAV byte payload (`bytes`) and original relative filename (`path`). |
| `duration` | `float64` | Audio clip length in seconds. |
| `text` | `string` | Verbatim text transcript of the spoken sentence. |
| `gender` | `string` | Speaker gender (`Female` / `Male`). |
| `age-group` | `string` | Age bracket of the speaker (e.g. `18-30`). |
| `primary_language` | `string` | Native language of the speaker (e.g. `Kannada`, `Hindi`, `Tamil`, `Bengali`). |
| `native_place_state` | `string` | State of origin in India (e.g. `Karnataka`, `Maharashtra`). |
| `native_place_district`| `string` | District of origin in India (e.g. `Hassan`). |
| `highest_qualification`| `string` | Speaker education level (e.g. `Post Graduate`). |
| `job_category` | `string` | Employment status (e.g. `Full Time`). |
| `occupation_domain` | `string` | Professional domain (e.g. `Technology and Services`). |

#### Pipeline Role
Provides ground-truth **real human speech (bona-fide)** baseline samples for:
* Voice Activity Detection (VAD) benchmark tests (`Module 1`).
* Anti-spoofing feature extraction (AASIST, Wav2Vec2 XLS-R 300M, 75-dim acoustic vectors).
* Training and calibrating feature fusion classifiers to avoid False Rejections (FRR) on genuine Indian accents.

---

### B. Synthetic Voice Dataset 1: `ar17to/orpheus_tts_english_indian_multispeaker`

* **Repository URL:** [https://huggingface.co/datasets/ar17to/orpheus_tts_english_indian_multispeaker](https://huggingface.co/datasets/ar17to/orpheus_tts_english_indian_multispeaker)
* **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* **Local Storage Path:** `data/syntheticvoice/orpheus_tts_english_indian_multispeaker/`

#### Key Characteristics
* **Speech Volume:** **13,306 synthetic speech clips** across 7 Parquet shards.
* **Audio Format:** 24,000 Hz mono audio (automatically resampled to 16,000 Hz by `Module 1` / `src/` wrappers).
* **Synthetic Voice Persona:** Multi-speaker synthetic Indian English text-to-speech voice generation.

#### File Structure & Schema
```
data/syntheticvoice/orpheus_tts_english_indian_multispeaker/
├── README.md
└── data/
    ├── train-00000-of-00007.parquet
    ├── train-00001-of-00007.parquet
    ├── train-00002-of-00007.parquet
    ├── train-00003-of-00007.parquet
    ├── train-00004-of-00007.parquet
    ├── train-00005-of-00007.parquet
    └── train-00006-of-00007.parquet
```

#### Metadata Field Definitions
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `audio` | `struct` | Struct containing raw audio bytes (`bytes`), sampling rate (`24000`), and file metadata. |
| `text` | `string` | Text transcript used to generate the synthetic audio clip. |
| `source` | `int64` | Numeric source index identifying synthetic speaker / generator configuration. |

#### Pipeline Role
Serves as large-scale **synthetic multi-speaker Indian English spoofing data** to train deep neural representation models (Wav2Vec2 XLS-R) and AASIST graph attention models to identify subtle TTS artifacts in Indian English accents.

---

### C. Synthetic Voice Dataset 2: `BH-Builds/indic-audio`

* **Repository URL:** [https://huggingface.co/datasets/BH-Builds/indic-audio](https://huggingface.co/datasets/BH-Builds/indic-audio)
* **Creators:** BH-Builds (Developers of `goonj-1-82M`).
* **License:** Open Synthetic Terms / Open Access.
* **Local Storage Path:** `data/syntheticvoice/indic-audio/`

#### Key Characteristics
* **Speech Volume:** **5,198 clips** (**7.13 hours** of synthetic speech, average 5.0s per clip).
* **Voice Personas:** **15 synthetic voices** (14 named personas + 1 Hindi "language bed").
* **Language & Script Breakdown:**
  * **Devanagari Hindi:** `bed_hindi` (1,900 clips, 3.22 hrs) + 4 personas (`hi_atul`, `hi_meera`, `hi_ravi`, `hi_shivani` - 960 clips, 1.55 hrs).
  * **Indian English:** 10 personas (`en_aman`, `en_ananya`, `en_arjun`, `en_dev`, `en_divya`, `en_kabir`, `en_nisha`, `en_priya`, `en_sameer`, `en_tara` - 1,600 clips, 1.75 hrs).
  * **Casual Hinglish:** Romanized Hindi in Latin script across all 14 personas (700 clips, 0.60 hrs).

#### File Structure & Schema
```
data/syntheticvoice/indic-audio/
├── README.md
├── metadata.jsonl
├── ref_texts.json
├── audio/
│   ├── bed_hindi/
│   ├── en_aman/
│   ├── en_ananya/
│   ├── en_arjun/
│   ├── hi_meera/
│   └── ... (15 persona directories containing .wav/.mp3 clips)
└── scripts/
    └── ... (text sentence pools)
```

#### Metadata Field Definitions (`metadata.jsonl`)
```json
{
  "audio": "audio/hi_meera/0000.mp3",
  "text": "अगली सुबह बाबा अजय से मिले...",
  "voice": "hi_meera"
}
```
* **`audio`**: Relative path to the generated audio clip.
* **`text`**: Source text prompt synthesized.
* **`voice`**: Persona identifier (`bed_hindi`, `hi_*`, `en_*`, `hing_*`).

#### Pipeline Role
Provides **multi-lingual Indic synthetic speech (Hindi, Hinglish, Indian English)** to ensure the anti-spoofing pipeline generalizes across regional Indian languages and code-switched conversational dialects.

---

## 4. Integration into Real-Time Pipeline

The datasets are integrated into the processing workflow as follows:

```
[ Raw Audio Datasets ]
 ├── Real Voice: ai4bharat/Svarah (6,656 clips)
 └── Synthetic Voice: Orpheus TTS (13,306 clips) + Indic Audio (5,198 clips)
         │
         ▼
[ Module 1 / VAD Windowing ]
 ├── Resampling to 16 kHz Mono Float32
 ├── Silero VAD Speech Activity Detection & Silence Removal
 └── 1.0s Sliding-Window Segmentation (hop size: 0.5s)
         │
         ▼
[ Parallel Feature Extraction ]
 ├── AASIST (Graph Attention Network Anti-Spoofing Score)
 ├── Wav2Vec2 XLS-R 300M (1024-dim Learned Speech Representation)
 └── Acoustic Feature Extractor (75-dim Spectral & Prosodic Vectors)
         │
         ▼
[ Feature Fusion & Wald SPRT Decision Engine ]
 └── Predicts P(AI | window) -> SAFE / INVESTIGATE / HIGH RISK
```

---

## 5. Remote Mirror & Cloud Synchronization

All dataset files, Parquet shards, audio WAVs, metadata files, and READMEs are backed up and hosted on the Hugging Face Dataset Hub:

* **Remote Dataset Repository:** [https://huggingface.co/datasets/kritansh/SIH-2026](https://huggingface.co/datasets/kritansh/SIH-2026)
* **Local Workspace Path:** `/home/kirat/coding/SIH/data/`
