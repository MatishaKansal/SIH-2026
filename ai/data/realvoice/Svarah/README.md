---
dataset_info:
  features:
  - name: audio_filepath
    dtype: audio
  - name: duration
    dtype: float64
  - name: text
    dtype: string
  - name: gender
    dtype: string
  - name: age-group
    dtype: string
  - name: primary_language
    dtype: string
  - name: native_place_state
    dtype: string
  - name: native_place_district
    dtype: string
  - name: highest_qualification
    dtype: string
  - name: job_category
    dtype: string
  - name: occupation_domain
    dtype: string
  splits:
  - name: test
    num_bytes: 1088823937.104
    num_examples: 6656
  download_size: 1094998590
  dataset_size: 1088823937.104
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# *Svarah*: An Indic Accented English Speech Dataset

<div style="display: flex; gap: 5px;">
  <a href="https://github.com/AI4Bharat/Lahaja"><img src="https://img.shields.io/badge/GITHUB-black?style=flat&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://arxiv.org/abs/2408.11440"><img src="https://img.shields.io/badge/arXiv-2411.02538-red?style=flat" alt="ArXiv"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="CC BY 4.0"></a>
</div>

## Dataset Description

- **Homepage:** [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Repository:** [Github](https://github.com/AI4Bharat/Svarah)
- **Paper:** [Svarah: Evaluating English ASR Systems on Indian Accents](https://arxiv.org/abs/2305.15760)

## Overview

India is the second largest English-speaking country in the world, with a speaker base of roughly 130 million. Unfortunately, Indian speakers are underrepresented in many existing English ASR benchmarks such as LibriSpeech, Switchboard, and the Speech Accent Archive.

To address this gap, we introduce **Svarah**—a benchmark that comprises 9.6 hours of transcribed English audio from 117 speakers across 65 districts in 19 states of India, representing a diverse range of accents. The native languages of the speakers cover 19 of the 22 constitutionally recognized languages of India, spanning 4 language families. *Svarah* includes both read speech and spontaneous conversational data, covering domains such as history, culture, tourism, government, sports, as well as real-world use cases like ordering groceries, digital payments, and accessing government services (e.g., checking pension claims or passport status).

We evaluated 6 open-source ASR models and 2 commercial ASR systems on *Svarah*, demonstrating clear scope for improvement in handling Indian accents.


This work is funded by Bhashini, MeitY and Nilekani Philanthropies

## Usage

The [datasets](https://huggingface.co/docs/datasets) library enables you to load and preprocess the dataset directly in Python. Ensure you have an active HuggingFace access token (obtainable from [Hugging Face settings](https://huggingface.co/settings/tokens)) before proceeding.

To load the dataset, run:

```python
from datasets import load_dataset
# Load the dataset from the HuggingFace Hub
dataset = load_dataset("ai4bharat/Svarah",split="test")
# Check the dataset structure
print(dataset)
```

You can also stream the dataset by enabling the `streaming=True` flag:

```python
from datasets import load_dataset
dataset = load_dataset("ai4bharat/Svarah",split="test", streaming=True)
print(next(iter(dataset)))
```

## Citation

If you use Svarah in your work, please cite us:

```bibtex
@inproceedings{DBLP:conf/interspeech/JavedJNSNRBKK23,
  author       = {Tahir Javed and
                  Sakshi Joshi and
                  Vignesh Nagarajan and
                  Sai Sundaresan and
                  Janki Nawale and
                  Abhigyan Raman and
                  Kaushal Santosh Bhogale and
                  Pratyush Kumar and
                  Mitesh M. Khapra},
  title        = {Svarah: Evaluating English {ASR} Systems on Indian Accents},
  booktitle    = {{INTERSPEECH}},
  pages        = {5087--5091},
  publisher    = {{ISCA}},
  year         = {2023}
}
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

For any questions or feedback, please contact:
- Tahir Javed (tahir@cse.iitm.ac.in)

