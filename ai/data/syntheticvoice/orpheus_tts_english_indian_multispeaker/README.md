---
dataset_info:
  features:
  - name: audio
    dtype:
      audio:
        sampling_rate: 24000
  - name: text
    dtype: string
  - name: source
    dtype: int64
  splits:
  - name: train
    num_bytes: 3471261488.916
    num_examples: 13306
  download_size: 3150662891
  dataset_size: 3471261488.916
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
