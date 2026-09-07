# Module 2D — frozen Wav2Vec2 XLS-R 300M embeddings

Module 2D is a pretrained multilingual speech-representation branch. It loads
`facebook/wav2vec2-xls-r-300m` once from `../models/xlsr-300m/`, keeps it in
`eval()` mode with gradients disabled, and mean-pools temporal hidden states
into one Float32 `(1024,)` embedding per speech window.

XLS-R is self-supervised multilingual speech representation learning. It is
**not** a deepfake detector, is not trained specifically for voice cloning, and
does not produce `P(AI)` or a spoof probability. The baseline does not
fine-tune it. Its only public result is an embedding for a future fusion model:

```text
Module 1 16 kHz speech chunk → XLS-R temporal hidden states → mean pool → 1024-D embedding → future fusion
```

This complements AASIST anti-spoofing evidence and future
spectral/prosodic features. Neither fusion nor a final decision is implemented
here.

## Input and padded-tail behavior

Module 1 already performs the only VAD, segmentation, sliding-window creation,
and normalization. Its 1-second windows use a 0.5-second hop and its manifest
contains the original `id`, timestamps, duration, `padded`, and region index.
Module 2D validates that a manifest specifies 16 kHz mono audio and does not
resample those chunks. For standalone callers, `prepare_waveform` safely
downmixes 2-D audio, resamples to 16 kHz, converts to Float32, and rejects
empty/non-finite input without applying loudness or peak normalization.

Padded Module 1 WAVs remain in the output. During extraction their manifest
`duration` determines `valid_length`, and the known zero-filled tail is removed
from XLS-R's temporal mean-pooling mask. Natural silence within the true source
duration is still retained.

## Run

Run from this directory:

```bash
python verify.py
python extract_embeddings.py --module1-dir "../Module 1/output/real_voice" --dataset demo_real --label 0 --output-dir output/embeddings/real
python extract_embeddings.py --module1-dir "../Module 1/output/synthetic_voice" --dataset demo_synthetic --label 1 --output-dir output/embeddings/synthetic
python benchmark.py --device cpu --repetitions 10 --batch-size 1
python -m unittest discover -s tests -v
```

`extract_embeddings.py` writes one dense memory-mapped `embeddings.npy` matrix
(not many small files), `metadata.csv`, and `summary.json`. Every metadata row retains
the dataset, source recording ID/path, chunk ID, start/end/duration, padding,
region, optional label, speaker/generator placeholders, and the exact matrix
row via `embedding_index`. Keep all rows sharing `source_id` together in any
downstream train/validation/calibration/test split; never split chunks from a
recording across partitions.

The existing project datasets should first be passed through Module 1, then
this command is run over the resulting source-manifest directories (repeat
`--module1-dir` to keep XLS-R loaded once). This intentionally reuses the
single speech-segmentation authority rather than duplicating VAD or window
generation. Embeddings are written incrementally in batches; a dataset-scale
runner must preserve the original dataset/source/speaker/generator IDs into the
same storage schema.

`download_model.py --confirm` is the only network-enabled model download path;
regular inference is local-only and will never silently fetch the approximately
1.27 GB weights.

## Fine-tuning and probes

Frozen XLS-R is the Module 2D baseline. A future, explicitly scoped experiment
may compare a classifier on frozen embeddings with fine-tuned XLS-R, but does
not belong in this module. A logistic-regression/SVM probe can be used solely
as a `diagnostic_probe` to assess information in frozen embeddings; it is not
the Module 2D detector and does not change this interface.
