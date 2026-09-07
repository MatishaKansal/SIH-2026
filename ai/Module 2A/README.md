# Module 2A — AASIST baseline, fine-tuning, and calibration

Module 2A accepts a speech chunk from Module 1, validates and normalizes it to
16 kHz mono float32, then runs the **unchanged pretrained AASIST**
anti-spoofing checkpoint. It produces baseline anti-spoofing evidence only.

It does **not** fine-tune a model, perform speaker verification, use XLS-R,
make a final risk decision, run SPRT, or block calls.

## Baseline inference

Set `INPUT_AUDIO` near the top of `config.py`, then run from this directory:

```bash
python input.py
```

`DEVICE="auto"` selects CUDA when available and CPU otherwise. Use `"cpu"` to
force CPU. The canonical checkpoint is `../models/aasist/AASIST.pth`; its
SHA-256 is written to `output/baseline/model_metadata.json` on every run.

## Score convention

The official AASIST ASVspoof protocol labels class `1` as `bonafide` and class
`0` as `spoof`; its evaluator saves class-1's logit. Module 2A reports that
native class-1 logit as `raw_score`. Its project-standard `spoof_score` is
`logit[0] - logit[1]`, so higher values mean stronger spoof evidence.

Neither value is a probability. `spoof_probability` is always `null`, and no
classification is emitted while `SPOOF_THRESHOLD` remains `None` because this
pretrained model has not been calibrated for the target audio domain.

The output is two logits, not a probability. Class `0` is spoof and class `1`
is bona-fide. `raw_score` is the native class-1 logit. `spoof_score` is
`logit[0] - logit[1]`, therefore higher means more spoof evidence. The baseline
observations are consistent with this code convention even if they are
counterintuitive for the two demonstration files: an unfamiliar target domain
can be systematically mis-ranked by a pretrained model.

## Baseline records

Each inference creates a timestamped JSON under `output/baseline/` and appends
to `baseline_results.csv`. `model_metadata.json` captures the exact checkpoint,
its hash, input specification, libraries, and scoring convention. These files
are the permanent pre-fine-tuning comparison record; future fine-tuning must
write to `output/finetuned/` instead.

## Dataset manifests and leakage control

Build the supplied target-domain manifest:

```bash
python build_dataset_manifest.py
```

It discovers 6,656 Svarah bona-fide rows and 13,306 Orpheus synthetic rows.
It creates frozen `train` (70%), `validation` (10%), `calibration` (10%), and
`test` (10%) splits. No recording may cross splits. Svarah has no local speaker
ID, so a demographic/location proxy is used; Orpheus exposes only two source
IDs, so its rows are isolated by recording but source-independent evaluation is
not established. The generated report records these limits.

For your own data, provide a CSV with `audio_path,label` where bona-fide is `0`
and spoof is `1`. Include `group_id` (speaker, recording session, synthesis
system, etc.) to prevent leakage and optionally a fixed `split` column:

```text
audio_path,label,group_id,split
real/a.wav,0,speaker_01,train
spoof/b.wav,1,tts_system_x,calibration
```

Run `python build_dataset_manifest.py --input-manifest my_audio.csv --output output/datasets/my_manifest.csv`.

## Fine-tuning

The default strategy is conservative full-model fine-tuning: AdamW at `1e-5`,
weight decay `1e-4`, inverse-frequency weighted cross entropy, validation-loss
early stopping, gradient clipping, and no augmentation by default. This keeps
the learning signal stable while the target-domain data first establishes
whether fine-tuning is justified. Optional gain/noise augmentation applies only
to training; validation, calibration, and test audio remain unaugmented.

```bash
python train.py --manifest output/datasets/target_domain_manifest.csv --device cpu
```

The calibration and test splits are explicitly loaded but never used to fit or
select the model. Checkpoints and run metadata go under `output/finetuned/`.
Resume with `--resume output/finetuned/checkpoints/last_model.pth`.

For a staged experiment, set `trainable_parameter_prefixes` in a copied
training JSON to `["out_layer"]` for a head-only stage, then start a new
full-model stage with `--pretrained-checkpoint` pointing at that stage's best
checkpoint and the default empty prefix list. This preserves reproducibility
without changing the pretrained original.

## Evaluation and calibration

Evaluate each model separately on calibration and test data. Do not calibrate a
pretrained score model and reuse it after fine-tuning.

```bash
# Pretrained held-out calibration scores
python evaluate.py --manifest output/datasets/target_domain_manifest.csv --split calibration --checkpoint ../models/aasist/AASIST.pth --output-dir output/baseline/evaluation

# Fit P(spoof | window) only on calibration scores
python calibrate.py --scores output/baseline/evaluation/calibration_predictions.csv --checkpoint ../models/aasist/AASIST.pth --output-dir output/baseline/calibration --model-id AASIST-pretrained

# Pretrained held-out test evaluation using the matching calibration model
python evaluate.py --manifest output/datasets/target_domain_manifest.csv --split test --checkpoint ../models/aasist/AASIST.pth --calibrator output/baseline/calibration/calibration_model.joblib --output-dir output/baseline/evaluation

# Fine-tuned calibration and test evaluation (after train.py)
python evaluate.py --manifest output/datasets/target_domain_manifest.csv --split calibration --checkpoint output/finetuned/checkpoints/best_model.pth --output-dir output/finetuned/evaluation
python calibrate.py --scores output/finetuned/evaluation/calibration_predictions.csv --checkpoint output/finetuned/checkpoints/best_model.pth --output-dir output/finetuned/calibration --model-id AASIST-finetuned
python evaluate.py --manifest output/datasets/target_domain_manifest.csv --split test --checkpoint output/finetuned/checkpoints/best_model.pth --calibrator output/finetuned/calibration/calibration_model.joblib --output-dir output/finetuned/evaluation
```

Platt scaling is used because it is a transparent, data-efficient binary score
calibrator. It writes the calibration model, Brier score, reliability curve,
score distributions, threshold, ROC-AUC, EER, accuracy, precision, recall, F1,
and confusion matrix. It refuses to fit fewer than 20 samples of either class.
Fit diagnostics are not final performance estimates: the untouched test split
is the final report.

Compare same-manifest held-out evaluations with:

```bash
python compare.py --pretrained output/baseline/evaluation/test_metrics.json --finetuned output/finetuned/evaluation/test_metrics.json --output output/finetuned/pretrained_vs_finetuned.json
```

For a short smoke test only, add `--max-samples-per-class 20` to `evaluate.py`.
Never use a bounded diagnostic file to fit the production calibration model or
claim final metrics.

## Tests

Run `python -m unittest discover -s tests -v`. The model test loads the
pretrained checkpoint on CPU, runs inference, and confirms its hash is unchanged.
