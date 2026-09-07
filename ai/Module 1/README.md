# Module 1 — Silero VAD

This standalone module normalizes audio to **16 kHz mono float32**, uses the
official [Silero VAD](https://github.com/snakers4/silero-vad) Python package to
identify speech regions, then emits configurable overlapping speech windows.
It makes no judgement about whether a voice is real, synthetic, or otherwise.

## Run

1. Install the dependencies from this directory: `pip install -r requirements.txt`.
2. Put an audio file in `input_audio/`.
3. Change `AUDIO_SOURCE` near the top of `input.py`.
4. Run `python input.py` from `Module 1/`.

`input.py` writes optional debug chunks into `output/`.  Each WAV is 16 kHz,
mono PCM; `output/manifest.json` describes its source timestamps.  The primary
API remains in-memory: `SileroVAD.process_file()` returns
`(normalized_waveform, regions, chunks)`, where every region/chunk has `start`,
`end`, and `audio` (a NumPy float32 waveform).

## Adjusting windows

Edit `WINDOW_SIZE_SECONDS` and `HOP_SIZE_SECONDS` in `config.py`. Defaults are
1.0 second windows with a 0.5 second hop.  Speech detection configuration is
kept separately in the same file.

## Model artifacts

`models/silero_vad/` is reserved for official Silero artifacts when an offline
deployment needs them. The default loader is `silero_vad.load_silero_vad`, the
official package interface, and runs on CPU. This first version leaves
microphone capture as an adapter point; live float32 audio can be supplied to
`detect_regions()` after normalization.

## Tests

Run `python -m unittest discover -s tests -v` from `Module 1/`.
