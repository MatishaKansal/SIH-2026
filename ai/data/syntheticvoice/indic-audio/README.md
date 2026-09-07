---
license: other
license_name: synthetic-tts-terms
language:
  - hi
  - en
task_categories:
  - text-to-speech
tags:
  - tts
  - hindi
  - hinglish
  - indian-english
  - synthetic
  - multi-speaker
  - voice-cloning
pretty_name: Indic Audio
size_categories:
  - 1K<n<10K
---

# indic-audio

A **multi-speaker synthetic speech dataset** for Hindi, Indian English, and Hinglish (Hindi in Latin script): **5,160 clips, 7.1 hours, 15 voices**. Built to train [goonj-1-82M](https://huggingface.co/BH-Builds/goonj-1-82M), an edge-sized Indian-language TTS model.

![Dataset pipeline](images/dataset-pipeline.png)

## Summary

| | |
|---|---|
| Clips | 5,160 |
| Total audio | 7.13 h (avg 5.0 s/clip) |
| Voices | 15 (14 named personas + 1 Hindi "language bed") |
| Languages | Hindi (Devanagari), Indian English, Hinglish (romanized) |
| Format | 44.1 kHz; WAV (bed_hindi) and MP3 (personas) |
| Transcripts | `metadata.jsonl` |

### Breakdown by voice group

| Group | Voices | Clips | Hours | Script |
|---|---|---|---|---|
| `bed_hindi` | 1 | 1,900 | 3.22 | Devanagari Hindi (long-form prose) |
| `hi_*` | atul, meera, ravi, shivani | 960 (240 each) | 1.55 | Devanagari Hindi |
| `en_*` | aman, ananya, arjun, dev, divya, kabir, nisha, priya, sameer, tara | 1,600 (160 each) | 1.75 | Indian English |
| `hing_*` | all 14 personas | 700 (50 each) | 0.60 | Hinglish (Latin script) |

## Files

```
audio/<voice>/<id>.{wav,mp3}   # 5,160 clips, organised by voice
metadata.jsonl                 # one row per clip: {"audio", "text", "voice"}
ref_texts.json                 # transcript of each persona's reference recording
scripts/<voice>.txt            # the sentences synthesized for each voice
scripts/.pool_{hindi,english,hinglish}.json   # source sentence pools (~40k each)
```

`metadata.jsonl` row:

```json
{"audio": "audio/hi_meera/0000.mp3", "text": "अगली सुबह बाबा अजय से मिले, ...", "voice": "hi_meera"}
```

## How it was built

1. **Persona design (ElevenLabs).** 15 voices were designed on the ElevenLabs platform. Each got one reference recording of ~30–35 s, read from deliberately "hard" scripts containing numbers, percentages, currency, exclamations, and questions, so the reference captures prosody variety rather than flat narration. Transcripts of those references are in `ref_texts.json`.
2. **Bulk synthesis (Fish Audio S2 Pro).** Using each persona's reference recording for voice cloning, the sentence pools were synthesized into the 5,160 clips here. In later evaluations, [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) produced noticeably better reference-conditioned Hindi/Hinglish speech; future versions of this dataset will likely use it.
3. **Sentence pools.** Hindi prose (bed), conversational Hindi, everyday Indian English, and casual Hinglish sentences (~40k per pool), generated per persona so each voice covers all its language modes.

## Intended use

Training and evaluating TTS models for Indian languages, especially small/edge models where a modest, clean, multi-speaker corpus is sufficient. The companion model [goonj-1-82M](https://huggingface.co/BH-Builds/goonj-1-82M) was finetuned from Kokoro-82M on exactly this data (train/val split: 4,986 / 174).

## Known limitations

- **Fully synthetic.** All audio is TTS output, not human recordings. Artifacts, occasional mispronunciations, and the acoustic signature of the generating system carry into any model trained on it.
- **Hinglish is romanized casually** (no standard transliteration scheme), so grapheme-to-phoneme handling needs a lexicon or a transliteration model. A ~300-word lexicon that covers the frequent words is shipped with goonj-1-82M.
- Persona coverage is uneven: the four `hi_*` personas have Hindi + Hinglish data, the ten `en_*` personas have English + Hinglish; no persona has all three.
- No speaker demographics beyond persona names; voices are synthetic designs, not real people.

## Licensing

The audio was produced with ElevenLabs (persona references) and Fish Audio S2 Pro (bulk clips); use of the generated audio is subject to those services' terms for synthesized output. Transcripts, scripts, and metadata are released by the dataset author. Please review the generating services' terms before commercial use.

## Citation

```
@misc{indic-audio-2026,
  title  = {indic-audio: a synthetic multi-speaker Hindi / Indian English / Hinglish TTS corpus},
  author = {BH-Builds},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BH-Builds/indic-audio}
}
```
