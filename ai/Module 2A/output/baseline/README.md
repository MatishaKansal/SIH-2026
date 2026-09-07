# Immutable baseline record

**BASELINE = PRETRAINED AASIST BEFORE FINE-TUNING.**

Every inference appends a row to `baseline_results.csv` and writes an individual
timestamped JSON result. `model_metadata.json` records the checkpoint SHA-256,
software versions, input format, and score direction. Part 2 must write only to
`output/finetuned/`; it must not revise these baseline records.
