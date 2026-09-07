from pathlib import Path
import json, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration import run_manifest

def test_module1_manifest_metadata_is_preserved(tmp_path):
    chunk=tmp_path/"speech_000.wav"; sf.write(chunk,np.zeros(16000,dtype=np.float32),16000)
    manifest=tmp_path/"manifest.json"; manifest.write_text(json.dumps({"source":"recording.wav","source_id":"recording-1","segments":[{"id":7,"start":3.5,"end":4.382,"duration":0.882,"file":chunk.name,"padded":True,"region_index":2}]}))
    result=run_manifest(manifest)[0]
    assert result["chunk_id"]==7 and result["source_id"]=="recording-1" and result["start"]==3.5 and result["padded"] is True and result["region_index"]==2

def test_module1_manifest_processes_all_chunks(tmp_path):
    chunks = []

    for i in range(3):
        chunk = tmp_path / f"speech_{i:03d}.wav"
        sf.write(
            chunk,
            np.zeros(16000, dtype=np.float32),
            16000,
        )
        chunks.append(chunk)

    manifest = tmp_path / "manifest.json"

    manifest.write_text(
        json.dumps(
            {
                "source": "recording.wav",
                "source_id": "recording-1",
                "sample_rate": 16000,
                "channels": 1,
                "segments": [
                    {
                        "id": i,
                        "start": i * 0.5,
                        "end": i * 0.5 + 1.0,
                        "duration": 1.0,
                        "file": chunk.name,
                        "padded": False,
                        "region_index": 0,
                    }
                    for i, chunk in enumerate(chunks)
                ],
            }
        )
    )

    results = run_manifest(manifest)

    assert len(results) == 3
    assert [r["chunk_id"] for r in results] == [0, 1, 2]
    assert all(r["source_id"] == "recording-1" for r in results)