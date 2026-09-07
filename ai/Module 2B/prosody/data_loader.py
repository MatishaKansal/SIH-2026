import io
from pathlib import Path
import numpy as np, soundfile as sf

def resolve_dataset_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.exists(): return p
    parts = p.parts
    if "data" in parts:
        candidate = project_root.joinpath(*parts[parts.index("data"):])
        if candidate.exists(): return candidate
    raise FileNotFoundError(f"Could not resolve dataset path: {path_str}")

def normalise_audio(audio, sr, target_sr=16000):
    import librosa
    y=np.asarray(audio,dtype=np.float32)
    if y.ndim > 1: y=np.mean(y,axis=1)
    if sr != target_sr: y=librosa.resample(y,orig_sr=sr,target_sr=target_sr)
    peak=np.max(np.abs(y)) if y.size else 0
    return y / peak if peak > 1 else y

def load_manifest_row(row, project_root):
    def _get(key):
        if isinstance(row, dict):
            return row.get(key)
        return getattr(row, key, None)

    audio_path = _get("audio_path")
    if audio_path and str(audio_path).strip() and str(audio_path) != "nan":
        resolved_audio = resolve_dataset_path(str(audio_path), Path(project_root))
        y, sr = sf.read(str(resolved_audio), always_2d=False)
        return normalise_audio(y, sr)

    parquet_path = _get("parquet_path")
    if not parquet_path or str(parquet_path) == "nan":
        raise ValueError(f"Invalid row with no valid audio_path or parquet_path: {row}")

    path = resolve_dataset_path(str(parquet_path), Path(project_root))
    import pyarrow.parquet as pq
    audio_column = _get("audio_column")
    row_group = int(_get("row_group"))
    row_in_group = int(_get("row_in_group"))

    table = pq.ParquetFile(path).read_row_group(row_group, columns=[audio_column])
    item = table.column(audio_column)[row_in_group].as_py()
    y, sr = sf.read(io.BytesIO(item["bytes"]), always_2d=False)
    return normalise_audio(y, sr)
