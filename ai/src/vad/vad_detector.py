import os
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

class SileroVADDetector:
    """
    Silero VAD wrapper for speech activity detection, streaming evaluation,
    and audio window segmentation. Optimized for CPU inference.
    """
    def __init__(self, model_dir="models/silero_vad", use_onnx=True):
        self.model_dir = model_dir
        self.use_onnx = use_onnx
        self.sample_rate = 16000
        
        onnx_path = os.path.join(model_dir, "silero_vad.onnx")
        jit_path = os.path.join(model_dir, "silero_vad.jit")

        if use_onnx and os.path.exists(onnx_path):
            self.model = load_silero_vad(onnx=True)
            self.model_path = onnx_path
        elif os.path.exists(jit_path):
            self.model = torch.jit.load(jit_path)
            self.model.eval()
            self.model_path = jit_path
        else:
            # Fallback to loading pretrained Silero VAD
            self.model = load_silero_vad(onnx=use_onnx)
            self.model_path = "official_package"

    def get_speech_timestamps(self, audio: np.ndarray, sampling_rate: int = 16000, threshold: float = 0.5):
        """
        Returns speech timestamp intervals for input 16kHz audio waveform.
        """
        if isinstance(audio, np.ndarray):
            tensor_audio = torch.from_numpy(audio).float()
        else:
            tensor_audio = audio.float()

        if tensor_audio.ndim > 1:
            tensor_audio = tensor_audio.squeeze()

        timestamps = get_speech_timestamps(
            tensor_audio,
            self.model,
            sampling_rate=sampling_rate,
            threshold=threshold
        )
        return timestamps

    def get_speech_probability(self, chunk: np.ndarray, sampling_rate: int = 16000) -> float:
        """
        Computes speech probability for a single chunk of audio (e.g. 512 samples for 16kHz).
        """
        if isinstance(chunk, np.ndarray):
            tensor_chunk = torch.from_numpy(chunk).float()
        else:
            tensor_chunk = chunk.float()

        if tensor_chunk.ndim == 1:
            tensor_chunk = tensor_chunk.unsqueeze(0)

        with torch.no_grad():
            prob = self.model(tensor_chunk, sampling_rate).item()
        return float(prob)

    def segment_speech_windows(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        window_sec: float = 2.0,
        overlap_sec: float = 1.0,
        vad_threshold: float = 0.5
    ):
        """
        Segments audio into overlapping speech windows of duration `window_sec` seconds.
        Filters out non-speech segments using VAD.
        """
        window_size = int(window_sec * sampling_rate)
        step_size = int((window_sec - overlap_sec) * sampling_rate)
        
        if len(audio) < window_size:
            # Pad audio if shorter than window_size
            padded = np.pad(audio, (0, window_size - len(audio)))
            return [{"window": padded, "start_sample": 0, "end_sample": len(audio), "speech_prob": 1.0}]

        windows = []
        for start in range(0, len(audio) - window_size + 1, step_size):
            end = start + window_size
            chunk = audio[start:end]
            
            # Check VAD on chunk
            timestamps = self.get_speech_timestamps(chunk, sampling_rate=sampling_rate, threshold=vad_threshold)
            speech_ratio = sum(t["end"] - t["start"] for t in timestamps) / float(window_size) if timestamps else 0.0
            
            if speech_ratio >= 0.2 or len(windows) == 0:  # Include if speech detected or fallback first window
                windows.append({
                    "window": chunk,
                    "start_sample": start,
                    "end_sample": end,
                    "speech_ratio": speech_ratio
                })

        return windows
