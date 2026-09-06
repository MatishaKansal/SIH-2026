"""Audio buffering and windowing service"""

import numpy as np
from collections import deque
from app.config import settings


class AudioBuffer:
    """
    Manages audio buffering and sliding window extraction
    Converts continuous audio stream to fixed-size inference windows
    """
    
    def __init__(self, window_duration: float = 2.0, hop_size: float = 0.5):
        """
        Args:
            window_duration: Window size in seconds (e.g., 2.0s)
            hop_size: Hop size in seconds (e.g., 0.5s for 75% overlap)
        """
        self.window_duration = window_duration
        self.hop_size = hop_size
        
        # Convert to samples (16 kHz)
        self.window_samples = int(window_duration * settings.SAMPLE_RATE)
        self.hop_samples = int(hop_size * settings.SAMPLE_RATE)
        
        # Buffer for accumulating audio
        self.buffer = deque(maxlen=self.window_samples * 2)
        self.total_samples_received = 0
        self.last_window_end = 0
    
    def add_audio_chunk(self, audio_bytes: bytes) -> list:
        """
        Add audio chunk to buffer and extract complete windows
        
        Args:
            audio_bytes: Raw PCM audio bytes (16-bit, mono, 16 kHz)
            
        Returns:
            List of complete windows (as bytes) ready for inference
        """
        # Convert bytes to numpy array (16-bit PCM)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to circular buffer
        for sample in audio_data:
            self.buffer.append(sample)
        
        self.total_samples_received += len(audio_data)
        
        # Extract complete windows
        windows = []
        while (self.total_samples_received - self.last_window_end) >= self.hop_samples:
            if len(self.buffer) >= self.window_samples:
                # Extract window
                window = np.array(list(self.buffer)[-self.window_samples:], dtype=np.float32)
                window_bytes = (window * 32768).astype(np.int16).tobytes()
                windows.append(window_bytes)
                
                self.last_window_end += self.hop_samples
        
        return windows
