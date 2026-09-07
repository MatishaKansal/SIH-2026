import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

class AcousticFeatureExtractor:
    """
    Handcrafted Spectral + Prosodic Feature Extractor.
    Extracts fixed-length 1D numeric feature vector from a 16 kHz audio window.
    """
    def __init__(self, sample_rate=16000, n_mfcc=20, n_mels=128):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts spectral and prosodic features from 16kHz audio window.
        Returns a 1D numpy array of fixed length.
        """
        if isinstance(audio, np.ndarray):
            y = audio.astype(np.float64)
        else:
            y = audio.cpu().numpy().astype(np.float64)

        if y.ndim > 1:
            y = y.squeeze()

        # Ensure minimum length for STFT operations (at least 512 samples)
        if len(y) < 512:
            y = np.pad(y, (0, 512 - len(y)))

        feature_list = []

        # ==================== 1. SPECTRAL FEATURES ==================== #
        # A. MFCC (20 coefficients: mean & std -> 40 dims)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        feature_list.extend(np.mean(mfcc, axis=1))
        feature_list.extend(np.std(mfcc, axis=1))

        # B. Mel Spectrogram Statistics (mean & std across mels -> 2 dims)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        feature_list.append(np.mean(mel_db))
        feature_list.append(np.std(mel_db))

        # C. Spectral Centroid (mean & std -> 2 dims)
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
        feature_list.append(np.mean(cent))
        feature_list.append(np.std(cent))

        # D. Spectral Bandwidth (mean & std -> 2 dims)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)
        feature_list.append(np.mean(spec_bw))
        feature_list.append(np.std(spec_bw))

        # E. Spectral Rolloff (mean & std -> 2 dims)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)
        feature_list.append(np.mean(rolloff))
        feature_list.append(np.std(rolloff))

        # F. Spectral Flux (onset strength mean & std -> 2 dims)
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        feature_list.append(np.mean(onset_env))
        feature_list.append(np.std(onset_env))

        # G. Zero-Crossing Rate (mean & std -> 2 dims)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        feature_list.append(np.mean(zcr))
        feature_list.append(np.std(zcr))

        # H. Spectral Contrast (7 bands: mean & std -> 14 dims)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate)
        feature_list.extend(np.mean(contrast, axis=1))
        feature_list.extend(np.std(contrast, axis=1))

        # ==================== 2. PROSODIC FEATURES ==================== #
        # Energy / RMS statistics (mean & std -> 2 dims)
        rms = librosa.feature.rms(y=y)
        feature_list.append(np.mean(rms))
        feature_list.append(np.std(rms))

        # F0 / Pitch & Voice Perturbation (Parselmouth / Praat)
        f0_mean, f0_std, f0_min, f0_max = 0.0, 0.0, 0.0, 0.0
        voicing_ratio = 0.0
        local_jitter = 0.0
        local_shimmer = 0.0

        try:
            snd = parselmouth.Sound(y, sampling_frequency=self.sample_rate)
            pitch = snd.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            voiced_values = pitch_values[pitch_values > 0]

            if len(voiced_values) > 0:
                f0_mean = float(np.mean(voiced_values))
                f0_std = float(np.std(voiced_values))
                f0_min = float(np.min(voiced_values))
                f0_max = float(np.max(voiced_values))
                voicing_ratio = float(len(voiced_values) / len(pitch_values))

                # Jitter & Shimmer via Praat PointProcess
                point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
                local_jitter = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
                local_shimmer = float(call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

                if np.isnan(local_jitter): local_jitter = 0.0
                if np.isnan(local_shimmer): local_shimmer = 0.0
        except Exception:
            pass

        feature_list.extend([f0_mean, f0_std, f0_min, f0_max, voicing_ratio, local_jitter, local_shimmer])

        return np.array(feature_list, dtype=np.float32)
