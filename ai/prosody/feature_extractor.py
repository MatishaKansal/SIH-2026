"""Robust, interpretable features for a 16 kHz speech window."""
import numpy as np
import librosa
from scipy.stats import skew, kurtosis

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:  # permits spectral-only development environments
    parselmouth = None


class ProsodyFeatureExtractor:
    FEATURE_NAMES = [
        "f0_mean_hz", "f0_median_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_range_hz", "f0_iqr_hz", "voiced_frame_ratio", "unvoiced_frame_ratio", "f0_slope_hz_per_frame", "f0_velocity_hz", "f0_acceleration_hz",
        "rms_mean", "rms_std", "rms_min", "rms_max", "rms_range", "rms_skewness", "rms_kurtosis", "voiced_to_unvoiced_energy_ratio", "energy_flux",
        "voiced_duration_s", "unvoiced_duration_s", "voiced_burst_count", "pause_count", "mean_voiced_segment_s", "mean_pause_s", "pause_variance_s2", "speaking_activity_ratio",
        "local_jitter", "local_shimmer", "hnr_db", "spectral_centroid_mean", "spectral_centroid_std", "spectral_bandwidth_mean", "spectral_bandwidth_std", "spectral_rolloff_mean", "spectral_rolloff_std", "spectral_flatness_mean", "spectral_flatness_std", "zcr_mean", "zcr_std",
    ] + [f"spectral_contrast_{stat}_band_{band}" for stat in ("mean", "std") for band in range(7)]

    def __init__(self, sample_rate=16000, hop_length=160, frame_length=400):
        self.sample_rate, self.hop_length, self.frame_length = sample_rate, hop_length, frame_length

    @property
    def feature_names(self): return list(self.FEATURE_NAMES)

    def _normalise(self, waveform, sample_rate):
        y = np.asarray(waveform, dtype=np.float32).squeeze()
        if y.ndim != 1: raise ValueError("waveform must be mono")
        if sample_rate != self.sample_rate:
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=self.sample_rate)
        y = np.nan_to_num(y, nan=0., posinf=0., neginf=0.)
        if y.size > 64000:
            start = max(0, (y.size - 64000) // 2)
            y = y[start : start + 64000]
        return np.pad(y, (0, max(0, 512 - y.size)))

    @staticmethod
    def _segments(mask):
        padded = np.r_[False, mask, False].astype(int)
        starts, ends = np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]
        return list(zip(starts, ends))

    def extract(self, waveform, sample_rate=16000):
        y = self._normalise(waveform, sample_rate)
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        try:
            # YIN is materially faster than probabilistic YIN in streaming CPU
            # inference. RMS gating supplies the conservative voicing decision.
            f0 = np.asarray(librosa.yin(y, fmin=75, fmax=500, sr=self.sample_rate,
                                        frame_length=1024, hop_length=self.hop_length))
            rms_for_f0 = librosa.feature.rms(y=y, frame_length=1024, hop_length=self.hop_length)[0]
            gate = max(1e-5, float(np.max(rms_for_f0)) * .08)
            voiced = (rms_for_f0 > gate) & np.isfinite(f0)
        except Exception:
            f0 = np.full(len(rms), np.nan); voiced = np.zeros(len(rms), dtype=bool)
        vals = f0[voiced]
        nan = float("nan")
        if vals.size:
            slope = np.polyfit(np.arange(vals.size), vals, 1)[0] if vals.size > 1 else 0.
            velocity = np.mean(np.abs(np.diff(vals))) if vals.size > 1 else 0.
            accel = np.mean(np.abs(np.diff(vals, n=2))) if vals.size > 2 else 0.
            pitch = [np.mean(vals), np.median(vals), np.std(vals), np.min(vals), np.max(vals), np.ptp(vals), np.subtract(*np.percentile(vals,[75,25])), voiced.mean(), 1-voiced.mean(), slope, velocity, accel]
        else: pitch = [nan]*7 + [0., 1., nan, nan, nan]
        safe_skew = skew(rms) if rms.size > 2 and np.std(rms) else 0.
        safe_kurt = kurtosis(rms) if rms.size > 3 and np.std(rms) else 0.
        voiced_energy = np.mean(rms[voiced]) if np.any(voiced) else np.nan
        unvoiced_energy = np.mean(rms[~voiced]) if np.any(~voiced) else np.nan
        energy_ratio = voiced_energy / unvoiced_energy if np.isfinite(unvoiced_energy) and unvoiced_energy > 1e-12 else np.nan
        energy = [np.mean(rms), np.std(rms), np.min(rms), np.max(rms), np.ptp(rms), safe_skew, safe_kurt, energy_ratio, np.mean(np.abs(np.diff(rms))) if rms.size > 1 else 0.]
        frame_s = self.hop_length / self.sample_rate
        vs, us = self._segments(voiced), self._segments(~voiced)
        vd, ud = [(b-a)*frame_s for a,b in vs], [(b-a)*frame_s for a,b in us]
        pauses = [x for x in ud if x > .05]
        rhythm = [sum(vd), sum(ud), len(vs), len(pauses), np.mean(vd) if vd else 0., np.mean(pauses) if pauses else 0., np.var(pauses) if pauses else 0., voiced.mean()]
        quality = self._quality(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate, hop_length=self.hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate, hop_length=self.hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate, roll_percent=.85, hop_length=self.hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate, hop_length=self.hop_length)
        spectral = [np.mean(centroid),np.std(centroid),np.mean(bandwidth),np.std(bandwidth),np.mean(rolloff),np.std(rolloff),np.mean(flatness),np.std(flatness),np.mean(zcr),np.std(zcr)]
        return np.asarray(pitch + energy + rhythm + quality + spectral + list(np.mean(contrast,axis=1)) + list(np.std(contrast,axis=1)), dtype=np.float32)

    def _quality(self, y):
        jitter = shimmer = hnr = float("nan")
        if parselmouth is not None and np.max(np.abs(y)) > 1e-7:
            try:
                snd = parselmouth.Sound(y, sampling_frequency=self.sample_rate)
                pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
                jitter = call(pp, "Get jitter (local)", 0, 0, .0001, .02, 1.3)
                shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, .0001, .02, 1.3, 1.6)
                h = snd.to_harmonicity_cc(); values = h.values[h.values > -200]
                hnr = np.mean(values) if values.size else float("nan")
            except Exception: pass
        return [jitter, shimmer, hnr]

    extract_features = extract
