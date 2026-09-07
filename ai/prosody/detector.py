import json,time
from pathlib import Path
import joblib,numpy as np
from .feature_extractor import ProsodyFeatureExtractor
from .preprocessing import FeaturePreprocessor
from .calibration import PlattCalibrator

class ProsodySpoofDetector:
    def __init__(self, model_dir='output'):
        base=Path(model_dir)
        if not (base/'models').exists(): base=Path(__file__).resolve().parents[1]/model_dir
        self.model_dir=base/'models'; self.model=joblib.load(self.model_dir/'best_boosting_model.joblib'); self.preprocessor=FeaturePreprocessor.load(self.model_dir)
        self.calibrator=PlattCalibrator.load(base/'calibration'/'calibration_model.joblib'); self.threshold=json.loads((base/'calibration'/'thresholds.json').read_text())['threshold']
        self.family=json.loads((self.model_dir/'model_metadata.json').read_text()).get('model_family','unknown'); self.extractor=ProsodyFeatureExtractor()
    def predict(self,waveform,sample_rate=16000):
        start=time.perf_counter(); raw=self.extractor.extract(waveform,sample_rate); active=self.preprocessor.transform(raw.reshape(1,-1)); score=float(self.model.predict_proba(active)[0,1]); prob=float(self.calibrator.predict_proba([score])[0]); cues=self._cues(raw)
        return {'model_family':self.family,'raw_score':score,'calibrated_spoof_prob':prob,'decision':'spoof' if prob>=self.threshold else 'bonafide','feature_vector':raw,'top_forensic_cues':cues,'latency_ms':(time.perf_counter()-start)*1000}
    def _cues(self, raw):
        names = self.extractor.feature_names
        explanations = {
            'local_jitter': lambda v: f'Vocal jitter is {"abnormally low" if v < 0.005 else "elevated"} ({v:.4f}) -- {"synthetic voices lack natural pitch wobble" if v < 0.005 else "possible vocoder artifact"}',
            'local_shimmer': lambda v: f'Vocal shimmer is {"abnormally low" if v < 0.02 else "elevated"} ({v:.4f}) -- {"too smooth for natural vocal folds" if v < 0.02 else "amplitude perturbation detected"}',
            'hnr_db': lambda v: f'Harmonics-to-noise ratio is {v:.1f} dB -- {"unusually clean, typical of neural vocoders" if v > 25 else "within human range" if v > 5 else "very noisy signal"}',
            'f0_std_hz': lambda v: f'Pitch variation is {"very low" if v < 5 else "normal"} ({v:.1f} Hz std) -- {"flat intonation suggests TTS" if v < 5 else "natural pitch dynamics"}',
            'f0_velocity_hz': lambda v: f'Pitch velocity is {"rigid" if v < 2 else "natural"} ({v:.2f} Hz/frame) -- {"monotone delivery typical of synthesis" if v < 2 else "dynamic human prosody"}',
            'pause_variance_s2': lambda v: f'Pause timing variance is {"near zero" if v < 0.001 else "natural"} ({v:.4f} s2) -- {"mechanically regular pauses suggest generation" if v < 0.001 else "organic breathing pattern"}',
            'spectral_flatness_mean': lambda v: f'Spectral flatness is {"high" if v > 0.1 else "low"} ({v:.4f}) -- {"noise-like texture from vocoder" if v > 0.1 else "tonal, harmonic-rich signal"}',
            'energy_flux': lambda v: f'Energy flux is {"very smooth" if v < 0.002 else "dynamic"} ({v:.4f}) -- {"unnaturally steady loudness" if v < 0.002 else "natural syllabic energy variation"}',
            'voiced_frame_ratio': lambda v: f'Voiced frame ratio is {v:.1%} -- {"unusually continuous voicing, no natural breathing gaps" if v > 0.95 else "normal voicing pattern"}',
        }
        cues = []
        for i, (name, val) in enumerate(zip(names, raw)):
            if name in explanations and np.isfinite(val):
                cues.append((abs(float(val)), explanations[name](float(val))))
        cues.sort(key=lambda x: x[0], reverse=True)
        result = [text for _, text in cues[:3]]
        if len(result) < 3:
            for i, (name, val) in enumerate(zip(names, raw)):
                if name not in explanations and np.isfinite(val) and abs(val) > 0:
                    result.append(f'{name} = {float(val):.4f}')
                if len(result) >= 3:
                    break
        return result
