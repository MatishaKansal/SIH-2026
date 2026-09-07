import numpy as np

class FeatureFusionClassifier:
    """
    Feature Fusion Module.
    Combines outputs from the 3 parallel branches:
    1. AASIST score / feature vector (160 dims or 1 dim score)
    2. Wav2Vec2 XLS-R 300M speech embedding (1024 dims)
    3. Handcrafted acoustic spectral/prosodic feature vector (75 dims)

    Outputs window-level AI probability P(AI | window).
    """
    def __init__(self, input_dim: int = 1259, seed: int = 42):
        self.input_dim = input_dim
        # Prototype projection weights and bias for fusion scoring
        np.random.seed(seed)
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0

        # Calibration scaling factors
        self.aasist_weight = 2.0
        self.xlsr_weight = 0.5
        self.acoustic_weight = 0.5

    def fuse_features(
        self,
        aasist_data: dict,
        xlsr_embedding: np.ndarray,
        acoustic_vector: np.ndarray
    ) -> np.ndarray:
        """
        Concatenates features from all 3 branches into a single fused vector.
        """
        if "feature_vector" in aasist_data:
            aasist_feat = aasist_data["feature_vector"]
        else:
            aasist_feat = np.array([aasist_data.get("raw_score", 0.0)], dtype=np.float32)

        fused_vector = np.concatenate([
            aasist_feat.flatten(),
            xlsr_embedding.flatten(),
            acoustic_vector.flatten()
        ], axis=0)

        return fused_vector

    def predict_p_ai(
        self,
        aasist_data: dict,
        xlsr_embedding: np.ndarray,
        acoustic_vector: np.ndarray
    ) -> float:
        """
        Computes window-level AI probability P(AI | window).
        """
        fused_vector = self.fuse_features(aasist_data, xlsr_embedding, acoustic_vector)
        
        # Prototype fusion logic combining AASIST spoof probability and fused features
        spoof_prob_aasist = aasist_data.get("spoof_prob", 0.5)
        raw_score_aasist = aasist_data.get("raw_score", 0.0)
        
        # AASIST raw score is higher for bona-fide, lower for spoof
        # Convert raw score to AI/spoof score component
        aasist_ai_evidence = 1.0 / (1.0 + np.exp(raw_score_aasist))

        # Acoustic variance / zcr contribution
        zcr_mean = acoustic_vector[12] if len(acoustic_vector) > 12 else 0.0
        acoustic_evidence = 1.0 / (1.0 + np.exp(-zcr_mean))

        # Linear projection score over fused vector
        if len(fused_vector) == len(self.weights):
            linear_score = np.dot(fused_vector, self.weights) + self.bias
        else:
            linear_score = 0.0

        # Calibrated fused logit
        fused_logit = (
            self.aasist_weight * (aasist_ai_evidence - 0.5) +
            self.acoustic_weight * (acoustic_evidence - 0.5) +
            linear_score
        )

        p_ai = float(1.0 / (1.0 + np.exp(-fused_logit)))
        # Clip to avoid exact 0.0 or 1.0 for numerical stability in SPRT
        return float(np.clip(p_ai, 0.001, 0.999))
