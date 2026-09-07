import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from aasist.aasist_model import Model

class AASISTDetector:
    """
    AASIST Anti-Spoofing Detector loading official pretrained AASIST.pth checkpoint.
    Note: Pretrained on ASVspoof2019 LA dataset.
    """
    def __init__(self, model_path="models/aasist/AASIST.pth", config_path="src/aasist/AASIST.conf", device="cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.config_path = config_path
        self.target_length = 64600  # 4 seconds at 16kHz (64,600 samples)

        with open(config_path, "r") as f:
            config = json.load(f)

        self.model_config = config["model_config"]
        self.model = Model(self.model_config).to(self.device)

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt)
            print(f"Loaded AASIST checkpoint from: {model_path}")
        else:
            raise FileNotFoundError(f"AASIST checkpoint not found at: {model_path}")

        self.model.eval()

    def prepare_input(self, audio: np.ndarray) -> torch.Tensor:
        """
        Pads or truncates 16kHz audio input to target length (64,600 samples).
        """
        if isinstance(audio, np.ndarray):
            tensor_audio = torch.from_numpy(audio).float()
        else:
            tensor_audio = audio.float()

        if tensor_audio.ndim > 1:
            tensor_audio = tensor_audio.squeeze()

        length = tensor_audio.shape[0]
        if length < self.target_length:
            # Repeat audio to fill target_length
            num_repeats = (self.target_length // length) + 1
            tensor_audio = tensor_audio.repeat(num_repeats)[:self.target_length]
        elif length > self.target_length:
            tensor_audio = tensor_audio[:self.target_length]

        return tensor_audio.unsqueeze(0)  # Shape: (1, 64600)

    def extract(self, audio: np.ndarray):
        """
        Evaluates input audio window and returns dict with:
        - raw_score: bona-fide minus spoof logit difference
        - bona_fide_prob: softmax probability for bona-fide
        - spoof_prob: softmax probability for spoof
        - feature_vector: 160-dim last hidden representation
        """
        input_tensor = self.prepare_input(audio).to(self.device)

        with torch.no_grad():
            last_hidden, output = self.model(input_tensor)
            # output shape: (1, 2) where index 0 is spoof, index 1 is bona-fide
            logits = output.squeeze(0)
            probs = F.softmax(logits, dim=-1)
            
            spoof_prob = float(probs[0].item())
            bona_fide_prob = float(probs[1].item())
            raw_score = float((logits[1] - logits[0]).item())  # Higher score = more bona-fide

        return {
            "raw_score": raw_score,
            "bona_fide_prob": bona_fide_prob,
            "spoof_prob": spoof_prob,
            "feature_vector": last_hidden.squeeze(0).cpu().numpy()  # 160-dim vector
        }
