import numpy as np
from typing import Dict, Any

class WaldSPRT:
    """
    Wald Sequential Probability Ratio Test (SPRT) for real-time window sequence evidence.

    Hypotheses:
    - H0: Audio is Human/Bona-fide (p_0 = 0.1)
    - H1: Audio is AI-Generated/Cloned (p_1 = 0.9)

    Decision States:
    - SAFE: LLR <= log(B)  -> High confidence Human
    - HIGH RISK: LLR >= log(A) -> High confidence AI Voice Clone
    - INVESTIGATE: log(B) < LLR < log(A) -> Further evidence required
    """
    def __init__(self, alpha: float = 0.01, beta: float = 0.01, p0: float = 0.1, p1: float = 0.9):
        self.alpha = alpha
        self.beta = beta
        self.p0 = p0
        self.p1 = p1

        # Decision boundaries
        self.A = (1.0 - beta) / alpha
        self.B = beta / (1.0 - alpha)

        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)

        self.reset()

    def reset(self):
        """Resets accumulated evidence for a new session."""
        self.cumulative_llr = 0.0
        self.window_count = 0
        self.history = []

    def update(self, p_ai_window: float) -> Dict[str, Any]:
        """
        Updates SPRT with a new window-level P(AI | window) probability.
        Returns updated state dict.
        """
        p = np.clip(p_ai_window, 1e-4, 1.0 - 1e-4)
        
        # Log likelihood ratio under Bernoulli evidence (p1 vs p0)
        llr_step = p * np.log(self.p1 / self.p0) + (1.0 - p) * np.log((1.0 - self.p1) / (1.0 - self.p0))

        self.cumulative_llr += llr_step
        self.window_count += 1

        if self.cumulative_llr <= self.log_B:
            decision = "SAFE"
        elif self.cumulative_llr >= self.log_A:
            decision = "HIGH RISK"
        else:
            decision = "INVESTIGATE"

        result = {
            "window_idx": self.window_count,
            "p_ai_window": float(p),
            "step_llr": float(llr_step),
            "cumulative_llr": float(self.cumulative_llr),
            "log_upper_bound_A": float(self.log_A),
            "log_lower_bound_B": float(self.log_B),
            "decision": decision
        }

        self.history.append(result)
        return result
