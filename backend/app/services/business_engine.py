"""Business rule engine - Maps AI scores to fraud prevention actions"""

from app.config import settings


class BusinessEngine:
    """
    Translates raw AI probabilities into business decisions:
    - SAFE (< 30%): Allow transaction, no additional verification
    - INVESTIGATE (30-70%): Caution, consider additional verification
    - CRITICAL (> 70%): Block transaction, require step-up authentication
    """
    
    def evaluate_risk(self, ai_response: dict) -> dict:
        """
        Evaluate fraud risk based on AI model outputs
        
        Args:
            ai_response: Output from AI service containing:
                - cumulative_risk_score (0-1 or 0-100)
                - wald_decision (SAFE, INVESTIGATE, HIGH_RISK)
                - metrics (aasist, prosody, speaker_match scores)
        
        Returns:
            {
                "action": "ALLOW" | "INVESTIGATE" | "BLOCK_TRANSACTION",
                "risk_level": "SAFE" | "INVESTIGATE" | "CRITICAL",
                "reason": str,
                "recommended_step_up": "NONE" | "CALLBACK" | "OTP" | "MANAGER_OVERRIDE"
            }
        """
        
        # Extract risk score (normalize to 0-1 range)
        risk_score = ai_response.get("cumulative_risk_score", 0.0)
        if risk_score > 1.0:
            risk_score = risk_score / 100.0  # Normalize if in 0-100 range
        
        # Get component scores
        metrics = ai_response.get("metrics", {})
        aasist_score = metrics.get("aasist_spoof_score", 0.0)
        prosody_score = metrics.get("prosody_anomaly_score", 0.0)
        speaker_match = metrics.get("speaker_match_score", 1.0)
        
        # Determine risk level
        if risk_score < settings.RISK_THRESHOLD_INVESTIGATE:
            risk_level = "SAFE"
            action = "ALLOW"
            step_up = "NONE"
            reason = "Call authenticated. Synthetic voice probability low."
        
        elif risk_score < settings.RISK_THRESHOLD_HIGH_RISK:
            risk_level = "INVESTIGATE"
            
            # Determine if we should escalate or just monitor
            if aasist_score > 0.7 or prosody_score > 0.6:
                action = "INVESTIGATE"
                step_up = "CALLBACK"
                reason = "Moderate fraud indicators detected. Out-of-band verification recommended."
            else:
                action = "ALLOW"
                step_up = "NONE"
                reason = "Risk score elevated but within acceptable range. Monitor call."
        
        else:  # HIGH RISK
            risk_level = "CRITICAL"
            action = "BLOCK_TRANSACTION"
            step_up = "MANAGER_OVERRIDE"
            
            reasons = []
            if aasist_score > 0.85:
                reasons.append(f"High synthetic voice confidence ({aasist_score*100:.0f}%)")
            if prosody_score > 0.7:
                reasons.append(f"Unnatural prosody detected ({prosody_score*100:.0f}%)")
            if speaker_match < 0.7:
                reasons.append(f"Voice mismatch with enrolled profile ({speaker_match*100:.0f}%)")
            
            reason = " | ".join(reasons) if reasons else "High fraud risk detected"
        
        return {
            "action": action,
            "risk_level": risk_level,
            "reason": reason,
            "recommended_step_up": step_up,
            "scores": {
                "cumulative_risk": risk_score,
                "aasist_spoof": aasist_score,
                "prosody_anomaly": prosody_score,
                "speaker_match": speaker_match
            }
        }
