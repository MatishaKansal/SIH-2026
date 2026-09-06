"""Audit logging service - Immutable event trail"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from app.models import AuditEvent


class AuditLogger:
    """
    Logs all security-relevant events to an immutable audit trail
    Each event is hashed for tamper detection
    """
    
    async def log_event(
        self,
        db: Session,
        call_id: str,
        event_type: str,
        action_taken: str,
        risk_score: float,
        event_hash: str,
        details: Optional[dict] = None
    ) -> str:
        """
        Log security event to audit trail
        
        Args:
            db: Database session
            call_id: Associated call identifier
            event_type: Type of event (WINDOW_PROCESSED, FRAUD_ALERT, etc.)
            action_taken: Action triggered (ALLOWED, BLOCKED, ESCALATED, etc.)
            risk_score: Risk score at time of event
            event_hash: SHA-256 hash for tamper detection
            details: JSON details of the event
        
        Returns:
            event_id: Unique event identifier
        """
        
        event_id = f"evt_{call_id}_{int(datetime.utcnow().timestamp() * 1000)}"
        
        audit_event = AuditEvent(
            event_id=event_id,
            call_id=call_id,
            event_type=event_type,
            action_taken=action_taken,
            risk_score=risk_score,
            event_hash=event_hash,
            details=str(details) if details else ""
        )
        
        db.add(audit_event)
        db.commit()
        
        return event_id
