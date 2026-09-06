"""Database models for call sessions and audit events"""

from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, Integer
from app.database import Base


class CallSession(Base):
    """Call session tracking"""
    __tablename__ = "call_sessions"
    
    call_id = Column(String(255), primary_key=True)
    claimed_identity = Column(String(255))
    source_phone = Column(String(20), nullable=True)
    transaction_amount = Column(Float, nullable=True)
    status = Column(String(50))  # IN_PROGRESS, COMPLETED, BLOCKED
    max_risk_score = Column(Float, default=0.0)
    decision = Column(String(50), nullable=True)  # SAFE, INVESTIGATE, BLOCKED
    windows_processed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)


class AuditEvent(Base):
    """Immutable audit trail with cryptographic hashes"""
    __tablename__ = "audit_events"
    
    event_id = Column(String(255), primary_key=True)
    call_id = Column(String(255))
    event_type = Column(String(100))  # WINDOW_PROCESSED, FRAUD_ALERT, TRANSACTION_VERIFICATION
    action_taken = Column(String(100))  # ALLOWED, BLOCKED, ESCALATED
    risk_score = Column(Float)
    event_hash = Column(String(256))  # SHA-256 hash of event
    details = Column(Text)  # JSON details
    timestamp = Column(DateTime, default=datetime.utcnow)


class AnalystUser(Base):
    """Registered analyst account credentials and profile details."""
    __tablename__ = "analyst_users"

    user_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
