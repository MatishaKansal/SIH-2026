"""Application configuration"""

from pathlib import Path

from pydantic_settings import BaseSettings


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_TITLE: str = "Enterprise Call Fraud Prevention API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./fraud_prevention.db"
    NEXT_PUBLIC_SUPABASE_URL: str = ""
    NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    
    # AI Service
    AI_SERVICE_URL: str = "http://localhost:8001"
    AI_SERVICE_TIMEOUT: int = 10
    
    # Audio Configuration
    SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_DURATION_MS: int = 500
    WINDOW_DURATION_SECONDS: float = 2.0
    HOP_SIZE_SECONDS: float = 0.5
    
    # Risk Thresholds
    RISK_THRESHOLD_INVESTIGATE: float = 0.30
    RISK_THRESHOLD_HIGH_RISK: float = 0.70
    
    # Feature Flags
    ENABLE_AUDIT_LOGGING: bool = True
    ENABLE_TRANSACTION_BLOCKING: bool = True
    
    class Config:
        env_file = ENV_FILE
        extra = "ignore"


settings = Settings()
