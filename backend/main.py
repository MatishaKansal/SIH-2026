"""
Enterprise Call Fraud Prevention Platform - Backend API
FastAPI server with WebSocket audio streaming and fraud detection
"""

import json
import logging
import hashlib
import io
import base64
import hmac
from uuid import uuid4
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from app.config import settings
from app.services.audio_buffer import AudioBuffer
from app.services.business_engine import BusinessEngine
from app.supabase_client import supabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the API lifecycle."""
    logger.info("Starting up Enterprise Fraud Prevention Platform")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Enterprise Call Fraud Prevention API",
    description="Real-time AI-powered voice authentication and fraud detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure per environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
audio_buffers = {}  # call_id -> AudioBuffer
business_engine = BusinessEngine()


# ============================================================================
# DATA MODELS
# ============================================================================

class CallInitPayload(BaseModel):
    """Initial call metadata"""
    call_id: str
    claimed_identity: str
    source_phone: str
    transaction_amount: Optional[float] = None
    enrolled_speaker_id: Optional[str] = None


class TransactionVerificationRequest(BaseModel):
    """Step-up verification request"""
    call_id: str
    action_type: str
    amount: Optional[float] = None
    step_up_method: str  # CALLBACK, OTP, MANAGER_OVERRIDE
    override_reason: Optional[str] = None


class TransactionVerificationResponse(BaseModel):
    """Transaction approval response"""
    status: str  # APPROVED, REJECTED
    transaction_id: str
    audit_hash: str
    message: str


class AuthCredentials(BaseModel):
    email: str
    password: str


class SignupRequest(AuthCredentials):
    name: str


def hash_password(password: str) -> str:
    salt = hashlib.sha256(uuid4().bytes).digest()[:16]
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 120_000)
    return f"pbkdf2_sha256$120000${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_value, digest_value = encoded.split('$')
        if algorithm != 'pbkdf2_sha256':
            return False
        salt = base64.b64decode(salt_value)
        expected = base64.b64decode(digest_value)
        actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def public_user(user: dict) -> dict:
    metadata = user.get("user_metadata") or {}
    return {
        "user_id": user.get("id"),
        "name": metadata.get("full_name") or user.get("full_name") or "Analyst",
        "email": user.get("email"),
    }


async def create_supabase_call(call_id: str, payload: dict) -> dict:
    rows = await supabase.insert("call_sessions", {
        "call_id": call_id,
        "claimed_identity": payload.get("claimed_identity") or "Unknown",
        "source_phone": payload.get("source_phone"),
        "transaction_amount": payload.get("transaction_amount") or 0,
        "status": "IN_PROGRESS",
        "risk_level": "SAFE",
        "final_decision": "PENDING",
    })
    return rows


async def persist_window(call: dict, window_number: int, audio_bytes: bytes, ai_response: dict) -> dict:
    audio_window = await supabase.insert("audio_windows", {
        "call_id": call["id"],
        "window_number": window_number,
        "start_time_ms": (window_number - 1) * 500,
        "end_time_ms": (window_number - 1) * 500 + 2000,
        "vad_active": bool(ai_response.get("vad_active")),
    })
    metrics = ai_response.get("metrics", {})
    prediction = await supabase.insert("ai_predictions", {
        "call_id": call["id"],
        "audio_window_id": audio_window.get("id"),
        "p_ai_window": ai_response.get("p_ai_window"),
        "aasist_spoof_score": metrics.get("aasist_spoof_score"),
        "prosody_anomaly_score": metrics.get("prosody_anomaly_score"),
        "speaker_match_score": metrics.get("speaker_match_score"),
        "wald_decision": ai_response.get("wald_decision"),
        "cumulative_risk_score": ai_response.get("cumulative_risk_score"),
        "latency_ms": round(float(ai_response.get("latency_ms") or 0)),
    })
    risk_score = float(ai_response.get("cumulative_risk_score") or 0)
    risk_level = ai_response.get("wald_decision") or "SAFE"
    await supabase.update(
        "call_sessions",
        {"id": f"eq.{call['id']}"},
        {
            "current_risk_score": risk_score,
            "max_risk_score": risk_score,
            "risk_level": risk_level,
            "final_decision": "BLOCKED" if risk_level == "HIGH_RISK" else "PENDING",
        },
    )
    await supabase.insert("risk_events", {
        "call_id": call["id"],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "action_required": "BLOCK_TRANSACTION" if risk_level == "HIGH_RISK" else "ALLOW",
        "explanation": metrics,
        "recommended_action": "SECURITY_ESCALATION" if risk_level == "HIGH_RISK" else "CONTINUE_MONITORING",
    })
    return prediction


async def complete_supabase_call(call_id: str, status: str = "COMPLETED") -> None:
    await supabase.update(
        "call_sessions",
        {"call_id": f"eq.{call_id}"},
        {"status": status, "ended_at": datetime.utcnow().isoformat()},
    )
    calls = await supabase.select(
        "call_sessions",
        columns="id,call_id,risk_level,final_decision,current_risk_score",
        params={"call_id": f"eq.{call_id}"},
    )
    call = calls[0] if calls else {}
    audit_data = {
        "call_id": call_id,
        "action_taken": "BLOCKED" if call.get("final_decision") == "BLOCKED" else "ALLOWED",
        "risk_score": float(call.get("current_risk_score") or 0),
        "risk_level": call.get("risk_level") or "SAFE",
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }
    await supabase.insert("audit_ledger", {
        "event_id": str(uuid4()),
        "call_id": call.get("id"),
        "event_type": "CALL_COMPLETED",
        "event_data": audit_data,
        "event_hash": compute_audit_hash(audit_data),
    })


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_audit_hash(event_data: dict) -> str:
    """Generate SHA-256 hash of event for immutable audit trail"""
    json_str = json.dumps(event_data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.post("/api/v1/auth/signup")
async def signup(request: SignupRequest):
    """Create a Supabase Auth account and its public users profile."""
    email = request.email.strip().lower()
    name = request.name.strip()
    if not name or not email or len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Name, valid email, and an 8-character password are required")
    try:
        auth_result = await supabase.auth_signup(email, request.password, name)
        auth_user = auth_result.get("user") or auth_result
        await supabase.insert("users", {
            "id": auth_user["id"],
            "full_name": name,
            "email": email,
        })
        return public_user({**auth_user, "full_name": name})
    except httpx.HTTPStatusError as error:
        try:
            error_body = error.response.json()
        except ValueError:
            error_body = {}
        detail = error_body.get("msg") or error_body.get("message") or "Unable to create account"
        raise HTTPException(status_code=409 if error.response.status_code in (409, 422) else 502, detail=detail)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))


@app.post("/api/v1/auth/signin")
async def signin(request: AuthCredentials):
    """Validate credentials with Supabase Auth."""
    email = request.email.strip().lower()
    try:
        auth_result = await supabase.auth_signin(email, request.password)
        auth_user = auth_result.get("user") or {}
        profiles = await supabase.select(
            "users",
            params={"id": f"eq.{auth_user.get('id')}"},
            use_service_role=True,
        )
        profile = profiles[0] if profiles else {}
        if not profile:
            profile = await supabase.insert("users", {
                "id": auth_user["id"],
                "full_name": (auth_user.get("user_metadata") or {}).get("full_name") or "Analyst",
                "email": email,
            })
        return public_user({**auth_user, **profile})
    except httpx.HTTPStatusError as error:
        try:
            error_body = error.response.json()
        except ValueError:
            error_body = {}
        message = error_body.get("msg") or error_body.get("message") or ""
        if "not confirmed" in message.lower():
            raise HTTPException(status_code=403, detail="Confirm your email before signing in")
        raise HTTPException(status_code=401, detail="Invalid email or password")
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "fraud-prevention-api"
    }


@app.get("/api/v1/calls")
async def list_calls():
    """List call sessions from Supabase."""
    calls = await supabase.select(
        "call_sessions",
        columns="id,call_id,claimed_identity,source_phone,status,max_risk_score,current_risk_score,risk_level,final_decision,transaction_amount,started_at,ended_at,created_at",
        params={"status": "neq.IN_PROGRESS", "order": "created_at.desc", "limit": 10},
    )
    call_ids = [call["id"] for call in calls]
    predictions_by_call = {call_id: [] for call_id in call_ids}
    if call_ids:
        predictions = await supabase.select(
            "ai_predictions",
            columns="call_id,aasist_spoof_score,prosody_anomaly_score,speaker_match_score,latency_ms",
            params={"call_id": f"in.({','.join(call_ids)})", "order": "created_at.asc"},
        )
        for prediction in predictions:
            predictions_by_call.setdefault(prediction["call_id"], []).append(prediction)
    result = []
    for call in calls:
        predictions = predictions_by_call.get(call["id"], [])
        count = len(predictions)
        average = lambda key: round(sum(float(row.get(key) or 0) for row in predictions) / count, 4) if count else None
        result.append({
            "call_id": call["call_id"],
            "claimed_identity": call["claimed_identity"],
            "source_phone": call.get("source_phone"),
            "status": call["status"],
            "max_risk_score": call["max_risk_score"],
            "decision": call.get("final_decision"),
            "risk_level": call["risk_level"],
            "transaction_amount": call.get("transaction_amount"),
            "created_at": call["created_at"],
            "acoustic_anomaly_score": average("aasist_spoof_score"),
            "prosody_deviation_score": average("prosody_anomaly_score"),
            "voiceprint_mismatch_score": average("speaker_match_score"),
            "ai_inference_count": count,
            "avg_latency_ms": average("latency_ms"),
        })
    return result


@app.get("/api/v1/speaker-profiles")
async def list_speaker_profiles():
    """List enrolled speaker profiles from Supabase."""
    return await supabase.select(
        "speaker_profiles",
        columns="id,identity_name,identity_type,external_reference,voiceprint_reference,status,enrolled_at",
        params={"order": "enrolled_at.desc"},
    )


@app.get("/api/v1/calls/{call_id}")
async def get_call_details(call_id: str):
    """Get detailed call session information from Supabase."""
    calls = await supabase.select("call_sessions", params={"call_id": f"eq.{call_id}"})
    if not calls:
        raise HTTPException(status_code=404, detail="Call not found")
    call = calls[0]
    return {
        **call,
    }


@app.get("/api/v1/audit/logs")
async def get_audit_logs(limit: int = 100):
    """Retrieve the immutable Supabase audit ledger."""
    logs = await supabase.select(
        "audit_ledger",
        columns="event_id,call_id,event_type,event_data,event_hash,created_at",
        params={"order": "created_at.desc", "limit": limit},
    )
    return [
        {
            "event_id": l["event_id"],
            "call_id": l.get("call_id"),
            "event_type": l["event_type"],
            "action_taken": (l.get("event_data") or {}).get("action_taken"),
            "risk_score": (l.get("event_data") or {}).get("risk_score", 0),
            "event_hash": l["event_hash"],
            "timestamp": l["created_at"],
        }
        for l in logs
    ]


@app.post("/api/v1/calls/{call_id}/verify-action")
async def verify_transaction(
    call_id: str,
    request: TransactionVerificationRequest,
):
    """
    Verify and approve/reject a high-risk transaction
    Applies step-up authentication (callback, OTP, manager override)
    """
    calls = await supabase.select(
        "call_sessions",
        columns="id,call_id,max_risk_score",
        params={"call_id": f"eq.{call_id}"},
    )
    if not calls:
        raise HTTPException(status_code=404, detail="Call not found")

    # Validate step-up method
    if request.step_up_method not in ["CALLBACK", "OTP", "MANAGER_OVERRIDE"]:
        raise HTTPException(status_code=400, detail="Invalid step-up method")

    call = calls[0]
    approval_status = "APPROVED" if request.step_up_method == "MANAGER_OVERRIDE" else "PENDING"
    message = (
        "Transaction approved via manager override"
        if approval_status == "APPROVED"
        else f"{request.step_up_method} verification created and awaiting confirmation"
    )

    # Generate transaction ID
    transaction_id = f"tx_{call_id}_{int(datetime.utcnow().timestamp() * 1000)}"

    transaction = await supabase.insert("transactions", {
        "transaction_id": transaction_id,
        "call_id": call["id"],
        "action_type": request.action_type,
        "amount": request.amount or 0,
        "status": approval_status,
    })

    verification_method = "DIRECT_CALLBACK" if request.step_up_method == "CALLBACK" else request.step_up_method
    await supabase.insert("verification_events", {
        "call_id": call["id"],
        "transaction_id": transaction.get("id"),
        "method": verification_method,
        "status": approval_status,
        "reason": request.override_reason,
        "verified_at": datetime.utcnow().isoformat() if approval_status == "APPROVED" else None,
    })

    # Create audit event
    audit_data = {
        "call_id": call_id,
        "transaction_id": transaction_id,
        "action_type": request.action_type,
        "amount": request.amount,
        "step_up_method": request.step_up_method,
        "status": approval_status,
        "timestamp": datetime.utcnow().isoformat(),
    }
    audit_hash = compute_audit_hash(audit_data)
    
    await supabase.insert("audit_ledger", {
        "event_id": str(uuid4()),
        "call_id": call["id"],
        "transaction_id": transaction.get("id"),
        "event_type": "TRANSACTION_VERIFICATION",
        "event_data": audit_data,
        "event_hash": audit_hash,
    })
    
    return TransactionVerificationResponse(
        status=approval_status,
        transaction_id=transaction_id,
        audit_hash=audit_hash,
        message=message
    )


# ============================================================================
# WEBSOCKET ENDPOINT: Real-Time Audio Streaming
# ============================================================================

@app.post("/api/v1/calls/upload")
async def upload_recorded_call(
    audio: UploadFile = File(...),
    claimed_identity: str = Form("Unknown"),
    source_phone: str = Form(""),
    transaction_amount: float = Form(0),
):
    """Analyze a recorded WAV/AIFF/FLAC/MP3 call as PCM windows."""
    call_id = None
    filename = (audio.filename or "").lower()
    raw_audio = await audio.read()
    supported_types = {
        "audio/wav", "audio/x-wav", "audio/wave", "audio/flac", "audio/x-flac",
        "audio/aiff", "audio/x-aiff", "audio/mpeg", "audio/mp3",
    }
    supported_extensions = (".mp3", ".mpeg", ".mpga", ".wav", ".flac", ".aiff", ".aif")
    is_wav_signature = raw_audio.startswith(b"RIFF") and raw_audio[8:12] == b"WAVE"
    is_mp3_signature = raw_audio.startswith(b"ID3") or (
        len(raw_audio) > 1 and raw_audio[0] == 0xFF and raw_audio[1] & 0xE0 == 0xE0
    )
    if audio.content_type not in supported_types and not filename.endswith(supported_extensions) and not (is_wav_signature or is_mp3_signature):
        raise HTTPException(
            status_code=415,
            detail="Allowed files: MP3, MPEG, WAV, FLAC, AIFF, and AIF only.",
        )

    try:
        import numpy as np
        is_mp3 = (
            filename.endswith(".mp3")
            or audio.content_type in {"audio/mpeg", "audio/mp3"}
            or is_mp3_signature
        )
        if is_mp3:
            import av
            container = av.open(io.BytesIO(raw_audio), mode="r")
            stream = next(stream for stream in container.streams if stream.type == "audio")
            decoded_frames = [frame.to_ndarray() for frame in container.decode(stream)]
            container.close()
            if not decoded_frames:
                raise ValueError("MP3 file contains no audio frames")
            samples = np.concatenate(decoded_frames, axis=-1)
            if samples.ndim > 1:
                samples = samples.mean(axis=0)
            if np.issubdtype(samples.dtype, np.integer):
                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            else:
                samples = samples.astype(np.float32)
            sample_rate = stream.rate
        else:
            import soundfile as sf
            samples, sample_rate = sf.read(io.BytesIO(raw_audio), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sample_rate != settings.SAMPLE_RATE:
            from scipy import signal
            samples = signal.resample(samples, int(len(samples) * settings.SAMPLE_RATE / sample_rate))
        samples = np.clip(samples, -1, 1)
        call_id = f"upload-{uuid4()}"
        call = await create_supabase_call(call_id, {
            "claimed_identity": claimed_identity,
            "source_phone": source_phone,
            "transaction_amount": transaction_amount,
        })
        buffer = AudioBuffer(window_duration=2.0, hop_size=0.5)
        results = []
        pcm = (samples * 32767).astype(np.int16).tobytes()
        for window_number, window in enumerate(buffer.add_audio_chunk(pcm), 1):
            ai_response = await call_ai_service(call_id, window_number, window)
            await persist_window(call, window_number, window, ai_response)
            results.append(ai_response)
        await complete_supabase_call(call_id)
        return {"call_id": call_id, "windows_processed": len(results), "results": results}
    except HTTPException:
        if call_id:
            await complete_supabase_call(call_id, status="TERMINATED")
        raise
    except Exception as error:
        logger.exception("Recorded call processing failed")
        if call_id:
            await complete_supabase_call(call_id, status="TERMINATED")
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.websocket("/api/v1/calls/{call_id}/stream")
async def websocket_stream(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for continuous audio streaming and real-time fraud detection
    
    Expected messages:
    1. Initial metadata: {"event": "start_call", "call_id": "...", "claimed_identity": "..."}
    2. Audio chunks: binary audio data (16 kHz mono PCM)
    3. End call: {"event": "end_call"}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for call {call_id}")
    
    try:
        call_payload = {"claimed_identity": "Unknown", "source_phone": "", "transaction_amount": 0}
        call = None
        window_number = 0
        audio_buffers[call_id] = AudioBuffer(window_duration=2.0, hop_size=0.5)
        
        # Process incoming messages
        while True:
            try:
                # Receive message (either JSON or binary)
                data = await websocket.receive()
                
                if "text" in data:
                    # JSON metadata message
                    msg = json.loads(data["text"])
                    
                    if msg.get("event") == "start_call":
                        call_payload.update(msg)
                        call = await create_supabase_call(call_id, call_payload)
                        
                        logger.info(
                            f"Call started: {call_payload['claimed_identity']} from {call_payload['source_phone']}"
                        )
                        
                        # Send acknowledgment
                        await websocket.send_json({
                            "event": "call_started",
                            "call_id": call_id,
                            "status": "READY"
                        })
                    
                    elif msg.get("event") == "end_call":
                        logger.info(f"Call ended: {call_id}")
                        if call is not None:
                            await complete_supabase_call(call_id)
                        break
                
                elif "bytes" in data:
                    # Audio chunk received
                    audio_chunk = data["bytes"]
                    
                    # Add to buffer and get complete windows
                    windows = audio_buffers[call_id].add_audio_chunk(audio_chunk)
                    
                    # Process each complete window
                    for window in windows:
                        if call is None:
                            raise RuntimeError("Send start_call before sending audio")
                        window_number += 1
                        
                        # Send to AI service for inference
                        ai_response = await call_ai_service(
                            call_id=call_id,
                            window_id=window_number,
                            audio_bytes=window
                        )
                        
                        await persist_window(call, window_number, window, ai_response)
                        action = business_engine.evaluate_risk(ai_response)
                        
                        # Send enriched response to frontend
                        await websocket.send_json({
                            "event": "window_result",
                            "call_id": call_id,
                            "window_id": window_number,
                            "status": "IN_PROGRESS",
                            "risk_level": action["risk_level"],
                            "cumulative_risk_score": ai_response.get("cumulative_risk_score", 0),
                            "action_required": action["action"],
                            "explanation": {
                                "spoof_confidence": f"{ai_response.get('metrics', {}).get('aasist_spoof_score', 0)*100:.0f}%",
                                "prosody": f"Anomaly score: {ai_response.get('metrics', {}).get('prosody_anomaly_score', 0):.2f}",
                                "target_match": f"Speaker match: {ai_response.get('metrics', {}).get('speaker_match_score', 0)*100:.0f}%"
                            }
                        })
                        
                        # If HIGH_RISK, lock transaction
                        if action["risk_level"] == "CRITICAL":
                            logger.warning(f"CRITICAL RISK DETECTED for call {call_id}")
                            await websocket.send_json({
                                "event": "fraud_alert",
                                "risk_level": "CRITICAL",
                                "action": "BLOCK_TRANSACTION",
                                "message": "Suspected AI voice clone detected. Transaction blocked. Awaiting manager verification."
                            })
            
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                await websocket.send_json({
                    "event": "error",
                    "message": str(e)
                })
                break
    
    except Exception as e:
        logger.error(f"WebSocket error for call {call_id}: {e}")
    
    finally:
        # Cleanup
        if call_id in audio_buffers:
            del audio_buffers[call_id]
        if call is not None:
            await complete_supabase_call(call_id)
        logger.info(f"WebSocket closed for call {call_id}")


# ============================================================================
# AI SERVICE COMMUNICATION
# ============================================================================

async def call_ai_service(call_id: str, window_id: int, audio_bytes: bytes) -> dict:
    """
    Send audio window to AI microservice and receive detection results
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.AI_SERVICE_URL}/detect-window",
                json={
                    "call_id": call_id,
                    "window_id": window_id,
                    "audio_base64": audio_bytes.hex()  # Send as hex string
                },
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"AI service error: {e}")
        raise HTTPException(status_code=503, detail="AI detection service is unavailable") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
