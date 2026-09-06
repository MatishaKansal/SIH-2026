import { useRef, useState, useEffect } from 'react';

function mapCallToState(call) {
  const riskScore = Number(call.max_risk_score || call.current_risk_score || 0);
  const isHighRisk = call.risk_level === 'HIGH_RISK' || call.decision === 'BLOCKED';

  return {
    callState: isHighRisk ? 'HIGH_RISK' : call.risk_level || 'SAFE',
    riskScore,
    metrics: {
      acoustic: call.acoustic_anomaly_score,
      prosody: call.prosody_deviation_score,
      voiceprint: call.voiceprint_mismatch_score,
    },
    callerInfo: {
      identity: call.claimed_identity,
      number: call.source_phone || 'Unavailable',
      duration: '--:--:--',
      location: 'Backend call session',
      company: 'Enterprise Platform',
      avatar: call.claimed_identity.slice(0, 2).toUpperCase(),
    },
    transaction: {
      type: 'Transaction',
      amount: call.transaction_amount ? `₹${Number(call.transaction_amount).toLocaleString('en-IN')}` : 'Not specified',
      destination: 'Transaction details unavailable',
      status: isHighRisk ? 'BLOCKED' : 'PENDING_AUTH',
      reference: call.call_id,
    },
  };
}

function mapAuditLog(log) {
  return {
    id: log.call_id || log.event_id,
    timestamp: log.timestamp,
    identity: log.call_id || 'Call session',
    peakRisk: Number(log.risk_score || 0),
    decision: log.action_taken === 'BLOCKED' ? 'BLOCKED' : 'ALLOWED',
    hash: log.event_hash,
    number: 'Unavailable',
  };
}

export function useCallEngine() {
  const [state, setState] = useState(null);
  const [currentView, setCurrentView] = useState('dashboard');
  const [auditLogs, setAuditLogs] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [liveStatus, setLiveStatus] = useState('idle');
  const [liveError, setLiveError] = useState(null);
  const [activeCallId, setActiveCallId] = useState(null);
  const liveRef = useRef({ socket: null, stream: null, context: null, processor: null });

  function downsampleToPcm16(input, inputRate, outputRate = 16000) {
    const ratio = inputRate / outputRate;
    const outputLength = Math.round(input.length / ratio);
    const output = new Int16Array(outputLength);
    for (let index = 0; index < outputLength; index += 1) {
      const sourceIndex = Math.min(Math.floor(index * ratio), input.length - 1);
      const sample = Math.max(-1, Math.min(1, input[sourceIndex]));
      output[index] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    return output.buffer;
  }

  async function startLiveCall(metadata = {}) {
    if (liveStatus === 'connecting' || liveStatus === 'live') return;
    setLiveError(null);
    setLiveStatus('connecting');
    const callId = `live-${crypto.randomUUID()}`;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${protocol}://${window.location.host}/api/v1/calls/${callId}/stream`);
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(downsampleToPcm16(event.inputBuffer.getChannelData(0), context.sampleRate));
        }
      };
      socket.onopen = () => {
        socket.send(JSON.stringify({ event: 'start_call', call_id: callId, ...metadata }));
        source.connect(processor);
        processor.connect(context.destination);
        setActiveCallId(callId);
        setLiveStatus('live');
        setState({
          callState: 'SAFE',
          riskScore: 0,
          metrics: { acoustic: 0, prosody: 0, voiceprint: 0 },
          callerInfo: {
            identity: metadata.claimed_identity || 'Unknown',
            number: metadata.source_phone || 'Unavailable',
            duration: 'LIVE', location: 'Browser microphone', company: 'Enterprise Platform',
            avatar: (metadata.claimed_identity || 'UC').slice(0, 2).toUpperCase(),
          },
          transaction: { type: 'Transaction', amount: 'Not specified', destination: 'Not specified', status: 'PENDING_AUTH', reference: callId },
        });
      };
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.event === 'window_result') {
          setState((current) => current ? { ...current, riskScore: Number(message.cumulative_risk_score || 0), callState: message.risk_level || current.callState } : current);
        }
        if (message.event === 'error') setLiveError(message.message);
      };
      socket.onerror = () => setLiveError('Live call connection failed');
      socket.onclose = () => setLiveStatus('idle');
      liveRef.socket = socket;
      liveRef.stream = stream;
      liveRef.context = context;
      liveRef.processor = processor;
    } catch (requestError) {
      setLiveStatus('idle');
      setLiveError(requestError.message);
    }
  }

  function stopLiveCall() {
    if (liveRef.socket?.readyState === WebSocket.OPEN) {
      liveRef.socket.send(JSON.stringify({ event: 'end_call' }));
      liveRef.socket.close();
    }
    liveRef.processor?.disconnect();
    liveRef.context?.close();
    liveRef.stream?.getTracks().forEach((track) => track.stop());
    liveRef.socket = null;
    liveRef.stream = null;
    liveRef.context = null;
    liveRef.processor = null;
    setActiveCallId(null);
    setLiveStatus('idle');
  }

  async function uploadRecording(file, metadata = {}) {
    setLiveError(null);
    setLiveStatus('uploading');
    const formData = new FormData();
    formData.append('audio', file);
    Object.entries(metadata).forEach(([key, value]) => formData.append(key, value));
    try {
      const response = await fetch('/api/v1/calls/upload', { method: 'POST', body: formData });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Recorded call upload failed');
      setLiveStatus('idle');
      return result;
    } catch (requestError) {
      setLiveStatus('idle');
      setLiveError(requestError.message);
      throw requestError;
    }
  }

  useEffect(() => {
    let isMounted = true;

    async function loadBackendData() {
      try {
        const [healthResponse, callsResponse, auditResponse] = await Promise.all([
          fetch('/health'),
          fetch('/api/v1/calls'),
          fetch('/api/v1/audit/logs'),
        ]);

        if (!healthResponse.ok || !callsResponse.ok || !auditResponse.ok) {
          throw new Error('Backend request failed');
        }

        const calls = await callsResponse.json();
        const logs = await auditResponse.json();
        if (!isMounted) return;

        setBackendStatus('connected');
        setState(calls.length > 0 ? mapCallToState(calls[0]) : null);
        setAuditLogs(logs.map(mapAuditLog));
        setError(null);
      } catch (requestError) {
        if (!isMounted) return;
        setBackendStatus('offline');
        setError(requestError.message);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    loadBackendData();
    return () => { isMounted = false; };
  }, []);

  return {
    ...(state || {}),
    isSimulating: false,
    currentView,
    setCurrentView,
    auditLogs,
    backendStatus,
    isLoading,
    error,
    liveStatus,
    liveError,
    activeCallId,
    startLiveCall,
    stopLiveCall,
    uploadRecording,
  };
}
