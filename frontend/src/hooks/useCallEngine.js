import { useRef, useState, useEffect } from 'react';

function mapCallToState(call) {
  const identity = call.claimed_identity || 'Unknown caller';
  const rawRiskScore = Number(call.max_risk_score || call.current_risk_score || 0);
  const riskScore = rawRiskScore <= 1 ? rawRiskScore * 100 : rawRiskScore;
  const isHighRisk = call.risk_level === 'HIGH_RISK' || call.decision === 'BLOCKED';

  return {
    callState: isHighRisk ? 'HIGH_RISK' : call.risk_level || 'SAFE',
    riskScore,
    metrics: {
      acoustic: call.acoustic_anomaly_score == null ? null : Number(call.acoustic_anomaly_score) * 100,
      prosody: call.prosody_deviation_score == null ? null : Number(call.prosody_deviation_score) * 100,
      voiceprint: call.voiceprint_mismatch_score == null ? null : (1 - Number(call.voiceprint_mismatch_score)) * 100,
    },
    callerInfo: {
      identity,
      number: call.source_phone || 'Unavailable',
      duration: '--:--:--',
      location: 'Backend call session',
      company: 'Enterprise Platform',
      avatar: identity.slice(0, 2).toUpperCase(),
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

function mapUploadResultToState(result, metadata) {
  const latest = result.results?.[result.results.length - 1] || {};
  const score = Number(latest.cumulative_risk_score || latest.p_ai_window || 0);
  const riskScore = score <= 1 ? score * 100 : score;
  const isHighRisk = latest.wald_decision === 'HIGH_RISK';

  return {
    callState: latest.wald_decision || 'SAFE',
    riskScore,
    metrics: {
      acoustic: Number(latest.metrics?.aasist_spoof_score || 0) * 100,
      prosody: Number(latest.metrics?.prosody_anomaly_score || 0) * 100,
      voiceprint: (1 - Number(latest.metrics?.speaker_match_score || 0)) * 100,
    },
    callerInfo: {
      identity: metadata.claimed_identity || 'Unknown',
      number: metadata.source_phone || 'Unavailable',
      duration: '--:--:--',
      location: 'Uploaded recording',
      company: 'Enterprise Platform',
      avatar: (metadata.claimed_identity || 'UC').slice(0, 2).toUpperCase(),
    },
    transaction: {
      type: 'Transaction',
      amount: 'Not specified',
      destination: 'Transaction details unavailable',
      status: isHighRisk ? 'BLOCKED' : 'PENDING_AUTH',
      reference: result.call_id,
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
  const [state, setState] = useState(() => {
    try {
      const snapshot = window.sessionStorage.getItem('echoguard.last-analysis');
      return snapshot ? JSON.parse(snapshot) : null;
    } catch {
      return null;
    }
  });
  const [currentView, setCurrentView] = useState('dashboard');
  const [auditLogs, setAuditLogs] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [liveStatus, setLiveStatus] = useState('idle');
  const [liveError, setLiveError] = useState(null);
  const [activeCallId, setActiveCallId] = useState(null);
  const [analysisStats, setAnalysisStats] = useState({ avgLatency: null, aiInferences: 0 });
  const liveRef = useRef({ socket: null, stream: null, context: null, processor: null });
  const refreshInFlight = useRef(false);

  async function refreshAuditLogs() {
    const response = await fetch('/api/v1/audit/logs');
    if (!response.ok) throw new Error('Audit log refresh failed');
    setAuditLogs((await response.json()).map(mapAuditLog));
  }

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
          const rawScore = Number(message.cumulative_risk_score || 0);
          const metrics = message.explanation || {};
          setState((current) => current ? {
            ...current,
            riskScore: rawScore <= 1 ? rawScore * 100 : rawScore,
            callState: message.risk_level || current.callState,
            metrics: {
              acoustic: Number(metrics.spoof_confidence?.replace('%', '') || 0),
              prosody: Number(metrics.prosody?.match(/[\d.]+$/)?.[0] || 0) * 100,
              voiceprint: Number(metrics.target_match?.replace('%', '') || 0),
            },
          } : current);
          setAnalysisStats((current) => ({
            avgLatency: message.latency_ms ?? current.avgLatency,
            aiInferences: current.aiInferences + 1,
          }));
        }
        if (message.event === 'error') setLiveError(message.message);
      };
      socket.onerror = () => setLiveError('Live call connection failed');
      socket.onclose = async () => {
        setLiveStatus('idle');
        try { await refreshAuditLogs(); } catch { /* health status remains usable */ }
      };
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
      const nextState = mapUploadResultToState(result, metadata);
      setState(nextState);
      window.sessionStorage.setItem('echoguard.last-analysis', JSON.stringify(nextState));
      const results = result.results || [];
      setAnalysisStats({
        avgLatency: results.length
          ? results.reduce((sum, item) => sum + Number(item.latency_ms || 0), 0) / results.length
          : null,
        aiInferences: results.length,
      });
      await refreshAuditLogs();
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

    async function fetchWithTimeout(url, timeoutMs = 4000) {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
      try {
        return await fetch(url, { signal: controller.signal });
      } finally {
        window.clearTimeout(timeout);
      }
    }

    async function loadBackendData() {
      if (refreshInFlight.current) return;
      refreshInFlight.current = true;
      try {
        const callsResponse = await fetchWithTimeout('/api/v1/calls');
        if (!callsResponse.ok) throw new Error('Call history request failed');
        const calls = await callsResponse.json();
        if (!isMounted) return;

        setBackendStatus('connected');
        const callWithResults = calls.find((call) => Number(call.ai_inference_count || 0) > 0) || calls[0];
        if (callWithResults) {
          const nextState = mapCallToState(callWithResults);
          setState(nextState);
          window.sessionStorage.setItem('echoguard.last-analysis', JSON.stringify(nextState));
        }
        if (callWithResults) {
          setAnalysisStats({
            avgLatency: callWithResults.avg_latency_ms,
            aiInferences: Number(callWithResults.ai_inference_count || 0),
          });
        }
        try {
          const auditResponse = await fetch('/api/v1/audit/logs');
          if (auditResponse.ok) setAuditLogs((await auditResponse.json()).map(mapAuditLog));
        } catch {
          // Call history remains usable if the audit endpoint is temporarily unavailable.
        }
        setError(null);
      } catch (requestError) {
        if (!isMounted) return;
        setBackendStatus('offline');
        setError(requestError.name === 'AbortError' ? 'Backend request timed out' : requestError.message);
      } finally {
        refreshInFlight.current = false;
        if (isMounted) setIsLoading(false);
      }
    }

    loadBackendData();
    const retryTimer = window.setTimeout(loadBackendData, 1200);
    const refreshTimer = window.setInterval(loadBackendData, 15000);
    const handleRefresh = () => loadBackendData();
    window.addEventListener('focus', handleRefresh);
    window.addEventListener('pageshow', handleRefresh);
    return () => {
      isMounted = false;
      window.clearTimeout(retryTimer);
      window.clearInterval(refreshTimer);
      window.removeEventListener('focus', handleRefresh);
      window.removeEventListener('pageshow', handleRefresh);
    };
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
    analysisStats,
  };
}
