import TopBar from '../components/TopBar';
import CallerCard from '../components/CallerCard';
import RiskGauge from '../components/RiskGauge';
import MetricsBars from '../components/MetricsBars';
import TransactionPanel from '../components/TransactionPanel';
import CallControls from '../components/CallControls';
import { BrainCircuit, ShieldCheck, ShieldAlert, TrendingUp, Clock3 } from 'lucide-react';

export default function DashboardView({
  callState,
  riskScore,
  metrics,
  callerInfo,
  transaction,
  isSimulating,
  liveStatus,
  liveError,
  startLiveCall,
  stopLiveCall,
  uploadRecording,
  auditLogs = [],
}) {
  if (!callState) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <TopBar />
        <div className="flex-1 flex items-center justify-center p-5">
          <div className="w-full max-w-xl space-y-4">
            <div className="glass-card p-8 text-center">
              <h2 className="text-base font-semibold text-white">No active calls</h2>
              <p className="mt-2 text-sm text-slate-500">Start a live microphone session or upload a recorded call for analysis.</p>
            </div>
            <CallControls {...{ liveStatus, liveError, startLiveCall, stopLiveCall, uploadRecording }} />
          </div>
        </div>
      </div>
    );
  }

  const isSafe = callState === 'SAFE';
  const blockedCalls = auditLogs.filter((log) => log.decision === 'BLOCKED').length;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Top Bar */}
      <TopBar />

      {/* ── Main Content ── */}
      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        <CallControls {...{ liveStatus, liveError, startLiveCall, stopLiveCall, uploadRecording }} />

        {/* Global alert banner */}
        {!isSafe && (
          <div
            className="animate-slide-down flex items-center gap-4 p-4 rounded-xl border border-red-500/50 bg-red-500/10 glow-red"
            role="alert"
          >
            <ShieldAlert className="w-6 h-6 text-red-400 flex-shrink-0 animate-pulse" />
            <div className="flex-1">
              <p className="text-sm font-bold text-red-300">⚠ DEEPFAKE ATTACK IN PROGRESS</p>
              <p className="text-xs text-red-400/70 mt-0.5">
                EchoGuard AI has detected a synthetic voice clone impersonating {callerInfo.identity}. All critical actions have been suspended.
              </p>
            </div>
            <div className="text-right flex-shrink-0">
              <p className="text-[10px] text-red-500 font-mono uppercase">Incident ID</p>
              <p className="text-xs text-red-300 font-mono">INC-{Date.now().toString(36).toUpperCase()}</p>
            </div>
          </div>
        )}

        {/* ── 3-column Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* ── LEFT: Caller + Audio ── */}
          <div className="lg:col-span-1 flex flex-col gap-5">
            <CallerCard callerInfo={callerInfo} isSafe={isSafe} />

            {/* Quick stats */}
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Calls Today',    value: auditLogs.length, icon: Clock3,       color: 'text-slate-300' },
                { label: 'Threats Caught', value: blockedCalls,     icon: ShieldAlert, color: 'text-red-400' },
                { label: 'Avg Latency',    value: '--',             icon: Clock3,       color: 'text-emerald-400' },
                { label: 'AI Inferences',  value: '--',             icon: BrainCircuit, color: 'text-amber-400' },
              ].map(({ label, value, icon: Icon, color }) => (
                <div key={label} className="glass-card p-3 flex flex-col gap-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
                    <Icon className={`w-3.5 h-3.5 ${color}`} />
                  </div>
                  <span className={`text-lg font-extrabold ${color}`}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── CENTER/RIGHT: Risk Engine ── */}
          <div className="lg:col-span-2 flex flex-col gap-5">
            <div className="glass-card p-5">
              {/* Header */}
              <div className="flex items-center gap-2 mb-5">
                <BrainCircuit className={`w-4 h-4 ${isSafe ? 'text-emerald-400' : 'text-red-400'}`} />
                <h3 className="text-sm font-semibold text-white">AI Risk Engine</h3>
                <div className={`ml-auto flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider
                  ${isSafe ? 'text-emerald-500' : 'text-red-400 animate-pulse'}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${isSafe ? 'bg-emerald-400' : 'bg-red-500'}`} />
                  {isSafe ? 'Nominal' : 'ALERT'}
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center gap-8">
                {/* Risk Gauge */}
                <div className={`flex-shrink-0 p-5 rounded-2xl transition-all duration-500 ${isSafe ? '' : 'glow-red'}`}
                  style={{ background: 'rgba(2,6,23,0.5)' }}>
                  <RiskGauge score={riskScore} />
                </div>

                {/* Metrics bars */}
                <div className="flex-1 w-full space-y-1">
                  <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-4">Analysis Breakdown</p>
                  <MetricsBars metrics={metrics} isSafe={isSafe} />

                  {/* Confidence note */}
                  <div className="mt-5 pt-4 border-t border-slate-800/60">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-600">VAD Voice Match</span>
                      <span className={`font-semibold font-mono ${isSafe ? 'text-emerald-400' : 'text-red-400'}`}>
                        --
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] mt-1">
                      <span className="text-slate-600">Inference Time</span>
                      <span className="text-slate-400 font-mono">--</span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] mt-1">
                      <span className="text-slate-600">Analysis Frames</span>
                      <span className="text-slate-400 font-mono">--</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* ── Transaction Panel ── */}
            <TransactionPanel transaction={transaction} isSafe={isSafe} />
          </div>
        </div>

        {/* Safe state indicator */}
        {isSafe && (
          <div className="flex items-center justify-center gap-2 py-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span className="text-xs text-emerald-400 font-medium">All security checks passed — Voice authentication nominal</span>
          </div>
        )}
      </div>
    </div>
  );
}
