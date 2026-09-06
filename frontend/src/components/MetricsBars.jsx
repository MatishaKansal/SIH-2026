function MetricBar({ label, value, isSafe }) {
  const hasValue = Number.isFinite(Number(value));
  const numericValue = hasValue ? Number(value) : 0;
  const color = isSafe
    ? 'from-emerald-500 to-emerald-400'
    : numericValue > 80
    ? 'from-red-700 to-red-500'
    : numericValue > 60
    ? 'from-orange-600 to-amber-400'
    : 'from-emerald-500 to-emerald-400';

  const textColor = isSafe
    ? 'text-emerald-400'
    : numericValue > 80
    ? 'text-red-400'
    : 'text-amber-400';

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400 font-medium">{label}</span>
        <span className={`text-xs font-bold font-mono ${hasValue ? textColor : 'text-slate-400'}`}>{hasValue ? `${numericValue}%` : '--'}</span>
      </div>
      <div className="h-2 rounded-full bg-slate-800/80 overflow-hidden">
        <div
          className={`metric-bar-fill h-full rounded-full bg-gradient-to-r ${color}`}
          style={{
            width: `${hasValue ? numericValue : 0}%`,
            boxShadow: hasValue && (isSafe || numericValue < 60)
              ? '0 0 8px rgba(52,211,153,0.5)'
              : numericValue > 80
              ? '0 0 8px rgba(239,68,68,0.7)'
              : '0 0 8px rgba(251,191,36,0.6)',
          }}
        />
      </div>
    </div>
  );
}

export default function MetricsBars({ metrics, isSafe }) {
  return (
    <div className="space-y-4">
      <MetricBar label="Acoustic Anomaly" value={metrics.acoustic} isSafe={isSafe} />
      <MetricBar label="Prosody Deviation" value={metrics.prosody} isSafe={isSafe} />
      <MetricBar label="Voiceprint Mismatch" value={metrics.voiceprint} isSafe={isSafe} />
    </div>
  );
}
