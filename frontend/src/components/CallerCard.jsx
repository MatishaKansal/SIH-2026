import { Phone, MapPin, Building2, Clock, Fingerprint } from 'lucide-react';
import AudioVisualizer from './AudioVisualizer';

export default function CallerCard({ callerInfo, isSafe, isHighRisk = false }) {
  const isInvestigate = !isSafe && !isHighRisk;
  const statusClass = isSafe
    ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
    : isInvestigate
    ? 'bg-amber-400/15 text-amber-400 border border-amber-400/30'
    : 'bg-red-500/15 text-red-400 border border-red-500/30 animate-pulse';
  return (
    <div className={`glass-card p-5 scan-lines relative overflow-hidden transition-all duration-500 ${isSafe ? 'glow-green' : 'glow-red'}`}>
      {/* Corner accent */}
      <div className={`absolute top-0 right-0 w-16 h-16 opacity-10 ${isSafe ? 'bg-emerald-400' : 'bg-red-500'}`}
        style={{ clipPath: 'polygon(100% 0, 0 0, 100% 100%)' }}
      />

      {/* Header */}
      <div className="flex items-start gap-4 mb-5">
        {/* Avatar */}
        <div className="relative flex-shrink-0">
          <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-lg font-bold text-white
              ${isSafe ? 'bg-gradient-to-br from-emerald-600 to-teal-700' : isInvestigate ? 'bg-gradient-to-br from-amber-500 to-orange-700' : 'bg-gradient-to-br from-red-700 to-rose-900'}`}>
            {callerInfo.avatar}
          </div>
          {/* Online ring */}
          <span className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-[#030712] flex items-center justify-center
            ${isSafe ? 'bg-emerald-400' : isInvestigate ? 'bg-amber-400' : 'bg-red-500 animate-pulse'}`} />
        </div>

        {/* Identity details */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-base font-bold text-white truncate">{callerInfo.identity}</h2>
            <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wider
              ${statusClass}`}>
              {isSafe ? 'Verified' : isInvestigate ? 'Investigate' : 'Spoofed'}
            </span>
          </div>
          <div className="flex items-center gap-1.5 mt-1">
            <Phone className="w-3 h-3 text-slate-500" />
            <span className="text-xs text-slate-400 font-mono">{callerInfo.number}</span>
          </div>
          <div className="flex items-center gap-3 mt-1.5 flex-wrap">
            <span className="flex items-center gap-1 text-[11px] text-slate-500">
              <Building2 className="w-3 h-3" />{callerInfo.company}
            </span>
            <span className="flex items-center gap-1 text-[11px] text-slate-500">
              <MapPin className="w-3 h-3" />{callerInfo.location}
            </span>
          </div>
        </div>

        {/* Call duration */}
        <div className="flex-shrink-0 text-right">
          <div className="flex items-center gap-1.5 justify-end">
            <span className={`w-1.5 h-1.5 rounded-full ${isSafe ? 'bg-emerald-500' : isInvestigate ? 'bg-amber-400' : 'bg-red-500 animate-blink'}`} />
            <span className={`text-[10px] uppercase font-semibold tracking-wider ${isSafe ? 'text-emerald-400' : isInvestigate ? 'text-amber-400' : 'text-red-400'}`}>{isSafe ? 'READY' : isInvestigate ? 'REVIEW' : 'ALERT'}</span>
          </div>
          <div className="flex items-center gap-1 mt-1">
            <Clock className="w-3 h-3 text-slate-500" />
            <span className="text-sm font-mono text-slate-300">{callerInfo.duration}</span>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-800/60 mb-5" />

      {/* Audio Visualizer */}
      <AudioVisualizer isSafe={isSafe} />

      {/* Voiceprint fingerprint row */}
      <div className="mt-4 flex items-center gap-2 px-1">
        <Fingerprint className={`w-3.5 h-3.5 flex-shrink-0 ${isSafe ? 'text-emerald-500' : 'text-red-500'}`} />
        <div className="flex-1 h-px bg-slate-800" />
        <span className="text-[10px] font-mono text-slate-600 truncate">
          VPRINT·{isSafe ? '4a2f...91c3·MATCH' : '??·??·NO_MATCH'}
        </span>
        <div className="flex-1 h-px bg-slate-800" />
      </div>
    </div>
  );
}
