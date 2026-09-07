import { Cpu, Wifi } from 'lucide-react';

export default function TopBar({ analysisStats = {}, backendStatus = 'checking' }) {
  const latency = Number.isFinite(Number(analysisStats.avgLatency))
    ? `${Math.round(Number(analysisStats.avgLatency))} ms`
    : '--';
  const cpu = typeof navigator !== 'undefined' && navigator.hardwareConcurrency
    ? `${navigator.hardwareConcurrency} threads`
    : '--';
  return (
    <header className="flex items-center justify-between px-8 py-5 border-b border-[#e2eaf1] bg-white/75 backdrop-blur-sm flex-shrink-0 z-10">
      {/* Left: Page title & breadcrumb */}
      <div>
        <h1 className="text-[25px] font-extrabold text-[#172b45]">Active Call Monitor</h1>
        <p className="text-[12px] text-[#8192a6] mt-0.5 flex items-center gap-2">
          Intelligence Platform <span>•</span> Voice Analytics <span>•</span> <span className="text-[#4d718c]">Live Monitor</span>
        </p>
      </div>

      {/* Right: Stats */}
      <div className="flex items-center gap-4">
        {/* Mini system stats */}
        <div className="hidden md:flex items-center gap-4 mr-2">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-[#63778d]" />
            <span className="text-xs text-[#5d7085]">CPU <span className="text-[#8294a6] font-mono">{cpu}</span></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Wifi className="w-3.5 h-3.5 text-[#63778d]" />
            <span className="text-xs text-[#5d7085]">Latency <span className={`font-mono ${backendStatus === 'connected' ? 'text-emerald-600' : 'text-[#8294a6]'}`}>{latency}</span></span>
          </div>
        </div>
      </div>
    </header>
  );
}
