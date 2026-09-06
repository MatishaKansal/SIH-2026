import { Cpu, Wifi } from 'lucide-react';

export default function TopBar() {
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
            <span className="text-xs text-[#5d7085]">CPU <span className="text-[#8294a6] font-mono">--</span></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Wifi className="w-3.5 h-3.5 text-[#63778d]" />
            <span className="text-xs text-[#5d7085]">Latency <span className="text-[#8294a6] font-mono">--</span></span>
          </div>
        </div>
      </div>
    </header>
  );
}
