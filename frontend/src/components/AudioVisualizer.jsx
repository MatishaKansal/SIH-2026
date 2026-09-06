import { useMemo } from 'react';

/* 18 bars with randomised animation parameters */
const BARS = Array.from({ length: 18 }, (_, i) => ({
  id: i,
  dur: (0.4 + Math.random() * 0.7).toFixed(2),
  min: (0.1 + Math.random() * 0.2).toFixed(2),
  max: (0.5 + Math.random() * 0.5).toFixed(2),
  delay: (Math.random() * 0.5).toFixed(2),
}));

export default function AudioVisualizer({ isSafe }) {
  const bars = useMemo(() => BARS, []);

  return (
    <div className="flex flex-col items-center w-full">
      {/* Label */}
      <div className="flex items-center gap-2 mb-4">
        <span className={`w-2 h-2 rounded-full animate-blink ${isSafe ? 'bg-emerald-400' : 'bg-red-500'}`} />
        <span className={`text-xs font-semibold uppercase tracking-widest ${isSafe ? 'text-emerald-400' : 'text-red-400'}`}>
          {isSafe ? 'Live Audio Stream' : '⚠ Synthetic Voice Detected'}
        </span>
      </div>

      {/* Visualizer bars */}
      <div
        className="relative flex items-end justify-center gap-[3px] h-20 w-full px-2"
        style={{ perspective: '200px' }}
      >
        {bars.map((bar) => (
          <div
            key={bar.id}
            className={`viz-bar ${isSafe ? 'viz-bar-safe' : 'viz-bar-danger'}`}
            style={{
              '--dur': `${isSafe ? bar.dur : (parseFloat(bar.dur) * 0.4).toFixed(2)}s`,
              '--min': bar.min,
              '--max': bar.max,
              animationDelay: `${bar.delay}s`,
              height: '64px',
            }}
          />
        ))}

        {/* Reflection/gradient overlay */}
        <div
          className="absolute bottom-0 left-0 right-0 h-8 pointer-events-none"
          style={{
            background: isSafe
              ? 'linear-gradient(to top, rgba(2,6,23,0.8), transparent)'
              : 'linear-gradient(to top, rgba(2,6,23,0.8), transparent)',
          }}
        />
      </div>

      {/* Frequency scale */}
      <div className="flex justify-between w-full px-1 mt-2">
        {['20Hz', '200Hz', '2kHz', '20kHz'].map((f) => (
          <span key={f} className="text-[9px] text-slate-600 font-mono">{f}</span>
        ))}
      </div>
    </div>
  );
}
