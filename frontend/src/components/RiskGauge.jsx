import { useMemo } from 'react';

const SIZE = 160;
const STROKE = 12;
const RADIUS = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
/* Arc is 240° (from 150° to 390°) */
const ARC_FRACTION = 240 / 360;
const ARC_LENGTH = CIRCUMFERENCE * ARC_FRACTION;

function getRiskColor(score) {
  if (score < 30)  return { stroke: '#34d399', text: 'text-emerald-400', label: 'LOW RISK',      labelColor: 'text-emerald-400' };
  if (score < 60)  return { stroke: '#fbbf24', text: 'text-amber-400',   label: 'MODERATE',      labelColor: 'text-amber-400' };
  if (score < 80)  return { stroke: '#f97316', text: 'text-orange-400',  label: 'HIGH RISK',     labelColor: 'text-orange-400' };
  return             { stroke: '#ef4444', text: 'text-red-500',   label: 'CRITICAL',      labelColor: 'text-red-500' };
}

export default function RiskGauge({ score }) {
  const { stroke, text, label, labelColor } = useMemo(() => getRiskColor(score), [score]);

  /* dashoffset: 0 = full arc, ARC_LENGTH = empty */
  const offset = ARC_LENGTH - (score / 100) * ARC_LENGTH;

  /* Rotation so arc starts at 150° */
  const rotation = 150;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: SIZE, height: SIZE }}>
        <svg width={SIZE} height={SIZE} className="overflow-visible">
          <g transform={`rotate(${rotation}, ${SIZE / 2}, ${SIZE / 2})`}>
            {/* Track */}
            <circle
              className="risk-gauge-track"
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={RADIUS}
              strokeDasharray={`${ARC_LENGTH} ${CIRCUMFERENCE}`}
              strokeDashoffset={0}
            />
            {/* Fill */}
            <circle
              className="risk-gauge-fill"
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={RADIUS}
              stroke={stroke}
              strokeDasharray={`${ARC_LENGTH} ${CIRCUMFERENCE}`}
              strokeDashoffset={offset}
              style={{
                filter: `drop-shadow(0 0 8px ${stroke})`,
                transition: 'stroke-dashoffset 1s cubic-bezier(0.34,1.56,0.64,1), stroke 0.6s ease',
              }}
            />
          </g>
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center mt-2">
          <span
            className={`text-4xl font-extrabold tabular-nums leading-none ${text}`}
            style={{ transition: 'color 0.5s ease' }}
          >
            {score}
          </span>
          <span className="text-[10px] text-slate-500 uppercase tracking-widest mt-1">Risk Score</span>
        </div>
      </div>

      {/* Label badge */}
      <div
        className={`mt-2 px-4 py-1 rounded-full text-xs font-bold uppercase tracking-widest border ${labelColor}`}
        style={{
          background: `${stroke}18`,
          borderColor: `${stroke}55`,
          transition: 'all 0.5s ease',
        }}
      >
        {label}
      </div>

      {/* Tick markers */}
      <div className="flex justify-between w-full max-w-[160px] mt-1 px-1">
        <span className="text-[9px] text-slate-600 font-mono">0</span>
        <span className="text-[9px] text-slate-600 font-mono">50</span>
        <span className="text-[9px] text-slate-600 font-mono">100</span>
      </div>
    </div>
  );
}
