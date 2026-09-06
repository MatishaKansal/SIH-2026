import { Lock, ShieldAlert, ShieldCheck, Fingerprint, ArrowRight, AlertTriangle } from 'lucide-react';

export default function TransactionPanel({ transaction, isSafe }) {
  const isBlocked = !isSafe;

  return (
    <div className="glass-card p-5 relative overflow-hidden">
      {/* Section header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={`w-1 h-5 rounded-full ${isBlocked ? 'bg-red-500' : 'bg-amber-400'}`} />
          <h3 className="text-sm font-semibold text-white">Transaction Interception</h3>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase
          ${isBlocked ? 'bg-red-500/15 text-red-400 border border-red-500/30'
                      : 'bg-amber-400/10 text-amber-400 border border-amber-400/30'}`}>
          {isBlocked ? 'BLOCKED' : 'PENDING AUTH'}
        </span>
      </div>

      {/* ── Threat Banner (HIGH_RISK only) ── */}
      {isBlocked && (
        <div className="animate-slide-down mb-4 flex items-start gap-3 p-3 rounded-xl border border-red-500/40 bg-red-500/10 glow-red">
          <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5 animate-pulse" />
          <div>
            <p className="text-sm font-bold text-red-400 leading-tight">Transaction Blocked</p>
            <p className="text-xs text-red-300/70 mt-0.5">Synthetic voice detected — authorization suspended by EchoGuard AI.</p>
          </div>
        </div>
      )}

      {/* Transaction details card */}
      <div className="glass-card-darker p-4 mb-4 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Type</span>
          <span className="text-xs font-medium text-white">{transaction.type}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Amount</span>
          <span className={`text-xl font-extrabold tracking-tight ${isBlocked ? 'text-red-400 line-through decoration-red-600' : 'text-white'}`}>
            {transaction.amount}
          </span>
        </div>
        <div className="border-t border-slate-700/60 my-2" />
        <div className="flex items-start justify-between gap-4">
          <span className="text-xs text-slate-500 flex-shrink-0">Destination</span>
          <span className="text-xs font-mono text-slate-300 text-right">{transaction.destination}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Reference</span>
          <span className="text-xs font-mono text-slate-400">{transaction.reference}</span>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3">
        {/* Authorize button — locked when HIGH_RISK */}
        <button
          id="btn-authorize"
          disabled={isBlocked}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300
            ${isBlocked
              ? 'bg-slate-800 text-slate-600 cursor-not-allowed border border-slate-700'
              : 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-lg shadow-emerald-500/20 hover:shadow-emerald-400/30'
            }`}
        >
          {isBlocked
            ? <><Lock className="w-4 h-4" /> Authorize</>
            : <><ShieldCheck className="w-4 h-4" /> Authorize</>
          }
        </button>

        {/* Trigger Verification */}
        <button
          id="btn-verify"
          className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold
            border border-amber-400/40 text-amber-400 hover:bg-amber-400/10 hover:border-amber-400/70
            transition-all duration-200"
        >
          {isBlocked
            ? <><ShieldAlert className="w-4 h-4" /> Escalate</>
            : <><Fingerprint className="w-4 h-4" /> Verify Voice</>
          }
        </button>
      </div>

      {/* Audit trail hint */}
      <p className="mt-3 text-[10px] text-slate-600 flex items-center gap-1">
        <ArrowRight className="w-3 h-3" />
        All actions are cryptographically signed and logged to the immutable audit ledger.
      </p>
    </div>
  );
}
