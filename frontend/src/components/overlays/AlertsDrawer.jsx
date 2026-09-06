import { useState, useEffect } from 'react';
import { X, ShieldAlert, Eye, Trash2, AlertTriangle, Clock, Phone, ChevronRight } from 'lucide-react';

const SEVERITY_STYLES = {
  CRITICAL: {
    badge: 'bg-red-500/15 text-red-400 border-red-500/30',
    dot:   'bg-red-500',
    border: 'border-red-500/20',
    glow:  'hover:border-red-500/40',
    icon:  'text-red-400',
  },
  HIGH: {
    badge: 'bg-amber-400/15 text-amber-400 border-amber-400/30',
    dot:   'bg-amber-400',
    border: 'border-amber-400/20',
    glow:  'hover:border-amber-400/40',
    icon:  'text-amber-400',
  },
};

export default function AlertsDrawer({ isOpen, onClose, alerts, setAlerts }) {
  const [visible, setVisible] = useState(false);
  const [dismissing, setDismissing] = useState(null);

  /* Drive enter/exit animation */
  useEffect(() => {
    if (isOpen) {
      requestAnimationFrame(() => setVisible(true));
    } else {
      setVisible(false);
    }
  }, [isOpen]);

  if (!isOpen && !visible) return null;

  const handleDismiss = (id) => {
    setDismissing(id);
    setTimeout(() => {
      setAlerts((prev) => prev.filter((a) => a.id !== id));
      setDismissing(null);
    }, 300);
  };

  const handleReview = (id) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, reviewed: true } : a))
    );
  };

  const unread = alerts.filter((a) => !a.reviewed).length;

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex justify-end"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      {/* Dim overlay */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm transition-opacity duration-300"
        style={{ opacity: visible ? 1 : 0 }}
      />

      {/* Drawer panel */}
      <div
        className="theme-panel relative w-full max-w-md h-full flex flex-col shadow-2xl transition-transform duration-300 ease-out"
        style={{
          transform: visible ? 'translateX(0)' : 'translateX(100%)',
          background: '#ffffff',
          borderLeft: '1px solid #dce6ee',
        }}
      >
        {/* ── Header ── */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800/60 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-red-500/15 border border-red-500/30 flex items-center justify-center">
              <ShieldAlert className="w-4 h-4 text-red-400" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">Threat Notifications</h2>
              <p className="text-[11px] text-slate-500 mt-0.5">
                {unread > 0 ? (
                  <span className="text-red-400 font-semibold">{unread} unreviewed</span>
                ) : (
                  'All reviewed'
                )}{' '}
                · {alerts.length} total
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-slate-800/60 hover:bg-slate-700/60 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* ── Alert list ── */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
          {alerts.length === 0 && (
            <div className="flex flex-col items-center justify-center h-48 text-center gap-3">
              <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                <ShieldAlert className="w-6 h-6 text-emerald-400" />
              </div>
              <p className="text-sm font-medium text-slate-400">All clear — no active threats</p>
              <p className="text-xs text-slate-600">New threats will appear here in real-time.</p>
            </div>
          )}

          {alerts.map((alert) => {
            const s = SEVERITY_STYLES[alert.severity];
            const isDismissing = dismissing === alert.id;
            return (
              <div
                key={alert.id}
                className={`glass-card p-4 border transition-all duration-300 ${s.border} ${s.glow}
                  ${alert.reviewed ? 'opacity-60' : ''}
                  ${isDismissing ? 'opacity-0 scale-95 translate-x-4' : 'opacity-100 scale-100 translate-x-0'}
                `}
              >
                {/* Alert header */}
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full flex-shrink-0 ${s.dot} ${!alert.reviewed ? 'animate-pulse' : ''}`} />
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider border ${s.badge}`}>
                      {alert.severity}
                    </span>
                    {alert.reviewed && (
                      <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold text-slate-500 bg-slate-800/60 border border-slate-700">
                        Reviewed
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] font-mono text-slate-600 flex-shrink-0">{alert.incidentId}</span>
                </div>

                <h3 className="text-sm font-semibold text-white mb-1.5 leading-snug">
                  {alert.title}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-3">
                  {alert.description}
                </p>

                {/* Meta row */}
                <div className="flex items-center gap-3 mb-3 text-[11px] text-slate-500">
                  <span className="flex items-center gap-1"><Phone className="w-3 h-3" />{alert.caller}</span>
                  <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{alert.timestamp}</span>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <button
                    onClick={() => handleReview(alert.id)}
                    disabled={alert.reviewed}
                    className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-semibold transition-all
                      ${alert.reviewed
                        ? 'bg-slate-800/40 text-slate-600 cursor-not-allowed'
                        : 'bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white border border-slate-700/60 hover:border-slate-600'
                      }`}
                  >
                    <Eye className="w-3.5 h-3.5" />
                    {alert.reviewed ? 'Reviewed' : 'Review'}
                    {!alert.reviewed && <ChevronRight className="w-3 h-3 ml-auto" />}
                  </button>
                  <button
                    onClick={() => handleDismiss(alert.id)}
                    className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold
                      bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 hover:border-red-500/40 transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Dismiss
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Footer ── */}
        <div className="px-4 py-3 border-t border-slate-800/60 flex-shrink-0">
          <button
            onClick={() => setAlerts([])}
            className="w-full py-2 rounded-lg text-xs font-semibold text-slate-500 hover:text-slate-300
              border border-slate-800 hover:border-slate-700 hover:bg-slate-800/40 transition-all flex items-center justify-center gap-2"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Dismiss All Alerts
          </button>
          <p className="text-center text-[10px] text-slate-700 mt-2 flex items-center justify-center gap-1">
            <AlertTriangle className="w-3 h-3" />
            All alerts are written to the immutable audit ledger.
          </p>
        </div>
      </div>
    </div>
  );
}
