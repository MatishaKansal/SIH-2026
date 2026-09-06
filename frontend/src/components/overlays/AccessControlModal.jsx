import { useEffect, useState } from 'react';
import { X, Lock, CheckCircle2, AlertCircle, ChevronDown } from 'lucide-react';

const STATUS_CONFIG = {
  ENROLLED: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30', label: 'Enrolled' },
  REVOKED: { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', label: 'Inactive' },
};

function mapProfile(profile) {
  return {
    id: profile.id,
    name: profile.identity_name,
    role: profile.identity_type,
    initials: profile.identity_name.split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase(),
    reference: profile.external_reference || 'Unavailable',
    voiceprint: profile.voiceprint_reference ? 'Configured' : 'Not configured',
    status: profile.status === 'ACTIVE' ? 'ENROLLED' : 'REVOKED',
    enrolledAt: profile.enrolled_at ? new Date(profile.enrolled_at).toLocaleDateString() : 'Unavailable',
    color: profile.status === 'ACTIVE' ? 'from-emerald-600 to-teal-700' : 'from-slate-600 to-slate-700',
  };
}

function ProfileRow({ profile }) {
  const [expanded, setExpanded] = useState(false);
  const { icon: StatusIcon, color, bg, label } = STATUS_CONFIG[profile.status];

  return (
    <div className="glass-card-darker border border-slate-700/60">
      <button className="w-full flex items-center gap-3.5 p-4 text-left" onClick={() => setExpanded((open) => !open)}>
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold text-white flex-shrink-0 bg-gradient-to-br ${profile.color}`}>
          {profile.initials}
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-semibold text-white truncate block">{profile.name}</span>
          <p className="text-[11px] text-slate-500 mt-0.5 truncate">{profile.role}</p>
        </div>
        <span className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-semibold border flex-shrink-0 ${bg} ${color}`}>
          <StatusIcon className="w-3 h-3" />{label}
        </span>
        <ChevronDown className={`w-4 h-4 text-slate-600 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <div className="px-4 pb-4 border-t border-slate-800/40 pt-3 grid grid-cols-2 gap-3">
          <div><p className="text-[10px] text-slate-600 uppercase tracking-wider">Reference</p><p className="text-xs text-slate-300 font-mono mt-0.5">{profile.reference}</p></div>
          <div><p className="text-[10px] text-slate-600 uppercase tracking-wider">Enrolled</p><p className="text-xs text-slate-300 font-mono mt-0.5">{profile.enrolledAt}</p></div>
          <div><p className="text-[10px] text-slate-600 uppercase tracking-wider">Voiceprint</p><p className="text-xs text-slate-300 font-mono mt-0.5">{profile.voiceprint}</p></div>
        </div>
      )}
    </div>
  );
}

export default function AccessControlModal({ isOpen, onClose }) {
  const [profiles, setProfiles] = useState([]);
  const [visible, setVisible] = useState(false);
  const [status, setStatus] = useState('loading');

  useEffect(() => {
    if (!isOpen) {
      setVisible(false);
      document.body.style.overflow = '';
      return undefined;
    }

    requestAnimationFrame(() => setVisible(true));
    document.body.style.overflow = 'hidden';
    let active = true;

    fetch('/api/v1/speaker-profiles')
      .then((response) => {
        if (!response.ok) throw new Error('Unable to load speaker profiles');
        return response.json();
      })
      .then((rows) => {
        if (!active) return;
        setProfiles(rows.map(mapProfile));
        setStatus('ready');
      })
      .catch(() => active && setStatus('error'));

    return () => {
      active = false;
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen && !visible) return null;
  const enrolled = profiles.filter((profile) => profile.status === 'ENROLLED').length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-md" onClick={onClose} />
      <div className="theme-panel relative w-full max-w-lg flex flex-col max-h-[85vh] rounded-2xl overflow-hidden bg-white border border-slate-700/70 shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center"><Lock className="w-4 h-4 text-emerald-400" /></div>
            <div><h2 className="text-sm font-bold text-white">Access Control</h2><p className="text-[11px] text-slate-500 mt-0.5">Speaker profiles · <span className="text-emerald-400 font-semibold">{enrolled} active</span></p></div>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-lg bg-slate-800/60 flex items-center justify-center text-slate-400 hover:text-white"><X className="w-4 h-4" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          {status === 'loading' && <p className="text-center text-sm text-slate-500 py-12">Loading speaker profiles...</p>}
          {status === 'error' && <p className="text-center text-sm text-red-400 py-12">Unable to load speaker profiles.</p>}
          {status === 'ready' && profiles.length === 0 && <p className="text-center text-sm text-slate-500 py-12">No speaker profiles enrolled.</p>}
          {profiles.map((profile) => <ProfileRow key={profile.id} profile={profile} />)}
        </div>
        <div className="px-4 py-3.5 border-t border-slate-800/60 flex justify-end"><button onClick={onClose} className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 border border-slate-700 hover:text-white">Close</button></div>
      </div>
    </div>
  );
}
