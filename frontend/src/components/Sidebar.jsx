import { Activity, FileText, Radio, Lock, Settings, Bell, LogOut } from 'lucide-react';

const navItems = [
  { id: 'dashboard', label: 'Live Monitor',  icon: Activity },
  { id: 'audit',     label: 'Audit Logs',    icon: FileText },
];

export default function Sidebar({ currentView, setCurrentView, isSimulating, backendStatus, onOpenPanel, unreadAlerts = 0, account, onLogout }) {
  const isBackendConnected = backendStatus === 'connected';

  return (
    <aside className="flex flex-col w-[270px] flex-shrink-0 bg-[#182a3e] border-r border-[#294159] h-full select-none">

      {/* ── Brand ── */}
      <div className="flex items-center gap-3 px-6 py-6 border-b border-white/10">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-[#2c4c68] border border-[#4d7892] overflow-hidden">
          <img src="/echoguard-logo.png" alt="EchoGuard Logo" className="w-full h-full object-cover" />
          {isSimulating && (
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
          )}
        </div>
        <div>
          <p className="text-[17px] font-bold text-white tracking-wide leading-tight mt-1">EchoGuard</p>
          <p className="text-[10px] text-[#9eb3c6] mt-0.5">Call Fraud Prevention</p>
        </div>
      </div>

      {/* ── Status indicator ── */}
      <div className="mx-4 mt-5 mb-4 px-4 py-4 rounded-2xl border border-white/10 bg-white/[0.06]">
        <div className="flex items-center gap-2">
          <span className="relative flex w-2 h-2">
            {isSimulating && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>}
            <span className={`relative inline-flex rounded-full w-2 h-2 ${isSimulating ? 'bg-red-500' : 'bg-emerald-500'}`}></span>
          </span>
          <span className={`text-xs font-semibold ${isSimulating ? 'text-red-300' : 'text-white'}`}>
            {isSimulating ? 'Threat Detected' : 'All Systems Nominal'}
          </span>
        </div>
        <div className="mt-2 flex items-center gap-1.5">
          <Radio className="w-3 h-3 text-slate-500" />
          <span className="text-[10px] text-[#9eb3c6] font-mono">
            {isBackendConnected ? '4/4 Services Online' : 'Services Offline'}
          </span>
        </div>
      </div>

      {/* ── Navigation ── */}
      <nav className="flex-1 px-3 mt-2 space-y-0.5">
        <p className="px-3 pt-2 pb-3 text-[10px] text-[#91a9bd] uppercase tracking-widest">Monitoring</p>
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setCurrentView(id)}
            className={`nav-item w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-left transition-all duration-200
              ${currentView === id ? 'nav-item-active' : 'text-[#b1c1cf]'}`}
          >
            <Icon className="w-4 h-4 flex-shrink-0" />
            {label}
          </button>
        ))}
      </nav>

      {/* ── Bottom Section ── */}
      <div className="border-t border-white/10 p-4 space-y-2">
        <button onClick={() => onOpenPanel('alerts')} className="nav-item w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs text-[#b1c1cf] hover:text-white">
          <Bell className="w-4 h-4" />
          Alerts
          {unreadAlerts > 0 && (
            <span className="ml-auto bg-red-500/80 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">{unreadAlerts}</span>
          )}
        </button>
        <button onClick={() => onOpenPanel('access')} className="nav-item w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs text-[#b1c1cf] hover:text-white">
          <Lock className="w-4 h-4" />
          Access Control
        </button>
        <button onClick={() => onOpenPanel('settings')} className="nav-item w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs text-[#b1c1cf] hover:text-white">
          <Settings className="w-4 h-4" />
          Settings
        </button>

        <div className="mt-3 pt-3 border-t border-white/10 flex items-center gap-2.5 px-1">
          <div className="w-9 h-9 rounded-full bg-[#344760] flex items-center justify-center text-[10px] font-bold text-white">{account?.name?.split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase()}</div>
          <div>
            <p className="text-xs font-medium text-white truncate max-w-[145px]">{account?.name}</p>
            <p className="text-[10px] text-[#9eb3c6] truncate max-w-[145px]">{account?.email}</p>
          </div>
          <button type="button" onClick={onLogout} title="Sign out" className="ml-auto text-[#9eb3c6] hover:text-white"><LogOut className="w-3.5 h-3.5" /></button>
        </div>
      </div>
    </aside>
  );
}
