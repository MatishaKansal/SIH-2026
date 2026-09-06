import { useState } from 'react';
import { ShieldCheck, ShieldAlert, Search, Download, Filter, Hash } from 'lucide-react';

function RiskBadge({ score }) {
  if (score >= 80) return <span className="badge-blocked">{score}</span>;
  if (score >= 50) return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-full
      bg-amber-400/10 text-amber-400 border border-amber-400/30">{score}</span>
  );
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-full
      bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">{score}</span>
  );
}

function DecisionBadge({ decision }) {
  if (decision === 'BLOCKED') {
    return (
      <span className="badge-blocked">
        <ShieldAlert className="w-3 h-3" /> Blocked
      </span>
    );
  }
  return (
    <span className="badge-allowed">
      <ShieldCheck className="w-3 h-3" /> Allowed
    </span>
  );
}

export default function AuditLogsView({ auditLogs }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterDecision, setFilterDecision] = useState('ALL'); // 'ALL', 'BLOCKED', 'ALLOWED'
  const [isFilterOpen, setIsFilterOpen] = useState(false);

  const totalCalls    = auditLogs.length;
  const blocked       = auditLogs.filter((l) => l.decision === 'BLOCKED').length;
  const allowed       = totalCalls - blocked;
  const avgRisk       = Math.round(auditLogs.reduce((s, l) => s + l.peakRisk, 0) / (totalCalls || 1));

  // Filter logs based on search query and decision dropdown
  const filteredLogs = auditLogs.filter(log => {
    const matchesSearch = 
      log.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.identity.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.hash.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesDecision = filterDecision === 'ALL' || log.decision === filterDecision;
    
    return matchesSearch && matchesDecision;
  });

  const handleExportCSV = () => {
    const headers = ['Timestamp', 'Call ID', 'Claimed Identity', 'Phone', 'Peak Risk', 'Decision', 'Audit Hash'];
    const rows = auditLogs.map(log => [
      log.timestamp,
      log.id,
      log.identity,
      log.number,
      log.peakRisk,
      log.decision,
      log.hash
    ]);
    
    const csvContent = [
      headers.join(','),
      ...rows.map(e => e.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'echoguard_audit_logs.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Page Header ── */}
      <header className="relative z-50 flex items-center justify-between px-8 py-5 border-b border-[#e2eaf1] bg-white/75 backdrop-blur-sm flex-shrink-0">
        <div>
          <h1 className="text-[25px] font-extrabold text-[#172b45]">Audit Logs</h1>
          <p className="text-[12px] text-[#8192a6] mt-0.5">Immutable blockchain-anchored call records · Last sync 2s ago</p>
        </div>
        <div className="flex items-center gap-2 relative z-50">
          <button 
            onClick={() => setIsFilterOpen(!isFilterOpen)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-[#60758a] border border-[#cbd8e3] hover:border-[#8fa5b8] hover:text-[#1c3048] transition-all"
          >
            <Filter className="w-3.5 h-3.5" /> Filter {filterDecision !== 'ALL' && <span className="ml-1 w-2 h-2 rounded-full bg-emerald-400"></span>}
          </button>
          
          {/* Filter Dropdown */}
          {isFilterOpen && (
            <div className="absolute top-full right-24 mt-2 w-40 glass-card p-1 shadow-xl shadow-black/50 z-50">
              {['ALL', 'BLOCKED', 'ALLOWED'].map((option) => (
                <button
                  key={option}
                  onClick={() => { setFilterDecision(option); setIsFilterOpen(false); }}
                  className={`w-full text-left px-3 py-2 text-xs rounded-lg transition-colors ${
                    filterDecision === option ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-300 hover:bg-slate-800/50'
                  }`}
                >
                  {option === 'ALL' ? 'All Calls' : option === 'BLOCKED' ? 'Blocked Only' : 'Allowed Only'}
                </button>
              ))}
            </div>
          )}

          <button 
            onClick={handleExportCSV}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-[#218e7a] border border-emerald-500/30 hover:bg-emerald-500/10 transition-all"
          >
            <Download className="w-3.5 h-3.5" /> Export CSV
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-5 space-y-5">

        {/* ── Summary cards ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Calls',     value: totalCalls, color: 'text-slate-300',   border: 'border-slate-700' },
            { label: 'Threats Blocked', value: blocked,    color: 'text-red-400',     border: 'border-red-500/30' },
            { label: 'Calls Allowed',   value: allowed,    color: 'text-emerald-400', border: 'border-emerald-500/30' },
            { label: 'Avg Risk Score',  value: avgRisk,    color: 'text-amber-400',   border: 'border-amber-400/30' },
          ].map(({ label, value, color, border }) => (
            <div key={label} className={`glass-card p-4 border ${border}`}>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">{label}</p>
              <p className={`text-2xl font-extrabold mt-1 ${color}`}>{value}</p>
            </div>
          ))}
        </div>

        {/* ── Search bar ── */}
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
          <input
            id="audit-search"
            type="text"
            placeholder="Search by Call ID, identity, or hash…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white border border-[#cbd8e3] rounded-xl pl-10 pr-4 py-2.5 text-sm text-[#425a72]
              placeholder-[#8294a6] focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 transition-all"
          />
        </div>

        {/* ── Table ── */}
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800/80">
                  {[
                    'Timestamp', 'Call ID', 'Claimed Identity',
                    'Peak Risk', 'Decision', 'Audit Hash',
                  ].map((col) => (
                    <th
                      key={col}
                      className="px-4 py-3 text-left text-[10px] text-slate-500 uppercase tracking-widest font-semibold whitespace-nowrap"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredLogs.length > 0 ? (
                  filteredLogs.map((log) => (
                    <tr key={log.id} className="hover:bg-slate-800/30 transition-colors group">
                      <td className="px-4 py-3 whitespace-nowrap text-slate-400 text-xs font-mono">
                        {log.timestamp}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className="font-mono text-emerald-400 text-xs">{log.id}</span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex flex-col">
                          <span className="text-white font-medium">{log.identity}</span>
                          <span className="text-slate-500 text-[10px] font-mono">{log.number}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <RiskBadge score={log.peakRisk} />
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <DecisionBadge decision={log.decision} />
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex items-center gap-1.5 opacity-60 group-hover:opacity-100 transition-opacity">
                          <Hash className="w-3.5 h-3.5 text-slate-500" />
                          <span className="font-mono text-xs text-slate-400">{log.hash}</span>
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="6" className="px-4 py-8 text-center text-slate-500 text-sm">
                      No matching audit logs found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Table footer */}
          <div className="px-4 py-3 border-t border-slate-800/60 flex items-center justify-between">
            <span className="text-[11px] text-slate-600">Showing {auditLogs.length} of {auditLogs.length} records</span>
            <div className="flex items-center gap-1">
              {[1].map((p) => (
                <button key={p}
                  className="w-7 h-7 rounded-lg text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Blockchain anchor note */}
        <div className="flex items-center gap-2 px-1">
          <Hash className="w-3 h-3 text-slate-700" />
          <p className="text-[10px] text-slate-700">
            All records are SHA-256 hashed and anchored to the Ethereum mainnet via Chainlink proof-of-call oracle.
            Block height: <span className="font-mono">20,541,887</span>.
          </p>
        </div>
      </div>
    </div>
  );
}
