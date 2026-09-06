import { useEffect, useState } from 'react';
import { useCallEngine } from './hooks/useCallEngine';
import AuthPage from './components/AuthPage';
import Sidebar from './components/Sidebar';
import DashboardView from './views/DashboardView';
import AuditLogsView from './views/AuditLogsView';
import AlertsDrawer from './components/overlays/AlertsDrawer';
import AccessControlModal from './components/overlays/AccessControlModal';
import SettingsModal from './components/overlays/SettingsModal';

export default function App() {
  const engine = useCallEngine();
  const [account, setAccount] = useState(null);
  const [activePanel, setActivePanel] = useState(null); // 'alerts' | 'access' | 'settings' | null
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    setAlerts(engine.auditLogs
      .filter((log) => log.decision === 'BLOCKED')
      .map((log) => ({
        id: log.id,
        severity: 'CRITICAL',
        title: 'High-risk call detected',
        description: `Risk event recorded for call ${log.id}.`,
        caller: log.number,
        timestamp: log.timestamp,
        incidentId: log.id,
        reviewed: false,
      })));
  }, [engine.auditLogs]);

  const openPanel  = (panel) => setActivePanel(panel);
  const closePanel = ()      => setActivePanel(null);

  const unreadAlerts = alerts.filter(a => !a.reviewed).length;

  if (!account) return <AuthPage onAuthenticated={setAccount} />;

  return (
    <div className="flex h-screen bg-[#f4f7fb] overflow-hidden font-sans">
      {/* ── Sidebar ── */}
      <Sidebar
        currentView={engine.currentView}
        setCurrentView={engine.setCurrentView}
        isSimulating={engine.isSimulating}
        backendStatus={engine.backendStatus}
        onOpenPanel={openPanel}
        unreadAlerts={unreadAlerts}
        account={account}
        onLogout={() => setAccount(null)}
      />

      {/* ── Main Content ── */}
      <main className="flex-1 flex flex-col overflow-hidden relative">
        {/* Subtle workspace texture */}
        <div
          className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: `
              radial-gradient(circle at 15% 20%, rgba(69,180,151,0.35), transparent 28%),
              linear-gradient(135deg, rgba(255,255,255,0.8), rgba(225,235,242,0.8))
            `,
            backgroundSize: 'auto, 100% 100%',
          }}
        />

        {/* View Router */}
        <div className="relative z-10 flex-1 flex flex-col overflow-hidden">
          {engine.currentView === 'dashboard' && (
            <DashboardView
              callState={engine.callState}
              riskScore={engine.riskScore}
              metrics={engine.metrics}
              callerInfo={engine.callerInfo}
              transaction={engine.transaction}
              isSimulating={engine.isSimulating}
              liveStatus={engine.liveStatus}
              liveError={engine.liveError}
              startLiveCall={engine.startLiveCall}
              stopLiveCall={engine.stopLiveCall}
              uploadRecording={engine.uploadRecording}
              auditLogs={engine.auditLogs}
            />
          )}
          {engine.currentView === 'audit' && (
            <AuditLogsView auditLogs={engine.auditLogs} />
          )}
        </div>
      </main>

      {/* ── Overlays (rendered at root so they're never clipped) ── */}
      <AlertsDrawer
        isOpen={activePanel === 'alerts'}
        onClose={closePanel}
        alerts={alerts}
        setAlerts={setAlerts}
      />
      <AccessControlModal
        isOpen={activePanel === 'access'}
        onClose={closePanel}
      />
      <SettingsModal
        isOpen={activePanel === 'settings'}
        onClose={closePanel}
      />
    </div>
  );
}
