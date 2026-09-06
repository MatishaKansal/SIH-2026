import { useState, useEffect } from 'react';
import {
  X, Settings, ShieldCheck, Shield, Save, RotateCcw,
  HardDrive, Mic, Bell, Cpu, Info,
} from 'lucide-react';

const DEFAULTS = {
  riskThreshold: 70,
  zeroAudioStorage: true,
  autoBlock: true,
  alertSensitivity: 'HIGH',
  retentionDays: 90,
  realtimeAnalysis: true,
  notifyOnBlock: true,
};

function Toggle({ checked, onChange, id }) {
  return (
    <button
      id={id}
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex w-11 h-6 rounded-full transition-colors duration-300 focus:outline-none flex-shrink-0
        ${checked ? 'bg-emerald-500' : 'bg-slate-700'}`}
    >
      <span
        className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow transition-transform duration-300
          ${checked ? 'translate-x-5' : 'translate-x-0'}`}
      />
    </button>
  );
}

function Section({ icon: Icon, title, children }) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <Icon className="w-3.5 h-3.5 text-slate-500" />
        <span className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold">{title}</span>
      </div>
      <div className="glass-card-darker p-4 space-y-4">
        {children}
      </div>
    </div>
  );
}

function SettingRow({ label, description, children }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-slate-300 leading-tight">{label}</p>
        {description && <p className="text-[11px] text-slate-600 mt-0.5 leading-relaxed">{description}</p>}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function ThresholdSlider({ value, onChange }) {
  const getColor = (v) => {
    if (v < 40) return '#ef4444';
    if (v < 60) return '#f97316';
    if (v < 80) return '#fbbf24';
    return '#34d399';
  };
  const color = getColor(value);

  return (
    <div className="space-y-3">
      {/* Value display */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="text-2xl font-extrabold tabular-nums transition-colors duration-300"
            style={{ color }}
          >
            {value}%
          </span>
          <span className="text-[11px] text-slate-600">blocking threshold</span>
        </div>
        <span
          className="text-[10px] px-2.5 py-1 rounded-full font-bold uppercase tracking-wider border transition-all duration-300"
          style={{
            color,
            backgroundColor: `${color}18`,
            borderColor: `${color}44`,
          }}
        >
          {value >= 80 ? 'Conservative' : value >= 60 ? 'Balanced' : value >= 40 ? 'Aggressive' : 'Max Sensitivity'}
        </span>
      </div>

      {/* Slider */}
      <div className="relative py-2">
        <div className="relative h-2 rounded-full bg-slate-800 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-150"
            style={{ width: `${value}%`, background: `linear-gradient(to right, #ef4444, ${color})` }}
          />
        </div>
        <input
          id="risk-threshold-slider"
          type="range"
          min={10}
          max={99}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full opacity-0 cursor-pointer h-full"
        />
        {/* Custom thumb */}
        <div
          className="absolute top-1/2 w-5 h-5 rounded-full border-2 border-slate-950 shadow-lg transition-all duration-150 pointer-events-none -translate-y-1/2"
          style={{
            left: `calc(${((value - 10) / 89) * 100}% - 10px)`,
            backgroundColor: color,
            boxShadow: `0 0 12px ${color}80`,
          }}
        />
      </div>

      {/* Scale labels */}
      <div className="flex justify-between text-[10px] text-slate-600 font-mono px-0.5">
        <span>10% <span className="text-red-500">(Max alert)</span></span>
        <span>55%</span>
        <span>99% <span className="text-emerald-500">(Conservative)</span></span>
      </div>

      {/* Info blurb */}
      <div className="flex items-start gap-2 p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
        <Info className="w-3.5 h-3.5 text-slate-600 flex-shrink-0 mt-0.5" />
        <p className="text-[11px] text-slate-600 leading-relaxed">
          Calls with a risk score at or above this threshold will be <span style={{ color }} className="font-semibold">automatically blocked</span>. 
          Lower values increase sensitivity but may trigger false positives.
        </p>
      </div>
    </div>
  );
}

export default function SettingsModal({ isOpen, onClose }) {
  const [settings, setSettings] = useState(DEFAULTS);
  const [saved, setSaved] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (isOpen) {
      requestAnimationFrame(() => setVisible(true));
      document.body.style.overflow = 'hidden';
    } else {
      setVisible(false);
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  if (!isOpen && !visible) return null;

  const set = (key) => (val) => setSettings((prev) => ({ ...prev, [key]: val }));

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => setSettings(DEFAULTS);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md transition-opacity duration-300"
        style={{ opacity: visible ? 1 : 0 }}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className="theme-panel relative w-full max-w-lg flex flex-col max-h-[88vh] transition-all duration-300 rounded-2xl overflow-hidden"
        style={{
          transform: visible ? 'scale(1) translateY(0)' : 'scale(0.95) translateY(16px)',
          opacity: visible ? 1 : 0,
          background: '#ffffff',
          border: '1px solid #dce6ee',
          boxShadow: '0 25px 60px rgba(28,48,72,0.18)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800/60 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-amber-400/10 border border-amber-400/30 flex items-center justify-center">
              <Settings className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">Engine Settings</h2>
              <p className="text-[11px] text-slate-500 mt-0.5">AI detection & privacy configuration</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-slate-800/60 hover:bg-slate-700/60 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-5 py-5 space-y-6">

          {/* ── Risk Engine ── */}
          <Section icon={Shield} title="Risk Engine">
            <div>
              <p className="text-xs font-medium text-slate-300 mb-3">Risk Blocking Threshold</p>
              <ThresholdSlider value={settings.riskThreshold} onChange={set('riskThreshold')} />
            </div>

            <div className="border-t border-slate-700/40" />

            <SettingRow
              label="Auto-Block Transactions"
              description="Automatically suspend all pending transactions when risk score exceeds threshold."
            >
              <Toggle id="toggle-autoblock" checked={settings.autoBlock} onChange={set('autoBlock')} />
            </SettingRow>

            <SettingRow
              label="Alert Sensitivity"
              description="Determines the minimum confidence level required to generate a threat alert."
            >
              <select
                value={settings.alertSensitivity}
                onChange={(e) => set('alertSensitivity')(e.target.value)}
                className="bg-slate-800 border border-slate-700 text-xs text-slate-300 rounded-lg px-2.5 py-1.5
                  focus:outline-none focus:border-emerald-500/50 transition-all cursor-pointer"
              >
                <option value="LOW">Low</option>
                <option value="MEDIUM">Medium</option>
                <option value="HIGH">High</option>
                <option value="PARANOID">Paranoid</option>
              </select>
            </SettingRow>
          </Section>

          {/* ── Privacy ── */}
          <Section icon={HardDrive} title="Privacy & Storage">
            <SettingRow
              label="Zero Audio Storage"
              description="Process audio entirely in-memory. No raw voice data is persisted to disk or cloud."
            >
              <Toggle id="toggle-zero-storage" checked={settings.zeroAudioStorage} onChange={set('zeroAudioStorage')} />
            </SettingRow>

            <div className={`flex items-start gap-2.5 p-3 rounded-xl border transition-all duration-300
              ${settings.zeroAudioStorage
                ? 'bg-emerald-500/5 border-emerald-500/20'
                : 'bg-amber-400/5 border-amber-400/20'}`}
            >
              <ShieldCheck className={`w-4 h-4 flex-shrink-0 mt-0.5 transition-colors ${settings.zeroAudioStorage ? 'text-emerald-400' : 'text-amber-400'}`} />
              <p className="text-[11px] text-slate-400 leading-relaxed">
                {settings.zeroAudioStorage
                  ? <><span className="text-emerald-400 font-semibold">Privacy Mode ON</span> — Audio is ephemeral. Only voiceprint embeddings are stored (FIPS 140-3 encrypted).</>
                  : <><span className="text-amber-400 font-semibold">Privacy Mode OFF</span> — Audio samples are retained for <span className="font-semibold">{settings.retentionDays} days</span> for model retraining.</>
                }
              </p>
            </div>

            {!settings.zeroAudioStorage && (
              <SettingRow label="Retention Period" description="How long raw audio samples are stored.">
                <select
                  value={settings.retentionDays}
                  onChange={(e) => set('retentionDays')(Number(e.target.value))}
                  className="bg-slate-800 border border-slate-700 text-xs text-slate-300 rounded-lg px-2.5 py-1.5
                    focus:outline-none focus:border-emerald-500/50 transition-all cursor-pointer"
                >
                  <option value={7}>7 days</option>
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                  <option value={365}>1 year</option>
                </select>
              </SettingRow>
            )}
          </Section>

          {/* ── Analysis ── */}
          <Section icon={Cpu} title="Analysis">
            <SettingRow
              label="Real-time Inference"
              description="Enable GPU-accelerated inference on every audio frame. Disable to save compute."
            >
              <Toggle id="toggle-realtime" checked={settings.realtimeAnalysis} onChange={set('realtimeAnalysis')} />
            </SettingRow>
          </Section>

          {/* ── Notifications ── */}
          <Section icon={Bell} title="Notifications">
            <SettingRow
              label="Notify on Block"
              description="Push a notification to the assigned SOC analyst whenever a call is blocked."
            >
              <Toggle id="toggle-notify" checked={settings.notifyOnBlock} onChange={set('notifyOnBlock')} />
            </SettingRow>
            <SettingRow label="Voice Enrollment Reminders" description="Alert when executives have pending voice enrollment.">
              <Toggle id="toggle-enroll-remind" checked={true} onChange={() => {}} />
            </SettingRow>
          </Section>
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-slate-800/60 flex items-center justify-between gap-3 flex-shrink-0">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold text-slate-500
              border border-slate-800 hover:border-slate-700 hover:text-slate-300 transition-all"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Reset Defaults
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400
                border border-slate-700 hover:border-slate-600 hover:text-white transition-all"
            >
              Cancel
            </button>
            <button
              id="btn-save-settings"
              onClick={handleSave}
              className={`flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-bold transition-all duration-300
                ${saved
                  ? 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/30'
                  : 'bg-amber-400/90 hover:bg-amber-400 text-slate-950 shadow-lg shadow-amber-400/20'
                }`}
            >
              <Save className="w-3.5 h-3.5" />
              {saved ? '✓ Saved!' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
