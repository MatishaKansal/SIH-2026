import { useRef, useState } from 'react';
import { Mic, Square, Upload, LoaderCircle } from 'lucide-react';

export default function CallControls({ liveStatus, liveError, startLiveCall, stopLiveCall, uploadRecording }) {
  const fileRef = useRef(null);
  const [identity, setIdentity] = useState('');
  const [phone, setPhone] = useState('');
  const [intakeError, setIntakeError] = useState('');

  const metadata = { claimed_identity: identity.trim(), source_phone: phone.trim() };
  const isLive = liveStatus === 'live' || liveStatus === 'connecting';
  const isBusy = liveStatus === 'connecting' || liveStatus === 'uploading';

  function validateIntake() {
    if (!metadata.claimed_identity || !metadata.source_phone) {
      setIntakeError('Claimed identity and source phone are required.');
      return false;
    }
    setIntakeError('');
    return true;
  }

  function handleStart() {
    if (validateIntake()) startLiveCall(metadata);
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (file) {
      if (!validateIntake()) {
        event.target.value = '';
        return;
      }
      try {
        await uploadRecording(file, metadata);
      } catch {
        // The hook exposes the backend error beside the controls.
      }
    }
    event.target.value = '';
  }

  return (
    <div className="glass-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">Call Intake</h3>
        <span className="text-[10px] uppercase tracking-widest text-slate-500">{liveStatus}</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <input required value={identity} onChange={(event) => { setIdentity(event.target.value); setIntakeError(''); }} placeholder="Claimed identity" className="rounded-lg border border-[#cbd8e3] bg-[#f1f5f8] px-3 py-2 text-xs text-[#1c3048] placeholder-[#8294a6] focus:border-[#4d9b88] focus:outline-none" />
        <input required value={phone} onChange={(event) => { setPhone(event.target.value); setIntakeError(''); }} placeholder="Source phone" className="rounded-lg border border-[#cbd8e3] bg-[#f1f5f8] px-3 py-2 text-xs text-[#1c3048] placeholder-[#8294a6] focus:border-[#4d9b88] focus:outline-none" />
      </div>
      <div className="flex flex-wrap gap-2">
        <button disabled={isBusy} onClick={() => (isLive ? stopLiveCall() : handleStart())} className="flex items-center gap-2 rounded-lg bg-emerald-500 px-3 py-2 text-xs font-semibold text-slate-950 disabled:opacity-50">
          {isLive ? <Square className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}{isLive ? 'Stop live call' : 'Start live call'}
        </button>
        <button disabled={isBusy || isLive} onClick={() => fileRef.current?.click()} className="flex items-center gap-2 rounded-lg border border-amber-400/40 px-3 py-2 text-xs font-semibold text-amber-400 disabled:opacity-50">
          {liveStatus === 'uploading' ? <LoaderCircle className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />} Upload recording
        </button>
        <input ref={fileRef} type="file" accept="audio/wav,audio/x-wav,audio/flac,audio/aiff" onChange={handleUpload} className="hidden" />
      </div>
      {intakeError && <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-700">{intakeError}</p>}
      {liveError && <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs leading-relaxed text-[#c84e58]">{liveError}</p>}
      <p className="text-[10px] text-slate-500">Live calls use microphone access. Upload WAV, AIFF, or FLAC recordings.</p>
    </div>
  );
}