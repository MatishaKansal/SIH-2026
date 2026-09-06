import { useState } from 'react';
import { ArrowRight, Eye, EyeOff, LockKeyhole, Mail, ShieldCheck, UserRound } from 'lucide-react';

export default function AuthPage({ onAuthenticated }) {
  const [mode, setMode] = useState('signin');
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const update = (field) => (event) => setForm((current) => ({ ...current, [field]: event.target.value }));

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    const email = form.email.trim().toLowerCase();
    if (!email || !form.password) {
      setError('Enter your email and password to continue.');
      return;
    }

    if (mode === 'signup' && !form.name.trim()) {
      setError('Enter your full name.');
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch(`/api/v1/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mode === 'signup'
          ? { name: form.name.trim(), email, password: form.password }
          : { email, password: form.password }),
      });
      const responseText = await response.text();
      let result;
      try {
        result = responseText ? JSON.parse(responseText) : {};
      } catch {
        result = { detail: 'The server returned an invalid response.' };
      }
      if (!response.ok) throw new Error(result.detail || 'Authentication failed');
      onAuthenticated(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-[#f4f7fb] px-5 py-10">
      <div className="absolute inset-0 pointer-events-none opacity-40" style={{ background: 'radial-gradient(circle at 20% 20%, rgba(76, 190, 157, .18), transparent 32%), radial-gradient(circle at 85% 75%, rgba(46, 87, 125, .12), transparent 35%)' }} />
      <section className="relative w-full max-w-5xl grid lg:grid-cols-[1.05fr_.95fr] overflow-hidden rounded-[28px] bg-white border border-[#dce6ee] shadow-[0_25px_80px_rgba(28,48,72,.14)]">
        <div className="hidden lg:flex flex-col justify-between bg-[#182a3e] p-12 text-white">
          <div>
            <div className="flex items-center gap-3">
              <img src="/echoguard-logo.png" alt="EchoGuard" className="w-11 h-11 rounded-xl" />
              <div><p className="text-xl font-extrabold tracking-wide">EchoGuard</p><p className="text-xs text-[#9eb3c6]">Call Fraud Prevention</p></div>
            </div>
            <div className="mt-24 max-w-sm">
              <p className="text-xs uppercase tracking-[.22em] text-[#69d2b0]">Enterprise voice security</p>
              <h1 className="mt-4 text-4xl font-extrabold leading-tight">Protect every conversation that matters.</h1>
              <p className="mt-5 text-sm leading-7 text-[#b7c7d4]">Monitor live calls, verify speaker identity, and stop high-risk transactions before they leave your organization.</p>
            </div>
          </div>
          
        </div>

        <div className="p-7 sm:p-12">
          <div className="lg:hidden flex items-center gap-3 mb-12"><img src="/echoguard-logo.png" alt="EchoGuard" className="w-10 h-10 rounded-xl" /><p className="text-lg font-extrabold text-[#182a3e]">EchoGuard</p></div>
          <div className="flex gap-6 border-b border-[#e6edf3]">
            {['signin', 'signup'].map((value) => (
              <button key={value} type="button" onClick={() => { setMode(value); setError(''); }} className={`pb-3 text-sm font-bold capitalize border-b-2 transition-colors ${mode === value ? 'text-[#218e7a] border-[#35c29b]' : 'text-[#8294a6] border-transparent'}`}>
                {value === 'signin' ? 'Sign in' : 'Create account'}
              </button>
            ))}
          </div>
          <div className="mt-9"><h2 className="text-2xl font-extrabold text-[#182a3e]">{mode === 'signin' ? 'Welcome back' : 'Create your account'}</h2><p className="mt-2 text-sm text-[#71869a]">{mode === 'signin' ? 'Sign in to your EchoGuard analyst workspace.' : ''}</p></div>

          <form onSubmit={submit} className="mt-8 space-y-4">
            {mode === 'signup' && <label className="block"><span className="mb-1.5 block text-xs font-bold text-[#425a72]">Full name</span><div className="relative"><UserRound className="absolute left-3 top-3 w-4 h-4 text-[#8294a6]" /><input value={form.name} onChange={update('name')} placeholder="e.g. Priya Sharma" className="auth-input pl-10" /></div></label>}
            <label className="block"><span className="mb-1.5 block text-xs font-bold text-[#425a72]">Work email</span><div className="relative"><Mail className="absolute left-3 top-3 w-4 h-4 text-[#8294a6]" /><input type="email" value={form.email} onChange={update('email')} placeholder="you@company.com" className="auth-input pl-10" /></div></label>
            <label className="block"><span className="mb-1.5 block text-xs font-bold text-[#425a72]">Password</span><div className="relative"><LockKeyhole className="absolute left-3 top-3 w-4 h-4 text-[#8294a6]" /><input type={showPassword ? 'text' : 'password'} value={form.password} onChange={update('password')} placeholder="Enter your password" className="auth-input pl-10 pr-10" /><button type="button" onClick={() => setShowPassword((visible) => !visible)} className="absolute right-3 top-2.5 text-[#8294a6]">{showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}</button></div></label>
            {error && <p role="alert" className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs font-semibold text-red-600">{error}</p>}
            <button type="submit" disabled={isSubmitting} className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-[#35b58f] py-3 text-sm font-extrabold text-white shadow-lg shadow-[#35b58f]/20 hover:bg-[#269e7a] disabled:opacity-60 disabled:cursor-wait transition-colors">{isSubmitting ? 'Connecting...' : mode === 'signin' ? 'Sign in to workspace' : 'Create Account'}{!isSubmitting && <ArrowRight className="w-4 h-4" />}</button>
          </form>
                 </div>
      </section>
    </main>
  );
}