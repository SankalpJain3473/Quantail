// src/components/AuthPages.tsx
// Login + Register pages with full validation and invite code flow

import { useState, FormEvent } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { authApi } from '../lib/api';
import { useTradingStore } from '../store/tradingStore';

// ── Shared UI ─────────────────────────────────────────────────────────────
function AuthCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-[#0f0f13] flex items-center justify-center p-4">
      <div className="w-full max-w-md">

        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-lg bg-[#534ab7] flex items-center justify-center">
              <span className="text-white text-xs font-bold">Q</span>
            </div>
            <h1 className="text-2xl font-bold">
              <span className="text-[#7f77dd]">Quant</span>
              <span className="text-[#5dcaa5]">ail</span>
            </h1>
          </div>
          <p className="text-[#888780] text-sm">
            Distributional Quantum RL Trading Platform
          </p>
          <p className="text-[#534ab7] text-xs mt-1">
            Sankalp Jain & Veronica Koval · Columbia University
          </p>
        </div>

        <div className="bg-[#18181f] border border-[#2a2a35] rounded-xl overflow-hidden">
          {children}
        </div>

        <p className="text-[#5f5e5a] text-xs text-center mt-4">
          JWT secured · bcrypt passwords · Invite-only access
        </p>
      </div>
    </div>
  );
}

function InputField({
  label, type = 'text', value, onChange, placeholder, hint, error,
}: {
  label: string; type?: string; value: string;
  onChange: (v: string) => void; placeholder?: string;
  hint?: string; error?: string;
}) {
  return (
    <div>
      <label className="block text-[#888780] text-xs font-semibold uppercase tracking-wider mb-1.5">
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={`w-full bg-[#0f0f13] border rounded-lg px-3 py-2.5 text-[#e0dfd8] text-sm
                   placeholder-[#5f5e5a] focus:outline-none transition-colors
                   ${error ? 'border-[#a32d2d] focus:border-[#f09595]' : 'border-[#2a2a35] focus:border-[#534ab7]'}`}
        autoComplete={type === 'password' ? 'new-password' : undefined}
      />
      {error && <p className="text-[#f09595] text-xs mt-1">{error}</p>}
      {hint && !error && <p className="text-[#5f5e5a] text-xs mt-1">{hint}</p>}
    </div>
  );
}

function SubmitButton({ loading, label }: { loading: boolean; label: string }) {
  return (
    <button
      type="submit"
      disabled={loading}
      className="w-full bg-[#534ab7] hover:bg-[#4a42a5] active:bg-[#3d3690]
                 text-white font-semibold py-2.5 rounded-lg text-sm
                 transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                 flex items-center justify-center gap-2"
    >
      {loading && (
        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
        </svg>
      )}
      {loading ? 'Please wait...' : label}
    </button>
  );
}

// ── Password strength indicator ───────────────────────────────────────────
function PasswordStrength({ password }: { password: string }) {
  const checks = [
    { label: 'At least 8 characters', ok: password.length >= 8 },
    { label: 'One uppercase letter', ok: /[A-Z]/.test(password) },
    { label: 'One number', ok: /\d/.test(password) },
    { label: 'One special character', ok: /[^a-zA-Z0-9]/.test(password) },
  ];
  const score = checks.filter(c => c.ok).length;
  const colors = ['bg-[#a32d2d]', 'bg-[#854f0b]', 'bg-[#854f0b]', 'bg-[#0f6e56]', 'bg-[#0f6e56]'];
  const labels = ['', 'Weak', 'Fair', 'Good', 'Strong'];

  if (!password) return null;

  return (
    <div className="mt-2">
      <div className="flex gap-1 mb-1.5">
        {[1,2,3,4].map(i => (
          <div key={i} className={`h-1 flex-1 rounded-full transition-colors ${i <= score ? colors[score] : 'bg-[#2a2a35]'}`} />
        ))}
        <span className={`text-xs ml-1 ${score >= 4 ? 'text-[#5dcaa5]' : score >= 2 ? 'text-[#ef9f27]' : 'text-[#f09595]'}`}>
          {labels[score]}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-1">
        {checks.map(({ label, ok }) => (
          <div key={label} className="flex items-center gap-1">
            <span className={`text-[10px] ${ok ? 'text-[#5dcaa5]' : 'text-[#5f5e5a]'}`}>
              {ok ? '✓' : '○'} {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════
// LOGIN PAGE
// ══════════════════════════════════════════════════════════════════════════
export function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { setAuth } = useTradingStore();
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    if (!username.trim() || !password) {
      setError('Please fill in all fields');
      return;
    }
    setLoading(true);
    try {
      const { data } = await authApi.login(username.trim(), password);
      setAuth({
        token: data.access_token,
        username: data.user.username,
        isAuthenticated: true,
      });
      // Store refresh token separately
      localStorage.setItem('quantail_refresh', data.refresh_token);
      navigate('/');
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Login failed. Please try again.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthCard>
      <div className="p-6">
        <h2 className="text-[#e0dfd8] font-bold text-lg mb-1">Sign in</h2>
        <p className="text-[#888780] text-xs mb-5">
          Access your Quantail trading dashboard
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <InputField
            label="Username"
            value={username}
            onChange={setUsername}
            placeholder="your-username"
          />
          <InputField
            label="Password"
            type="password"
            value={password}
            onChange={setPassword}
            placeholder="••••••••••"
          />

          {error && (
            <div className="bg-[#3d1515] border border-[#a32d2d] rounded-lg px-3 py-2.5
                           text-[#f09595] text-sm flex items-start gap-2">
              <span className="mt-0.5">⚠</span>
              <span>{error}</span>
            </div>
          )}

          <SubmitButton loading={loading} label="Sign in" />
        </form>

        <div className="mt-4 pt-4 border-t border-[#2a2a35] text-center">
          <p className="text-[#888780] text-xs">
            Don't have an account?{' '}
            <Link to="/register" className="text-[#7f77dd] hover:text-[#9f97f7] transition-colors font-semibold">
              Register with invite code
            </Link>
          </p>
        </div>
      </div>

      {/* Info footer */}
      <div className="px-6 py-3 bg-[#0f0f13] border-t border-[#2a2a35]">
        <p className="text-[#5f5e5a] text-xs text-center">
          Access is restricted — contact an admin for an invite code
        </p>
      </div>
    </AuthCard>
  );
}

// ══════════════════════════════════════════════════════════════════════════
// REGISTER PAGE
// ══════════════════════════════════════════════════════════════════════════
export function RegisterPage() {
  const [form, setForm] = useState({
    username: '', email: '', password: '', confirm: '',
    fullName: '', inviteCode: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [globalError, setGlobalError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const { setAuth } = useTradingStore();
  const navigate = useNavigate();

  // Pre-fill invite from URL: /register?invite=CODE
  useState(() => {
    const params = new URLSearchParams(window.location.search);
    const invite = params.get('invite');
    if (invite) setForm(f => ({ ...f, inviteCode: invite }));
  });

  const update = (field: string, value: string) => {
    setForm(f => ({ ...f, [field]: value }));
    if (errors[field]) setErrors(e => ({ ...e, [field]: '' }));
    setGlobalError('');
  };

  const validate = () => {
    const errs: Record<string, string> = {};
    if (!form.username.trim()) errs.username = 'Username is required';
    else if (!/^[a-zA-Z0-9_-]{3,50}$/.test(form.username))
      errs.username = 'Letters, numbers, _ and - only (3-50 chars)';
    if (!form.email.trim()) errs.email = 'Email is required';
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      errs.email = 'Invalid email address';
    if (!form.password) errs.password = 'Password is required';
    else if (form.password.length < 8) errs.password = 'At least 8 characters';
    else if (!/[A-Z]/.test(form.password)) errs.password = 'Need one uppercase letter';
    else if (!/\d/.test(form.password)) errs.password = 'Need one number';
    else if (!/[^a-zA-Z0-9]/.test(form.password)) errs.password = 'Need one special character';
    if (form.password !== form.confirm) errs.confirm = 'Passwords do not match';
    if (!form.inviteCode.trim()) errs.inviteCode = 'Invite code is required';
    return errs;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length > 0) { setErrors(errs); return; }

    setLoading(true);
    setGlobalError('');
    try {
      const { data } = await authApi.register({
        username:    form.username.trim(),
        email:       form.email.trim(),
        password:    form.password,
        full_name:   form.fullName.trim(),
        invite_code: form.inviteCode.trim(),
      });

      setSuccess(true);
      setAuth({ token: data.access_token, username: data.user.username, isAuthenticated: true });
      localStorage.setItem('quantail_refresh', data.refresh_token);

      setTimeout(() => navigate('/'), 1500);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      if (typeof detail === 'string') {
        if (detail.includes('Username')) setErrors(e => ({ ...e, username: detail }));
        else if (detail.includes('Email')) setErrors(e => ({ ...e, email: detail }));
        else if (detail.includes('nvite') || detail.includes('code')) setErrors(e => ({ ...e, inviteCode: detail }));
        else if (detail.includes('assword')) setErrors(e => ({ ...e, password: detail }));
        else setGlobalError(detail);
      } else {
        setGlobalError('Registration failed. Please check your details.');
      }
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <AuthCard>
        <div className="p-8 text-center">
          <div className="w-14 h-14 rounded-full bg-[#0a2e1e] border border-[#0f6e56] flex items-center justify-center mx-auto mb-4">
            <span className="text-[#5dcaa5] text-2xl">✓</span>
          </div>
          <h2 className="text-[#e0dfd8] font-bold text-lg mb-2">Account created!</h2>
          <p className="text-[#888780] text-sm">Redirecting to dashboard...</p>
        </div>
      </AuthCard>
    );
  }

  return (
    <AuthCard>
      <div className="p-6">
        <h2 className="text-[#e0dfd8] font-bold text-lg mb-1">Create account</h2>
        <p className="text-[#888780] text-xs mb-5">
          You need a valid invite code to register
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">

          {/* Invite code — first, most important */}
          <div className="bg-[#0a1e30] border border-[#185fa5] rounded-lg px-4 py-3">
            <label className="block text-[#378add] text-xs font-semibold uppercase tracking-wider mb-1.5">
              Invite Code *
            </label>
            <input
              type="text"
              value={form.inviteCode}
              onChange={(e) => update('inviteCode', e.target.value)}
              placeholder="Paste your invite code here"
              className={`w-full bg-[#0f0f13] border rounded-lg px-3 py-2.5 text-[#e0dfd8]
                         text-sm placeholder-[#5f5e5a] focus:outline-none transition-colors
                         font-mono ${errors.inviteCode ? 'border-[#a32d2d]' : 'border-[#2a2a35] focus:border-[#378add]'}`}
            />
            {errors.inviteCode
              ? <p className="text-[#f09595] text-xs mt-1">{errors.inviteCode}</p>
              : <p className="text-[#5f5e5a] text-xs mt-1">Ask an admin to generate an invite code for you</p>
            }
          </div>

          <div className="grid grid-cols-2 gap-3">
            <InputField
              label="Username *"
              value={form.username}
              onChange={(v) => update('username', v)}
              placeholder="sankalp_j"
              hint="Letters, numbers, _ and -"
              error={errors.username}
            />
            <InputField
              label="Full name"
              value={form.fullName}
              onChange={(v) => update('fullName', v)}
              placeholder="Sankalp Jain"
            />
          </div>

          <InputField
            label="Email address *"
            type="email"
            value={form.email}
            onChange={(v) => update('email', v)}
            placeholder="you@columbia.edu"
            error={errors.email}
          />

          <div>
            <InputField
              label="Password *"
              type="password"
              value={form.password}
              onChange={(v) => update('password', v)}
              placeholder="Min 8 chars, uppercase, number, special"
              error={errors.password}
            />
            <PasswordStrength password={form.password} />
          </div>

          <InputField
            label="Confirm password *"
            type="password"
            value={form.confirm}
            onChange={(v) => update('confirm', v)}
            placeholder="Repeat your password"
            error={errors.confirm}
          />

          {globalError && (
            <div className="bg-[#3d1515] border border-[#a32d2d] rounded-lg px-3 py-2.5 text-[#f09595] text-sm">
              ⚠ {globalError}
            </div>
          )}

          <SubmitButton loading={loading} label="Create account" />
        </form>

        <div className="mt-4 pt-4 border-t border-[#2a2a35] text-center">
          <p className="text-[#888780] text-xs">
            Already have an account?{' '}
            <Link to="/login" className="text-[#7f77dd] hover:text-[#9f97f7] transition-colors font-semibold">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </AuthCard>
  );
}
