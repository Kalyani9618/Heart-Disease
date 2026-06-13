
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { apiClient } from '../services/apiClient';
import { authService } from '../services/authService';

interface FormErrors {
  name?: string;
  email?: string;
  phone?: string;
  password?: string;
  confirmPassword?: string;
}

const SignUpScreen: React.FC = () => {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Form state
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [agreeTerms, setAgreeTerms] = useState(false);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState<FormErrors>({});
  const [success, setSuccess] = useState(false);

  // Validation helpers
  const validateEmail = (val: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val);
  const validatePhone = (val: string) => !val || /^[+]?[\d\s()-]{7,15}$/.test(val);
  const validatePassword = (val: string) => val.length >= 8;

  const validate = (): boolean => {
    const errors: FormErrors = {};

    if (!name.trim()) errors.name = 'Full name is required';
    else if (name.trim().length < 2) errors.name = 'Name must be at least 2 characters';

    if (!email.trim()) errors.email = 'Email is required';
    else if (!validateEmail(email)) errors.email = 'Please enter a valid email';

    if (phone && !validatePhone(phone)) errors.phone = 'Please enter a valid phone number';

    if (!password) errors.password = 'Password is required';
    else if (!validatePassword(password)) errors.password = 'Password must be at least 8 characters';
    else if (!/[A-Z]/.test(password)) errors.password = 'Password must contain an uppercase letter';
    else if (!/[0-9]/.test(password)) errors.password = 'Password must contain a number';

    if (!confirmPassword) errors.confirmPassword = 'Please confirm your password';
    else if (password !== confirmPassword) errors.confirmPassword = 'Passwords do not match';

    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!validate()) return;

    if (!agreeTerms) {
      setError('You must agree to the Terms of Service and Privacy Policy.');
      return;
    }

    setLoading(true);
    try {
      const response = await apiClient.register({
        name: name.trim(),
        email: email.trim().toLowerCase(),
        password,
      });

      // Store auth tokens on successful registration
      if (response.token) {
        authService.setToken(response.token);
        if (response.refresh_token) {
          authService.setRefreshToken(response.refresh_token);
        }
        authService.setUser(response.user);
      }

      setSuccess(true);
      // Navigate to dashboard after short delay for feedback
      setTimeout(() => navigate('/dashboard'), 1200);
    } catch (err: any) {
      console.error('Registration error:', err);
      if (err?.status === 409) {
        setError('An account with this email already exists. Please log in instead.');
      } else {
        setError(err?.message || 'Registration failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Password strength indicator
  const getPasswordStrength = (): { label: string; color: string; width: string } => {
    if (!password) return { label: '', color: '', width: '0%' };
    let score = 0;
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[0-9]/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;

    if (score <= 2) return { label: 'Weak', color: 'bg-red-500', width: '33%' };
    if (score <= 3) return { label: 'Medium', color: 'bg-yellow-500', width: '66%' };
    return { label: 'Strong', color: 'bg-green-500', width: '100%' };
  };

  const strength = getPasswordStrength();

  // Success state
  if (success) {
    return (
      <div className="min-h-screen bg-[#111111] text-white flex flex-col items-center justify-center p-6">
        <div className="w-20 h-20 bg-green-500/10 rounded-full flex items-center justify-center mb-6 border border-green-500/20">
          <span className="material-symbols-outlined text-green-500 text-4xl">check_circle</span>
        </div>
        <h1 className="text-2xl font-bold mb-2">Account Created!</h1>
        <p className="text-slate-400 text-center">Redirecting to your dashboard...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#111111] text-white flex flex-col items-center justify-center p-6 relative overflow-x-hidden">
      <div className="w-16 h-16 bg-red-900/30 rounded-2xl flex items-center justify-center mb-6 border border-red-900/50 shadow-[0_0_30px_rgba(220,38,38,0.3)]">
        <span className="material-symbols-outlined text-red-500 text-3xl">ecg_heart</span>
      </div>

      <h1 className="text-3xl font-bold mb-2">Get Started</h1>
      <p className="text-slate-400 mb-8 text-center max-w-xs">Create an account to monitor your heart health.</p>

      <form onSubmit={handleSubmit} className="w-full max-w-sm space-y-4">
        {/* Global error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm flex items-start gap-2">
            <span className="material-symbols-outlined text-sm mt-0.5">error</span>
            <span>{error}</span>
          </div>
        )}

        {/* Full Name */}
        <div>
          <div className="relative">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">person</span>
            <input
              type="text"
              placeholder="Full Name"
              required
              value={name}
              onChange={(e) => { setName(e.target.value); setFieldErrors(prev => ({ ...prev, name: undefined })); }}
              className={`w-full h-14 bg-slate-800/30 border ${fieldErrors.name ? 'border-red-500' : 'border-slate-700'} rounded-xl pl-12 pr-4 outline-none focus:border-red-500 focus:ring-2 focus:ring-red-500/20 transition-colors text-white placeholder:text-slate-500`}
            />
          </div>
          {fieldErrors.name && <p className="text-red-400 text-xs mt-1 ml-1">{fieldErrors.name}</p>}
        </div>

        {/* Email */}
        <div>
          <div className="relative">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">mail</span>
            <input
              type="email"
              placeholder="Email Address"
              required
              value={email}
              onChange={(e) => { setEmail(e.target.value); setFieldErrors(prev => ({ ...prev, email: undefined })); }}
              className={`w-full h-14 bg-slate-800/30 border ${fieldErrors.email ? 'border-red-500' : 'border-slate-700'} rounded-xl pl-12 pr-4 outline-none focus:border-red-500 focus:ring-2 focus:ring-red-500/20 transition-colors text-white placeholder:text-slate-500`}
            />
          </div>
          {fieldErrors.email && <p className="text-red-400 text-xs mt-1 ml-1">{fieldErrors.email}</p>}
        </div>

        {/* Phone */}
        <div>
          <div className="relative">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">phone</span>
            <input
              type="tel"
              placeholder="Mobile Number (optional)"
              value={phone}
              onChange={(e) => { setPhone(e.target.value); setFieldErrors(prev => ({ ...prev, phone: undefined })); }}
              className={`w-full h-14 bg-slate-800/30 border ${fieldErrors.phone ? 'border-red-500' : 'border-slate-700'} rounded-xl pl-12 pr-4 outline-none focus:border-red-500 focus:ring-2 focus:ring-red-500/20 transition-colors text-white placeholder:text-slate-500`}
            />
          </div>
          {fieldErrors.phone && <p className="text-red-400 text-xs mt-1 ml-1">{fieldErrors.phone}</p>}
        </div>

        {/* Password */}
        <div>
          <div className="relative">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">lock</span>
            <input
                type={showPassword ? "text" : "password"}
                placeholder="Password"
                required
                value={password}
                onChange={(e) => { setPassword(e.target.value); setFieldErrors(prev => ({ ...prev, password: undefined })); }}
                className={`w-full h-14 bg-slate-800/30 border ${fieldErrors.password ? 'border-red-500' : 'border-slate-700'} rounded-xl pl-12 pr-12 outline-none focus:border-red-500 focus:ring-2 focus:ring-red-500/20 transition-colors text-white placeholder:text-slate-500`}
            />
            <span
                className="material-symbols-outlined absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 cursor-pointer hover:text-slate-300 transition-colors select-none"
                onClick={() => setShowPassword(!showPassword)}
            >
                {showPassword ? 'visibility' : 'visibility_off'}
            </span>
          </div>
          {fieldErrors.password && <p className="text-red-400 text-xs mt-1 ml-1">{fieldErrors.password}</p>}
          {/* Password strength bar */}
          {password && (
            <div className="mt-2">
              <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className={`h-full ${strength.color} transition-all duration-300 rounded-full`} style={{ width: strength.width }} />
              </div>
              <p className={`text-xs mt-1 ml-1 ${strength.color === 'bg-red-500' ? 'text-red-400' : strength.color === 'bg-yellow-500' ? 'text-yellow-400' : 'text-green-400'}`}>
                {strength.label} password
              </p>
            </div>
          )}
        </div>

        {/* Confirm Password */}
        <div>
          <div className="relative">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">lock_reset</span>
            <input
                type={showConfirmPassword ? "text" : "password"}
                placeholder="Confirm Password"
                required
                value={confirmPassword}
                onChange={(e) => { setConfirmPassword(e.target.value); setFieldErrors(prev => ({ ...prev, confirmPassword: undefined })); }}
                className={`w-full h-14 bg-slate-800/30 border ${fieldErrors.confirmPassword ? 'border-red-500' : 'border-slate-700'} rounded-xl pl-12 pr-12 outline-none focus:border-red-500 focus:ring-2 focus:ring-red-500/20 transition-colors text-white placeholder:text-slate-500`}
            />
            <span
                className="material-symbols-outlined absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 cursor-pointer hover:text-slate-300 transition-colors select-none"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
            >
                {showConfirmPassword ? 'visibility' : 'visibility_off'}
            </span>
          </div>
          {fieldErrors.confirmPassword && <p className="text-red-400 text-xs mt-1 ml-1">{fieldErrors.confirmPassword}</p>}
          {/* Match indicator */}
          {confirmPassword && password && !fieldErrors.confirmPassword && password === confirmPassword && (
            <p className="text-green-400 text-xs mt-1 ml-1 flex items-center gap-1">
              <span className="material-symbols-outlined text-xs">check_circle</span> Passwords match
            </p>
          )}
        </div>

        {/* Terms checkbox */}
        <label className="flex items-start gap-3 cursor-pointer pt-2">
          <input
            type="checkbox"
            checked={agreeTerms}
            onChange={(e) => setAgreeTerms(e.target.checked)}
            className="mt-1 w-4 h-4 rounded border-slate-600 bg-slate-800 text-red-500 focus:ring-red-500"
          />
          <span className="text-slate-400 text-xs leading-relaxed">
            I agree to the{' '}
            <Link to="/consent" className="text-red-400 hover:underline">Terms of Service</Link>
            {' '}and{' '}
            <Link to="/consent" className="text-red-400 hover:underline">Privacy Policy</Link>
          </span>
        </label>

        <button
          type="submit"
          disabled={loading}
          className="w-full h-14 bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl mt-4 transition-all shadow-lg shadow-red-600/30 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
              Creating Account...
            </>
          ) : (
            'Create Account'
          )}
        </button>
      </form>

      <p className="text-center mt-8 text-slate-400 text-sm">
         Already have an account? <Link to="/login" className="text-red-500 font-bold hover:underline">Log In</Link>
      </p>
    </div>
  );
};

export default SignUpScreen;
