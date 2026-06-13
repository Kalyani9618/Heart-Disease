
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

const LoginScreen: React.FC = () => {
    const navigate = useNavigate();
    const { login } = useAuth();
    const [showForgotModal, setShowForgotModal] = useState(false);
    const [showPassword, setShowPassword] = useState(false);

    // Form state
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Forgot Password Wizard State
    const [forgotStep, setForgotStep] = useState<'method' | 'input' | 'otp' | 'reset' | 'success'>('method');
    const [resetMethod, setResetMethod] = useState<'email' | 'sms'>('email');
    const [contactInput, setContactInput] = useState('');
    const [otp, setOtp] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [forgotError, setForgotError] = useState('');

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await login(email, password);
            navigate('/dashboard');
        } catch (err: any) {
            console.error('Login error:', err);
            setError(err?.message || 'Invalid email or password. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleLogin = async () => {
        // TODO: Implement real Google OAuth via Firebase Auth
        // For now, show user feedback instead of silently navigating
        setError('Google Sign-In is not yet configured. Please use email and password.');
    };

    const openForgotModal = () => {
        setForgotStep('method');
        setContactInput('');
        setOtp('');
        setNewPassword('');
        setConfirmPassword('');
        setForgotError('');
        setShowForgotModal(true);
    };

    const handleMethodSelect = (method: 'email' | 'sms') => {
        setResetMethod(method);
        setForgotStep('input');
    };

    const handleSendOtp = (e: React.FormEvent) => {
        e.preventDefault();
        if (contactInput) {
            // Simulate API call to send OTP
            setForgotStep('otp');
        }
    };

    const handleVerifyOtp = (e: React.FormEvent) => {
        e.preventDefault();
        setForgotError('');
        // Simulate OTP verification
        if (otp.length === 6) {
            setForgotStep('reset');
        } else {
            setForgotError('Please enter a 6-digit code.');
        }
    };

    const handleResetSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setForgotError('');
        if (!newPassword) {
            setForgotError('Please enter a new password.');
        } else if (newPassword.length < 8) {
            setForgotError('Password must be at least 8 characters.');
        } else if (newPassword !== confirmPassword) {
            setForgotError('Passwords do not match.');
        } else {
            setForgotStep('success');
        }
    };

    const closeResetModal = () => {
        setShowForgotModal(false);
    };

    // Helper to mask contact info for display
    const maskedContact = () => {
        if (!contactInput) return '';
        if (resetMethod === 'email') {
            const [name, domain] = contactInput.split('@');
            return `${name.substring(0, 3)}***@${domain}`;
        }
        return `${contactInput.substring(0, 3)} **** ${contactInput.substring(contactInput.length - 2)}`;
    };

    return (
        <div className="min-h-screen bg-[#1a0b0b] text-white flex flex-col items-center justify-center p-6 relative overflow-hidden">
            {/* Background Gradient Effect */}
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-transparent to-black/80 pointer-events-none"></div>

            <div className="w-16 h-16 bg-red-900/30 rounded-2xl flex items-center justify-center mb-8 border border-red-900/50 relative z-10 shadow-[0_0_30px_rgba(220,38,38,0.3)]">
                <span className="material-symbols-outlined text-red-500 text-3xl">ecg_heart</span>
            </div>

            <h1 className="text-3xl font-bold mb-2 relative z-10">Welcome Back</h1>
            <p className="text-slate-400 mb-10 relative z-10">Log in to your account to continue</p>

            <form onSubmit={handleLogin} className="w-full max-w-sm space-y-4 relative z-10">
                {error && (
                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
                        {error}
                    </div>
                )}

                <div className="space-y-2">
                    <label className="text-sm text-slate-300 ml-1">Email or Username</label>
                    <div className="relative">
                        <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">person</span>
                        <input
                            type="text"
                            required
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="Enter your email or username"
                            className="w-full h-14 bg-slate-800/50 border border-slate-700 rounded-xl pl-12 pr-4 text-white placeholder:text-slate-600 focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all"
                        />
                    </div>
                </div>

                <div className="space-y-2">
                    <label className="text-sm text-slate-300 ml-1">Password</label>
                    <div className="relative">
                        <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">lock</span>
                        <input
                            type={showPassword ? "text" : "password"}
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter your password"
                            className="w-full h-14 bg-slate-800/50 border border-slate-700 rounded-xl pl-12 pr-12 text-white placeholder:text-slate-600 focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all"
                        />
                        <span
                            className="material-symbols-outlined absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 cursor-pointer hover:text-white select-none"
                            onClick={() => setShowPassword(!showPassword)}
                        >
                            {showPassword ? 'visibility' : 'visibility_off'}
                        </span>
                    </div>
                </div>

                <div className="flex justify-end">
                    <button
                        type="button"
                        onClick={openForgotModal}
                        className="text-sm text-red-400 hover:text-red-300 transition-colors"
                    >
                        Forgot Password?
                    </button>
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full h-14 bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all shadow-lg shadow-red-600/30 mt-4"
                >
                    {loading ? 'Logging in...' : 'Log In'}
                </button>
            </form>

            <div className="w-full max-w-sm relative z-10 mt-8">
                <div className="flex items-center gap-4 mb-6">
                    <div className="h-px bg-slate-800 flex-1"></div>
                    <span className="text-slate-500 text-sm">OR</span>
                    <div className="h-px bg-slate-800 flex-1"></div>
                </div>

                <div className="space-y-3">
                    <button
                        onClick={handleGoogleLogin}
                        className="w-full h-12 bg-slate-800 border border-slate-700 rounded-xl flex items-center justify-center gap-3 hover:bg-slate-700 transition-colors"
                    >
                        <img src="https://www.svgrepo.com/show/475656/google-color.svg" className="w-5 h-5" alt="Google" />
                        <span className="font-medium text-sm">Continue with Google</span>
                    </button>
                </div>

                <p className="text-center mt-8 text-slate-400 text-sm">
                    Don't have an account? <Link to="/signup" className="text-red-500 font-bold hover:underline">Sign Up</Link>
                </p>
            </div>

            {/* Forgot Password Modal */}
            {showForgotModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-[#1a0b0b] border border-red-900/30 rounded-2xl p-6 w-full max-w-sm shadow-2xl relative overflow-hidden animate-in zoom-in-95 duration-200">
                        {/* Decorative glow */}
                        <div className="absolute top-0 right-0 w-32 h-32 bg-red-600/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>

                        <div className="relative z-10">
                            {/* Step 1: Select Method */}
                            {forgotStep === 'method' && (
                                <>
                                    <h3 className="text-xl font-bold text-white mb-2">Forgot Password</h3>
                                    <p className="text-slate-400 text-sm mb-6">Select which contact details should we use to reset your password:</p>

                                    <div className="space-y-3">
                                        <button
                                            onClick={() => handleMethodSelect('email')}
                                            className="w-full p-4 bg-slate-800/50 border border-slate-700 rounded-xl flex items-center gap-4 hover:bg-slate-700 transition-colors group"
                                        >
                                            <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center group-hover:bg-slate-600 transition-colors">
                                                <span className="material-symbols-outlined text-slate-300">mail</span>
                                            </div>
                                            <div className="text-left">
                                                <p className="text-sm font-medium text-slate-300">via Email</p>
                                                <p className="text-xs text-slate-500">Code sent to your email</p>
                                            </div>
                                        </button>

                                        <button
                                            onClick={() => handleMethodSelect('sms')}
                                            className="w-full p-4 bg-slate-800/50 border border-slate-700 rounded-xl flex items-center gap-4 hover:bg-slate-700 transition-colors group"
                                        >
                                            <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center group-hover:bg-slate-600 transition-colors">
                                                <span className="material-symbols-outlined text-slate-300">sms</span>
                                            </div>
                                            <div className="text-left">
                                                <p className="text-sm font-medium text-slate-300">via SMS</p>
                                                <p className="text-xs text-slate-500">Code sent to mobile number</p>
                                            </div>
                                        </button>
                                    </div>

                                    <button onClick={closeResetModal} className="w-full mt-6 py-3 text-slate-400 hover:text-white text-sm font-medium">Cancel</button>
                                </>
                            )}

                            {/* Step 2: Input Contact Info */}
                            {forgotStep === 'input' && (
                                <form onSubmit={handleSendOtp}>
                                    <h3 className="text-xl font-bold text-white mb-2">
                                        {resetMethod === 'email' ? 'Enter Email' : 'Enter Mobile Number'}
                                    </h3>
                                    <p className="text-slate-400 text-sm mb-6">
                                        We will send a 6-digit OTP to this {resetMethod === 'email' ? 'email address' : 'mobile number'}.
                                    </p>

                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <div className="relative">
                                                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">
                                                    {resetMethod === 'email' ? 'mail' : 'phone'}
                                                </span>
                                                <input
                                                    type={resetMethod === 'email' ? 'email' : 'tel'}
                                                    required
                                                    value={contactInput}
                                                    onChange={(e) => setContactInput(e.target.value)}
                                                    placeholder={resetMethod === 'email' ? "name@example.com" : "+1 123 456 7890"}
                                                    className="w-full h-12 bg-slate-800/50 border border-slate-700 rounded-xl pl-12 pr-4 text-white placeholder:text-slate-600 focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all"
                                                />
                                            </div>
                                        </div>
                                        <div className="flex gap-3 pt-2">
                                            <button
                                                type="button"
                                                onClick={() => setForgotStep('method')}
                                                className="flex-1 h-12 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded-xl transition-colors"
                                            >
                                                Back
                                            </button>
                                            <button
                                                type="submit"
                                                disabled={!contactInput}
                                                className="flex-1 h-12 bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-colors shadow-lg shadow-red-600/20"
                                            >
                                                Send OTP
                                            </button>
                                        </div>
                                    </div>
                                </form>
                            )}

                            {/* Step 3: Enter OTP */}
                            {forgotStep === 'otp' && (
                                <form onSubmit={handleVerifyOtp}>
                                    <h3 className="text-xl font-bold text-white mb-2">Verify Code</h3>
                                    <p className="text-slate-400 text-sm mb-6">
                                        Enter the 6-digit code sent to <br />
                                        <span className="text-white font-medium">{maskedContact()}</span>
                                    </p>

                                    {forgotError && (
                                        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-red-400 text-sm mb-4">
                                            {forgotError}
                                        </div>
                                    )}

                                    <div className="space-y-6">
                                        <div className="flex justify-center">
                                            <input
                                                type="text"
                                                required
                                                maxLength={6}
                                                value={otp}
                                                onChange={(e) => setOtp(e.target.value.replace(/[^0-9]/g, ''))}
                                                className="w-full h-14 bg-slate-800/50 border border-slate-700 rounded-xl text-center text-2xl tracking-[0.5em] text-white focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all font-mono"
                                                placeholder="000000"
                                            />
                                        </div>
                                        <div className="text-center">
                                            <p className="text-xs text-slate-500">
                                                Didn't receive code? <button type="button" className="text-red-400 hover:text-red-300 font-bold">Resend</button>
                                            </p>
                                        </div>
                                        <div className="flex gap-3 pt-2">
                                            <button
                                                type="button"
                                                onClick={() => setForgotStep('input')}
                                                className="flex-1 h-12 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded-xl transition-colors"
                                            >
                                                Back
                                            </button>
                                            <button
                                                type="submit"
                                                disabled={otp.length !== 6}
                                                className="flex-1 h-12 bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-colors shadow-lg shadow-red-600/20"
                                            >
                                                Verify
                                            </button>
                                        </div>
                                    </div>
                                </form>
                            )}

                            {/* Step 4: Reset Password */}
                            {forgotStep === 'reset' && (
                                <form onSubmit={handleResetSubmit}>
                                    <h3 className="text-xl font-bold text-white mb-2">Create New Password</h3>
                                    <p className="text-slate-400 text-sm mb-6">Enter your new password below.</p>

                                    {forgotError && (
                                        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-red-400 text-sm mb-4">
                                            {forgotError}
                                        </div>
                                    )}

                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <div className="relative">
                                                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">lock</span>
                                                <input
                                                    type="password"
                                                    required
                                                    value={newPassword}
                                                    onChange={(e) => setNewPassword(e.target.value)}
                                                    placeholder="New Password"
                                                    className="w-full h-12 bg-slate-800/50 border border-slate-700 rounded-xl pl-12 pr-4 text-white placeholder:text-slate-600 focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all"
                                                />
                                            </div>
                                            <div className="relative">
                                                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">lock_reset</span>
                                                <input
                                                    type="password"
                                                    required
                                                    value={confirmPassword}
                                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                                    placeholder="Confirm Password"
                                                    className="w-full h-12 bg-slate-800/50 border border-slate-700 rounded-xl pl-12 pr-4 text-white placeholder:text-slate-600 focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all"
                                                />
                                            </div>
                                        </div>
                                        <button
                                            type="submit"
                                            className="w-full h-12 bg-red-600 hover:bg-red-500 text-white font-bold rounded-xl transition-colors shadow-lg shadow-red-600/20 mt-2"
                                        >
                                            Reset Password
                                        </button>
                                    </div>
                                </form>
                            )}

                            {/* Step 5: Success */}
                            {forgotStep === 'success' && (
                                <div className="text-center py-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
                                    <div className="w-16 h-16 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-4 text-green-500 border border-green-500/20">
                                        <span className="material-symbols-outlined text-3xl">check_circle</span>
                                    </div>
                                    <h3 className="text-xl font-bold text-white mb-2">Password Reset!</h3>
                                    <p className="text-slate-400 text-sm mb-6">
                                        Your password has been successfully updated. You can now log in.
                                    </p>
                                    <button
                                        onClick={closeResetModal}
                                        className="w-full h-12 bg-slate-800 hover:bg-slate-700 text-white font-bold rounded-xl transition-colors"
                                    >
                                        Back to Login
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default LoginScreen;
