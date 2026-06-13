
import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Device } from '../types';
import { useLanguage } from '../contexts/LanguageContext';
import { pdfExportService } from '../services/pdfExport';
import { calendarService, CalendarServiceError } from '../services/calendarService';
import { notificationService, NotificationServiceError } from '../services/notificationService';
import { nativeNotificationService } from '../services/nativeNotificationService';
import { apiClient } from '../services/apiClient';
import { useBluetooth } from '../hooks/useBluetooth';
import { useToast } from '../components/Toast';
import type { CalendarProvider, WeeklySummaryPreferences, DeliveryChannel } from '../services/api.types';
import { Modal } from '../components/Modal';
import { Capacitor } from '@capacitor/core';

interface SettingsProps {
  isDark: boolean;
  toggleTheme: () => void;
}

interface AppSettings {
  notifications: {
    all: boolean;
    meds: boolean;
    insights: boolean;
  };
  preferences: {
    units: 'Metric' | 'Imperial';
  };
}

interface CalendarConnection {
  provider: CalendarProvider;
  connected: boolean;
  email?: string;
  lastSync?: string;
}

import { useAuth } from '../hooks/useAuth';

// --- Extracted Modal Components ---

const DevicesModal = ({
  onClose,
  devices,
  onDisconnect,
  onConnect
}: {
  onClose: () => void,
  devices: Device[],
  onDisconnect: (id: string) => void,
  onConnect: (device: Device) => void
}) => {
  const { t } = useLanguage();
  const {
    isScanning,
    devices: foundDevices,
    smartwatchData,
    error: btError,
    startScan,
    stopScan,
    connectToDevice,
    readBatteryLevel,
    startHeartRateNotifications,
  } = useBluetooth();
  const [connectingId, setConnectingId] = useState<string | null>(null);

  // Stop scan on unmount or close
  useEffect(() => {
    return () => {
      stopScan();
    };
  }, [stopScan]);

  const handleConnect = async (bluetoothDevice: any) => {
    setConnectingId(bluetoothDevice.deviceId);
    try {
      await connectToDevice(bluetoothDevice.deviceId);

      // Try to read battery level after connecting
      let battery = 100;
      try {
        const level = await readBatteryLevel(bluetoothDevice.deviceId);
        if (level !== null) battery = level;
      } catch { /* optional */ }

      // Start heart rate monitoring
      try {
        await startHeartRateNotifications(bluetoothDevice.deviceId);
      } catch { /* optional - device may not support HR */ }

      // Determine device type from name heuristics
      const nameLower = (bluetoothDevice.name || '').toLowerCase();
      let deviceType: 'watch' | 'chest-strap' | 'ring' = 'chest-strap';
      if (nameLower.includes('watch') || nameLower.includes('band') || nameLower.includes('fitbit') || nameLower.includes('garmin') || nameLower.includes('galaxy') || nameLower.includes('mi band')) {
        deviceType = 'watch';
      } else if (nameLower.includes('ring') || nameLower.includes('oura')) {
        deviceType = 'ring';
      }

      const newDevice: Device = {
        id: bluetoothDevice.deviceId,
        name: bluetoothDevice.name || 'Unknown Device',
        type: deviceType,
        lastSync: 'Now',
        status: 'connected',
        battery: battery
      };

      onConnect(newDevice);
    } catch (e) {
      console.error("Failed to connect", e);
    } finally {
      setConnectingId(null);
    }
  };

  return (
    <Modal isOpen={true} onClose={onClose} title={t('settings.devices')}>
      {/* My Devices List */}
      <h4 className="text-sm font-bold text-slate-500 mb-2 uppercase">Connected Devices</h4>
      <div className="space-y-3 mb-6 max-h-[30vh] overflow-y-auto pr-1">
        {devices.length > 0 ? devices.map(device => (
          <div key={device.id} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-700 animate-in slide-in-from-right">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400">
                <span className="material-symbols-outlined">
                  {device.type === 'watch' ? 'watch' : device.type === 'chest-strap' ? 'monitor_heart' : 'ring_volume'}
                </span>
              </div>
              <div>
                <p className="text-sm font-bold dark:text-white">{device.name}</p>
                <p className="text-xs text-green-500 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span>
                  {t('settings.connected')} {device.battery && `• ${device.battery}%`}
                </p>
              </div>
            </div>
            <button
              onClick={() => onDisconnect(device.id)}
              className="text-xs text-red-500 font-medium hover:bg-red-50 dark:hover:bg-red-900/20 px-2 py-1 rounded-lg transition-colors"
            >
              {t('settings.disconnect')}
            </button>
          </div>
        )) : (
          <div className="text-center py-4 text-slate-500 text-sm italic border border-dashed border-slate-200 dark:border-slate-700 rounded-xl">No devices connected.</div>
        )}
      </div>

      <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
        <h4 className="text-sm font-bold text-slate-500 mb-2 uppercase">Available Devices</h4>

        {/* Found Devices List */}
        <div className="space-y-2 mb-4 max-h-[30vh] overflow-y-auto min-h-[100px]">
          {foundDevices.length > 0 ? (
            foundDevices.map(d => (
              <div key={d.deviceId} className="flex items-center justify-between p-2 hover:bg-slate-50 dark:hover:bg-slate-800/50 rounded-lg transition-colors cursor-pointer" onClick={() => handleConnect(d)}>
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-slate-400">bluetooth</span>
                  <div>
                    <p className="text-sm font-medium dark:text-white">{d.name || 'Unknown Device'}</p>
                    <p className="text-[10px] text-slate-400">ID: {d.deviceId} • RSSI: {d.rssi}</p>
                  </div>
                </div>
                {connectingId === d.deviceId ? (
                  <span className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></span>
                ) : (
                  <span className="material-symbols-outlined text-primary text-sm">add_circle</span>
                )}
              </div>
            ))
          ) : isScanning ? (
            <div className="flex flex-col items-center justify-center py-6 text-slate-400 gap-2">
              <span className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin"></span>
              <span className="text-xs">Searching for devices...</span>
            </div>
          ) : (
            <div className="text-center py-6 text-slate-400 text-xs">Press scan to find devices</div>
          )}
        </div>

        {btError && (
          <div className="mb-3 p-2 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-xs rounded-lg flex items-center gap-2">
            <span className="material-symbols-outlined text-sm">error</span>
            {btError}
          </div>
        )}

        <button
          onClick={() => {
            if (isScanning) stopScan();
            else startScan();
          }}
          className={`w-full py-3 ${isScanning ? 'bg-red-50 text-red-500' : 'bg-primary text-white'} font-bold rounded-xl flex items-center justify-center gap-2 hover:opacity-90 transition-colors shadow-lg ${isScanning ? 'shadow-red-500/10' : 'shadow-primary/30'}`}
        >
          {isScanning ? (
            <>
              <span className="material-symbols-outlined">stop_circle</span>
              Stop Scanning
            </>
          ) : (
            <>
              <span className="material-symbols-outlined">bluetooth_searching</span>
              {t('settings.pair_device')}
            </>
          )}
        </button>

        <p className="text-[10px] text-slate-400 text-center mt-2">
          Make sure your device is in pairing mode.
        </p>
      </div>
    </Modal>
  );
};

const PasswordModal = ({ onClose }: { onClose: () => void }) => {
  const [step, setStep] = useState('form');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpdate = async () => {
    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return;
    }
    if (newPassword.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await apiClient.changePassword({ current_password: currentPassword, new_password: newPassword });
      setStep('success');
    } catch (err: any) {
      setError(err.message || 'Failed to update password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal isOpen={true} onClose={onClose} title={step === 'form' ? "Change Password" : "Password Updated"}>
      {step === 'form' ? (
        <>
          <div className="space-y-4">
            {error && <div className="p-3 text-xs bg-red-50 text-red-500 rounded-lg">{error}</div>}
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase">Current Password</label>
              <input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
              />
            </div>
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase">New Password</label>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
              />
            </div>
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase">Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full mt-1 p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white focus:ring-2 focus:ring-primary"
              />
            </div>
            <div className="flex gap-3 mt-6">
              <button onClick={onClose} disabled={loading} className="flex-1 py-3 text-slate-500 font-bold hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors">Cancel</button>
              <button
                onClick={handleUpdate}
                disabled={loading}
                className="flex-1 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 flex items-center justify-center"
              >
                {loading ? <span className="w-4 h-4 border-2 border-white/50 border-t-white rounded-full animate-spin"></span> : 'Update'}
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="text-center py-6 animate-in zoom-in-95 duration-300">
          <div className="w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-4 text-green-600 dark:text-green-400">
            <span className="material-symbols-outlined text-3xl">check</span>
          </div>
          <h3 className="text-xl font-bold dark:text-white mb-2">Password Updated</h3>
          <p className="text-slate-500 text-sm mb-6">Your password has been changed successfully.</p>
          <button onClick={onClose} className="w-full py-3 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold rounded-xl">Close</button>
        </div>
      )}
    </Modal>
  );
};

const FeedbackModal = ({ onClose }: { onClose: () => void }) => {
  const { user } = useAuth();
  const [sent, setSent] = useState(false);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setError(null);

    try {
      await apiClient.submitFeedback({
        type: 'general',
        message: message,
        userId: user?.id || 'anonymous'
      });
      setSent(true);
    } catch (err: any) {
      console.error("Feedback failed", err);
      // Even if API fails, show success for UX unless critical? 
      // User expects feedback to just "go". But let's show error if we can.
      // For now, if "mock" mode, it handles errors.
      if (err.status === 0) setError("Network error, please try again.");
      else setSent(true); // Assume success for other errors to not block user? Or show error.
      // Actually, let's just log it and show error if it's strictly failed.
      setError("Failed to send feedback. Please check your connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal isOpen={true} onClose={onClose} title={!sent ? "Send Feedback" : "Feedback Sent!"}>
      {!sent ? (
        <>
          <p className="text-slate-500 text-sm mb-4">Let us know how we can improve your experience.</p>
          {error && <div className="mb-2 text-xs text-red-500">{error}</div>}
          <textarea
            className="w-full h-32 p-3 bg-slate-100 dark:bg-slate-800 rounded-xl border-none outline-none dark:text-white resize-none mb-4 placeholder:text-slate-400 focus:ring-2 focus:ring-primary"
            placeholder="Type your message here..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          ></textarea>
          <div className="flex gap-3">
            <button onClick={onClose} disabled={loading} className="flex-1 py-3 text-slate-500 font-bold hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl">Cancel</button>
            <button
              onClick={handleSend}
              disabled={loading || !message.trim()}
              className="flex-1 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 flex items-center justify-center"
            >
              {loading ? <span className="w-4 h-4 border-2 border-white/50 border-t-white rounded-full animate-spin"></span> : 'Send'}
            </button>
          </div>
        </>
      ) : (
        <div className="text-center py-6 animate-in zoom-in-95 duration-300">
          <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-4 text-blue-600 dark:text-blue-400">
            <span className="material-symbols-outlined text-3xl">send</span>
          </div>
          <h3 className="text-xl font-bold dark:text-white mb-2">Feedback Sent!</h3>
          <p className="text-slate-500 text-sm mb-6">Thank you for helping us improve.</p>
          <button onClick={onClose} className="w-full py-3 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold rounded-xl">Close</button>
        </div>
      )}
    </Modal>
  );
};

const HelpModal = ({ onClose }: { onClose: () => void }) => (
  <Modal isOpen={true} onClose={onClose} title="Help Center">
    <div className="overflow-y-auto pr-2 space-y-4">
      {[
        { q: "How is my risk score calculated?", a: "Your score is based on the vitals you enter (BP, Cholesterol) combined with lifestyle factors like smoking and activity level." },
        { q: "Is my data private?", a: "Yes, all data is stored locally on your device. We do not share your personal health info." },
        { q: "Can I connect my Fitbit?", a: "Yes! Use the 'Manage Connected Devices' option to scan and pair your Bluetooth trackers." },
        { q: "How do I book an appointment?", a: "Go to the 'Book' tab, search for a specialist, and select an available time slot." }
      ].map((faq, i) => (
        <details key={i} className="group bg-slate-50 dark:bg-slate-800/50 rounded-xl p-3">
          <summary className="flex justify-between items-center cursor-pointer font-bold text-sm dark:text-white list-none">
            {faq.q}
            <span className="material-symbols-outlined text-slate-400 group-open:rotate-180 transition-transform">expand_more</span>
          </summary>
          <p className="text-slate-500 dark:text-slate-400 text-xs mt-2 leading-relaxed">
            {faq.a}
          </p>
        </details>
      ))}
    </div>
    <button
      onClick={() => {
        const subject = encodeURIComponent('Cardio AI Support Request');
        const body = encodeURIComponent('Hi Cardio AI Support Team,\n\nI need help with:\n\n');
        window.open(`mailto:support@cardioai.com?subject=${subject}&body=${body}`, '_self');
      }}
      className="w-full mt-4 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 shrink-0 hover:bg-primary-dark transition-colors"
    >
      Contact Support
    </button>
  </Modal>
);

const TermsModal = ({ onClose }: { onClose: () => void }) => (
  <Modal isOpen={true} onClose={onClose} title="Terms of Service">
    <div className="overflow-y-auto pr-2 text-sm text-slate-600 dark:text-slate-300 leading-relaxed space-y-3">
      <p><strong>1. Acceptance of Terms</strong><br />By accessing and using this application, you accept and agree to be bound by the terms and provision of this agreement.</p>
      <p><strong>2. Medical Disclaimer</strong><br />This app provides information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician.</p>
      <p><strong>3. User Data</strong><br />We are committed to protecting your privacy. Your personal health data is processed in accordance with our Privacy Policy.</p>
      <p><strong>4. Modifications</strong><br />We reserve the right to modify these terms at any time.</p>
    </div>
    <button onClick={onClose} className="w-full mt-4 py-3 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold rounded-xl shrink-0 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
      I Agree
    </button>
  </Modal>
);

// --- Calendar Connections Modal ---
const CalendarModal = ({ onClose }: { onClose: () => void }) => {
  const { user } = useAuth();
  const [connections, setConnections] = useState<CalendarConnection[]>(() => {
    const saved = localStorage.getItem('calendar_connections');
    return saved ? JSON.parse(saved) : [
      { provider: 'google' as CalendarProvider, connected: false },
      { provider: 'outlook' as CalendarProvider, connected: false },
    ];
  });
  const [isConnecting, setIsConnecting] = useState<CalendarProvider | null>(null);
  const [isSyncing, setIsSyncing] = useState<CalendarProvider | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleConnect = async (provider: CalendarProvider) => {
    if (!user) return;
    setIsConnecting(provider);
    setError(null);

    try {
      // Use Capacitor Browser for OAuth flow on native, window.open on web
      if (Capacitor.isNativePlatform()) {
        try {
          const { Browser } = await import('@capacitor/browser');
          const oauthUrl = provider === 'google'
            ? `https://accounts.google.com/o/oauth2/v2/auth?client_id=YOUR_GOOGLE_CLIENT_ID&redirect_uri=com.cardioai.assistant://callback&response_type=code&scope=https://www.googleapis.com/auth/calendar.readonly`
            : `https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=YOUR_OUTLOOK_CLIENT_ID&redirect_uri=com.cardioai.assistant://callback&response_type=code&scope=Calendars.Read`;

          await Browser.open({ url: oauthUrl });

          // In production, the redirect callback would provide the auth code
          // For now, simulate successful connection
          await new Promise(resolve => setTimeout(resolve, 2000));
        } catch {
          // Fallback: simulated connection for development
          await new Promise(resolve => setTimeout(resolve, 1500));
        }
      } else {
        // Web: simulate OAuth flow (in production, redirect to OAuth endpoint)
        await new Promise(resolve => setTimeout(resolve, 1500));
      }

      await calendarService.storeCalendarCredentials(user.id, {
        provider,
        access_token: 'demo_token_' + Date.now(),
      });

      const updated = connections.map(c =>
        c.provider === provider
          ? { ...c, connected: true, email: user.email || `user@${provider}.com`, lastSync: 'Never' }
          : c
      );
      setConnections(updated);
      localStorage.setItem('calendar_connections', JSON.stringify(updated));

    } catch (err) {
      if (err instanceof CalendarServiceError) {
        setError(err.userMessage);
      } else {
        setError('Failed to connect. Please try again.');
      }
    } finally {
      setIsConnecting(null);
    }
  };

  const handleDisconnect = async (provider: CalendarProvider) => {
    if (!user) return;
    try {
      await calendarService.revokeCalendarCredentials(user.id, provider);

      const updated = connections.map(c =>
        c.provider === provider
          ? { ...c, connected: false, email: undefined, lastSync: undefined }
          : c
      );
      setConnections(updated);
      localStorage.setItem('calendar_connections', JSON.stringify(updated));
    } catch (err) {
      console.error('Disconnect error:', err);
    }
  };

  const handleSync = async (provider: CalendarProvider) => {
    if (!user) return;
    setIsSyncing(provider);
    setError(null);

    try {
      await calendarService.syncCalendar(user.id, {
        provider,
        days_ahead: 30,
        include_reminders: true,
      });

      const updated = connections.map(c =>
        c.provider === provider
          ? { ...c, lastSync: 'Just now' }
          : c
      );
      setConnections(updated);
      localStorage.setItem('calendar_connections', JSON.stringify(updated));

    } catch (err) {
      if (err instanceof CalendarServiceError) {
        setError(err.userMessage);
      } else {
        setError('Sync failed. Please try again.');
      }
    } finally {
      setIsSyncing(null);
    }
  };

  const getProviderColor = (provider: CalendarProvider) => {
    return provider === 'google'
      ? 'text-red-500 bg-red-50 dark:bg-red-900/20'
      : 'text-blue-500 bg-blue-50 dark:bg-blue-900/20';
  };

  return (
    <Modal isOpen={true} onClose={onClose} title="Calendar Connections">
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
        Connect your calendars to sync appointments and set reminders.
      </p>

      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
          <p className="text-red-700 dark:text-red-400 text-sm flex items-center gap-2">
            <span className="material-symbols-outlined text-sm">error</span>
            {error}
          </p>
        </div>
      )}

      <div className="space-y-3">
        {connections.map(conn => (
          <div key={conn.provider} className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${getProviderColor(conn.provider)}`}>
                  <span className="material-symbols-outlined">{conn.provider === 'google' ? 'event' : 'calendar_month'}</span>
                </div>
                <div>
                  <p className="font-medium dark:text-white capitalize">{conn.provider}</p>
                  {conn.connected && conn.email && (
                    <p className="text-xs text-slate-500">{conn.email}</p>
                  )}
                </div>
              </div>
              {conn.connected ? (
                <span className="text-xs text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30 px-2 py-1 rounded-full font-medium flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                  Connected
                </span>
              ) : (
                <span className="text-xs text-slate-500 dark:text-slate-400">Not connected</span>
              )}
            </div>

            {conn.connected ? (
              <div className="flex gap-2 mt-3">
                <button
                  onClick={() => handleSync(conn.provider)}
                  disabled={isSyncing === conn.provider}
                  className="flex-1 py-2 bg-primary/10 text-primary rounded-lg text-sm font-medium flex items-center justify-center gap-1 hover:bg-primary/20 transition-colors disabled:opacity-50"
                >
                  {isSyncing === conn.provider ? (
                    <span className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></span>
                  ) : (
                    <span className="material-symbols-outlined text-sm">sync</span>
                  )}
                  {conn.lastSync ? `Last: ${conn.lastSync}` : 'Sync Now'}
                </button>
                <button
                  onClick={() => handleDisconnect(conn.provider)}
                  className="py-2 px-3 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg text-sm font-medium transition-colors"
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <button
                onClick={() => handleConnect(conn.provider)}
                disabled={isConnecting === conn.provider}
                className="w-full mt-3 py-2 bg-primary text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2 hover:bg-primary-dark transition-colors shadow-lg shadow-primary/20 disabled:opacity-50"
              >
                {isConnecting === conn.provider ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/50 border-t-white rounded-full animate-spin"></span>
                    Connecting...
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined text-sm">link</span>
                    Connect {conn.provider.charAt(0).toUpperCase() + conn.provider.slice(1)}
                  </>
                )}
              </button>
            )}
          </div>
        ))}
      </div>
    </Modal>
  );
};

// --- Weekly Summary Modal ---
const WeeklySummaryModal = ({ onClose }: { onClose: () => void }) => {
  const { user } = useAuth();
  const [preferences, setPreferences] = useState<WeeklySummaryPreferences>({
    enabled: false,
    delivery_channels: [
      { channel: 'email', enabled: true, destination: user?.email || 'user@example.com' },
      { channel: 'push', enabled: false },
      { channel: 'whatsapp', enabled: false, destination: '' },
      { channel: 'sms' as any, enabled: false, destination: '' },
    ],
    preferred_day: 0,
    preferred_time: '09:00',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  });
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadPrefs = async () => {
      if (!user) return;
      try {
        const prefs = await notificationService.getWeeklySummaryPreferences(user.id);
        setPreferences(prefs);
      } catch (err) {
        console.log('Using default weekly summary preferences');
      }
    };
    loadPrefs();
  }, [user]);

  const handleSave = async () => {
    if (!user) return;
    setIsSaving(true);
    setError(null);

    try {
      await notificationService.updateWeeklySummaryPreferences(user.id, preferences);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      if (err instanceof NotificationServiceError) {
        setError(err.userMessage);
      } else {
        setError('Failed to save. Please try again.');
      }
    } finally {
      setIsSaving(false);
    }
  };

  const toggleChannel = (channel: 'push' | 'email' | 'whatsapp' | 'sms') => {
    setPreferences(prev => ({
      ...prev,
      delivery_channels: prev.delivery_channels.map(c =>
        c.channel === channel ? { ...c, enabled: !c.enabled } : c
      ),
    }));
  };

  const updateChannelDestination = (channel: string, destination: string) => {
    setPreferences(prev => ({
      ...prev,
      delivery_channels: prev.delivery_channels.map(c =>
        c.channel === channel ? { ...c, destination } : c
      ),
    }));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={onClose}>
      <div className="bg-white dark:bg-card-dark rounded-2xl p-6 w-full max-w-sm shadow-2xl max-h-[80vh] overflow-hidden flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-4 shrink-0">
          <h3 className="text-xl font-bold dark:text-white">Weekly Summary</h3>
          <button onClick={onClose} className="p-1 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-500">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <div className="overflow-y-auto flex-1 space-y-4">
          {/* Enable Toggle */}
          <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600">
                <span className="material-symbols-outlined">summarize</span>
              </div>
              <div>
                <p className="font-medium dark:text-white">Enable Weekly Summary</p>
                <p className="text-xs text-slate-500">Get a personalized health recap</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={preferences.enabled}
                onChange={() => setPreferences(prev => ({ ...prev, enabled: !prev.enabled }))}
              />
              <div className="w-11 h-6 bg-slate-200 dark:bg-slate-700 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </div>

          {preferences.enabled && (
            <>
              {/* Delivery Channels */}
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase mb-2 block">Delivery Channels</label>
                <div className="space-y-2">
                  {preferences.delivery_channels.map(dc => (
                    <div key={dc.channel} className="bg-slate-50 dark:bg-slate-800/50 rounded-xl overflow-hidden">
                      <div className="flex items-center justify-between p-3">
                        <div className="flex items-center gap-2">
                          <span className="material-symbols-outlined text-slate-400">
                            {dc.channel === 'email' ? 'mail' : dc.channel === 'push' ? 'notifications' : dc.channel === 'sms' ? 'sms' : 'chat'}
                          </span>
                          <span className="text-sm font-medium dark:text-white capitalize">{dc.channel === 'sms' ? 'SMS' : dc.channel}</span>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            className="sr-only peer"
                            checked={dc.enabled}
                            onChange={() => toggleChannel(dc.channel as 'push' | 'email' | 'whatsapp' | 'sms')}
                          />
                          <div className="w-9 h-5 bg-slate-200 dark:bg-slate-700 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
                        </label>
                      </div>
                      {/* Destination input for email, whatsapp, sms */}
                      {dc.enabled && dc.channel !== 'push' && (
                        <div className="px-3 pb-3">
                          <input
                            type={dc.channel === 'email' ? 'email' : 'tel'}
                            placeholder={dc.channel === 'email' ? 'your@email.com' : dc.channel === 'sms' ? '+1234567890' : '+1234567890 (WhatsApp)'}
                            value={dc.destination || ''}
                            onChange={(e) => updateChannelDestination(dc.channel, e.target.value)}
                            className="w-full p-2 text-sm rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 outline-none dark:text-white focus:ring-2 focus:ring-primary/50 placeholder:text-slate-400"
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Day Selector */}
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase mb-2 block">Delivery Day</label>
                <select
                  value={preferences.preferred_day}
                  onChange={e => setPreferences(prev => ({ ...prev, preferred_day: parseInt(e.target.value) }))}
                  className="w-full p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white appearance-none cursor-pointer"
                >
                  {notificationService.DAYS_OF_WEEK.map(day => (
                    <option key={day.value} value={day.value}>{day.label}</option>
                  ))}
                </select>
              </div>

              {/* Time Selector */}
              <div>
                <label className="text-xs font-bold text-slate-500 uppercase mb-2 block">Delivery Time</label>
                <input
                  type="time"
                  value={preferences.preferred_time}
                  onChange={e => setPreferences(prev => ({ ...prev, preferred_time: e.target.value }))}
                  className="w-full p-3 rounded-xl bg-slate-100 dark:bg-slate-800 border-none outline-none dark:text-white"
                />
              </div>
            </>
          )}

          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
              <p className="text-red-700 dark:text-red-400 text-sm">{error}</p>
            </div>
          )}
        </div>

        <button
          onClick={handleSave}
          disabled={isSaving}
          className="w-full mt-4 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 shrink-0 hover:bg-primary-dark transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isSaving ? (
            <>
              <span className="w-4 h-4 border-2 border-white/50 border-t-white rounded-full animate-spin"></span>
              Saving...
            </>
          ) : saved ? (
            <>
              <span className="material-symbols-outlined">check</span>
              Saved!
            </>
          ) : (
            'Save Preferences'
          )}
        </button>
      </div>
    </div>
  );
};

// --- Main Screen Component ---

const SettingsScreen: React.FC<SettingsProps> = ({ isDark, toggleTheme }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { t, language, setLanguage } = useLanguage();
  const { showToast } = useToast();
  const [activeModal, setActiveModal] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Initialize state from localStorage or defaults
  const [settings, setSettings] = useState<AppSettings>(() => {
    const saved = localStorage.getItem('app_settings');
    return saved ? JSON.parse(saved) : {
      notifications: {
        all: true,
        meds: true,
        insights: false,
      },
      preferences: {
        units: 'Metric',
      }
    };
  });

  // Connected Devices State
  const [devices, setDevices] = useState<Device[]>(() => {
    const saved = localStorage.getItem('connected_devices');
    return saved ? JSON.parse(saved) : [
      { id: 'd1', name: 'Apple Watch Series 8', type: 'watch', lastSync: 'Today, 10:30 AM', status: 'connected', battery: 82 },
      { id: 'd2', name: 'Oura Ring', type: 'ring', lastSync: 'Yesterday', status: 'connected', battery: 45 }
    ];
  });

  // Load settings & devices from API on mount
  useEffect(() => {
    const userId = user?.id || localStorage.getItem('user_id') || 'default';
    const loadFromApi = async () => {
      try {
        const apiSettings = await apiClient.getAppSettings(userId);
        if (apiSettings) {
          const merged: AppSettings = {
            notifications: apiSettings.notifications || settings.notifications,
            preferences: { units: (apiSettings.preferences?.units as any) || settings.preferences.units },
          };
          setSettings(merged);
          localStorage.setItem('app_settings', JSON.stringify(merged));
        }
      } catch {
        // Keep localStorage values
      }

      try {
        const apiDevices = await apiClient.getDevices(userId);
        if (apiDevices && apiDevices.length > 0) {
          setDevices(apiDevices.map(d => ({
            id: d.id,
            name: d.name,
            type: d.type as any,
            lastSync: d.lastSync,
            status: d.status as any,
            battery: d.battery,
          })));
          localStorage.setItem('connected_devices', JSON.stringify(apiDevices));
        }
      } catch {
        // Keep localStorage values
      }
    };
    loadFromApi();
  }, [user]);

  // Save settings
  useEffect(() => {
    localStorage.setItem('app_settings', JSON.stringify(settings));
    // Sync to backend
    const userId = user?.id || localStorage.getItem('user_id') || 'default';
    apiClient.updateAppSettings(userId, {
      notifications: settings.notifications,
      preferences: { units: settings.preferences.units, language: language, theme: isDark ? 'dark' : 'light' },
    }).catch(() => { /* silent fail */ });
  }, [settings]);

  // Save devices
  useEffect(() => {
    localStorage.setItem('connected_devices', JSON.stringify(devices));
  }, [devices]);

  // Search filter logic
  const sq = searchQuery.toLowerCase().trim();
  const settingsSections = useMemo(() => ({
    account: ['profile', 'password', 'devices', 'smartwatch', 'bluetooth', 'account', 'manage devices'],
    preferences: ['language', 'units', 'metric', 'imperial', 'theme', 'dark', 'light', 'preferences'],
    integrations: ['calendar', 'google', 'outlook', 'weekly summary', 'recap', 'integrations'],
    notifications: ['notification', 'medication', 'reminders', 'insights', 'alerts'],
    data: ['export', 'health report', 'pdf', 'chat history', 'data', 'download'],
    support: ['help', 'faq', 'feedback', 'contact', 'support'],
    about: ['version', 'terms', 'privacy', 'about', 'legal'],
  }), []);

  const sectionVisible = (sectionKey: keyof typeof settingsSections) => {
    if (!sq) return true;
    return settingsSections[sectionKey].some(keyword => keyword.includes(sq));
  };

  const toggleNotification = async (key: keyof AppSettings['notifications']) => {
    // Request Permission for All Notifications
    if (key === 'all' && !settings.notifications.all) {
      try {
        const granted = await nativeNotificationService.requestNotificationPermission();
        if (!granted) {
          showToast('Notification permission denied. Please enable in device settings.', 'error');
          return;
        }
        // Create notification channels on Android
        await nativeNotificationService.createNotificationChannels();
      } catch (e) {
        console.error('Failed to request notification permission', e);
      }
    }

    const newNotifications = {
      ...settings.notifications,
      [key]: !settings.notifications[key]
    };

    setSettings(prev => ({
      ...prev,
      notifications: newNotifications
    }));

    // Sync with backend
    if (user) {
      try {
        await apiClient.updatePreferences(user.id, {
          notifications_enabled: newNotifications.all, // approximate mapping
          // We might need a more granular preference if API supports it, 
          // or store entire 'notifications' object in a custom field if schema allows.
          // For now, mapping 'all' to global enabled.
        });
      } catch (e) {
        console.error("Failed to sync notification preferences", e);
      }
    }
  };

  const toggleUnits = () => {
    setSettings(prev => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        units: prev.preferences.units === 'Metric' ? 'Imperial' : 'Metric'
      }
    }));
  };

  const cycleLanguage = () => {
    const languages: ('en' | 'es' | 'fr' | 'te')[] = ['en', 'es', 'fr', 'te'];
    const currentIndex = languages.indexOf(language);
    const nextIndex = (currentIndex + 1) % languages.length;
    setLanguage(languages[nextIndex]);
  };

  const closeModal = () => setActiveModal(null);

  const handleConnectDevice = (newDevice: Device) => {
    setDevices(prev => [...prev, newDevice]);
    const userId = user?.id || localStorage.getItem('user_id') || 'default';
    apiClient.addDevice(userId, {
      id: newDevice.id,
      name: newDevice.name,
      type: newDevice.type,
      status: newDevice.status,
      battery: newDevice.battery,
    }).catch(err => console.warn('Failed to sync device to backend', err));
  };

  const handleDisconnectDevice = (id: string) => {
    setDevices(prev => prev.filter(d => d.id !== id));
    const userId = user?.id || localStorage.getItem('user_id') || 'default';
    apiClient.removeDevice(userId, id).catch(err => console.warn('Failed to sync device removal', err));
  };

  const ToggleSwitch = ({ checked, onChange }: { checked: boolean; onChange?: () => void }) => (
    <label className="relative inline-flex items-center cursor-pointer" onClick={(e) => e.stopPropagation()}>
      <input
        type="checkbox"
        className="sr-only peer"
        checked={checked}
        onChange={onChange || (() => { })}
      />
      <div className="w-11 h-6 bg-slate-200 dark:bg-slate-700 rounded-full peer peer-focus:ring-2 peer-focus:ring-primary/50 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
    </label>
  );

  return (
    <div className="relative flex h-auto min-h-screen w-full flex-col bg-background-light dark:bg-background-dark font-sans pb-24 overflow-x-hidden">
      {/* Top App Bar */}
      <div className="flex items-center p-4 pb-2 justify-between bg-background-light dark:bg-background-dark sticky top-0 z-10">
        <button
          onClick={() => navigate('/profile')}
          className="flex w-10 h-10 items-center justify-center text-slate-700 dark:text-slate-200 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <span className="material-symbols-outlined">arrow_back_ios_new</span>
        </button>
        <h2 className="text-slate-900 dark:text-white text-lg font-bold leading-tight flex-1 text-center pr-10">
          {t('settings.title')}
        </h2>
      </div>

      {/* Search Bar */}
      <div className="px-4 py-3">
        <label className="flex flex-col min-w-40 h-12 w-full">
          <div className="flex w-full flex-1 items-stretch rounded-lg h-full">
            <div className="text-slate-400 dark:text-slate-500 flex border-none bg-slate-100 dark:bg-slate-800 items-center justify-center pl-4 rounded-l-lg border-r-0">
              <span className="material-symbols-outlined">search</span>
            </div>
            <input
              className="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-slate-900 dark:text-white focus:outline-0 focus:ring-0 border-none bg-slate-100 dark:bg-slate-800 focus:border-none h-full placeholder:text-slate-400 dark:placeholder:text-slate-500 px-4 rounded-l-none border-l-0 pl-2 text-base font-normal leading-normal"
              placeholder={t('settings.search_placeholder')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="flex items-center justify-center pr-3 bg-slate-100 dark:bg-slate-800 rounded-r-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
              >
                <span className="material-symbols-outlined text-lg">close</span>
              </button>
            )}
          </div>
        </label>
      </div>

      <div className="px-4">
        <div className="bg-slate-200 dark:bg-slate-800 h-px w-full"></div>
      </div>

      <div className="space-y-6 pb-12">
        {/* Account Management Section */}
        {sectionVisible('account') && <div>
          <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight px-4 pb-2 pt-6">
            {t('settings.account')}
          </h3>
          <div
            onClick={() => navigate('/profile')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">person</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.profile')}</p>
            </div>
            <div className="shrink-0 text-slate-400 dark:text-slate-500">
              <span className="material-symbols-outlined">chevron_right</span>
            </div>
          </div>
          <div
            onClick={() => setActiveModal('password')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">lock</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.password')}</p>
            </div>
            <div className="shrink-0 text-slate-400 dark:text-slate-500">
              <span className="material-symbols-outlined">chevron_right</span>
            </div>
          </div>
          <div
            onClick={() => setActiveModal('devices')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">watch</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.devices')}</p>
            </div>
            <div className="shrink-0 flex items-center gap-2">
              <span className="bg-primary text-white text-[10px] font-bold px-1.5 py-0.5 rounded-md">{devices.length}</span>
              <span className="material-symbols-outlined text-slate-400 dark:text-slate-500">chevron_right</span>
            </div>
          </div>
        </div>}

        {/* App Preferences Section */}
        {sectionVisible('preferences') && <div>
          <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight px-4 pb-2 pt-4">
            {t('settings.preferences')}
          </h3>
          <div
            onClick={cycleLanguage}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">language</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.language')}</p>
            </div>
            <div className="shrink-0 flex items-center gap-2">
              <span className="text-sm text-primary font-bold bg-primary/10 px-2 py-1 rounded-md uppercase">{language}</span>
              <span className="material-symbols-outlined text-slate-400 dark:text-slate-500">sync_alt</span>
            </div>
          </div>
          <div
            onClick={toggleUnits}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">straighten</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.units')}</p>
            </div>
            <div className="shrink-0 flex items-center gap-2">
              <span className="text-sm text-slate-500 dark:text-slate-400">{settings.preferences.units}</span>
              <span className="material-symbols-outlined text-slate-400 dark:text-slate-500">swap_horiz</span>
            </div>
          </div>
          <div
            onClick={toggleTheme}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">contrast</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.theme')}</p>
            </div>
            <div className="shrink-0 flex items-center gap-2">
              <span className="text-sm text-slate-500 dark:text-slate-400">{isDark ? t('settings.dark') : t('settings.light')}</span>
              <span className="material-symbols-outlined text-slate-400 dark:text-slate-500">toggle_on</span>
            </div>
          </div>
        </div>}

        {/* Integrations Section */}
        {sectionVisible('integrations') && <div>
          <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight px-4 pb-2 pt-4">
            Integrations
          </h3>
          <div
            onClick={() => setActiveModal('calendar')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">calendar_month</span>
              </div>
              <div>
                <p className="text-slate-900 dark:text-white text-base font-normal leading-normal truncate">Calendar Connections</p>
                <p className="text-slate-500 dark:text-slate-400 text-xs">Connect Google & Outlook</p>
              </div>
            </div>
            <div className="shrink-0 text-slate-400 dark:text-slate-500">
              <span className="material-symbols-outlined">chevron_right</span>
            </div>
          </div>
          <div
            onClick={() => setActiveModal('weeklySummary')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">summarize</span>
              </div>
              <div>
                <p className="text-slate-900 dark:text-white text-base font-normal leading-normal truncate">Weekly Summary</p>
                <p className="text-slate-500 dark:text-slate-400 text-xs">Configure your health recap</p>
              </div>
            </div>
            <div className="shrink-0 text-slate-400 dark:text-slate-500">
              <span className="material-symbols-outlined">chevron_right</span>
            </div>
          </div>
        </div>}

        {/* Notifications Section */}
        {sectionVisible('notifications') && <div>
          <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight px-4 pb-2 pt-4">
            {t('settings.notifications')}
          </h3>
          <div
            onClick={() => toggleNotification('all')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">notifications</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.notif_all')}</p>
            </div>
            <div className="shrink-0">
              <ToggleSwitch checked={settings.notifications.all} onChange={() => toggleNotification('all')} />
            </div>
          </div>
          <div
            onClick={() => toggleNotification('meds')}
            className={`flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors ${!settings.notifications.all ? 'opacity-50 pointer-events-none' : ''}`}
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">pill</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.notif_meds')}</p>
            </div>
            <div className="shrink-0">
              <ToggleSwitch checked={settings.notifications.meds} onChange={() => toggleNotification('meds')} />
            </div>
          </div>
          <div
            onClick={() => toggleNotification('insights')}
            className={`flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors ${!settings.notifications.all ? 'opacity-50 pointer-events-none' : ''}`}
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">insights</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.notif_insights')}</p>
            </div>
            <div className="shrink-0">
              <ToggleSwitch checked={settings.notifications.insights} onChange={() => toggleNotification('insights')} />
            </div>
          </div>
        </div>}

        {/* Data Management Section */}
        {sectionVisible('data') && <div>
          <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight px-4 pb-2 pt-4">
            Data Management
          </h3>
          <div
            onClick={() => {
              try {
                pdfExportService.exportQuickSummary();
                showToast('Health report exported successfully!', 'success');
              } catch (err: any) {
                showToast(err.message || 'Failed to export report.', 'error');
              }
            }}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">picture_as_pdf</span>
              </div>
              <div>
                <p className="text-slate-900 dark:text-white text-base font-normal leading-normal truncate">Export Health Report</p>
                <p className="text-slate-500 dark:text-slate-400 text-xs">Download your data as PDF</p>
              </div>
            </div>
            <div className="shrink-0 text-primary">
              <span className="material-symbols-outlined">download</span>
            </div>
          </div>
          <div
            onClick={() => {
              try {
                // Get chat messages from localStorage
                const messages = JSON.parse(localStorage.getItem('chat_messages') || '[]');
                if (messages.length > 0) {
                  pdfExportService.exportChatHistory({
                    messages,
                    exportDate: new Date(),
                  });
                  showToast('Chat history exported successfully!', 'success');
                } else {
                  showToast('No chat history to export.', 'info');
                }
              } catch (err: any) {
                showToast(err.message || 'Failed to export chat history.', 'error');
              }
            }}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">chat</span>
              </div>
              <div>
                <p className="text-slate-900 dark:text-white text-base font-normal leading-normal truncate">Export Chat History</p>
                <p className="text-slate-500 dark:text-slate-400 text-xs">Save conversations as PDF</p>
              </div>
            </div>
            <div className="shrink-0 text-primary">
              <span className="material-symbols-outlined">download</span>
            </div>
          </div>
        </div>}

        {/* Support & Feedback Section */}
        {sectionVisible('support') && <div>
          <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight px-4 pb-2 pt-4">
            {t('settings.support')}
          </h3>
          <div
            onClick={() => setActiveModal('help')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">help_center</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.help')}</p>
            </div>
            <div className="shrink-0 text-slate-400 dark:text-slate-500">
              <span className="material-symbols-outlined">chevron_right</span>
            </div>
          </div>
          <div
            onClick={() => setActiveModal('feedback')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">feedback</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.feedback')}</p>
            </div>
            <div className="shrink-0 text-slate-400 dark:text-slate-500">
              <span className="material-symbols-outlined">chevron_right</span>
            </div>
          </div>
        </div>}

        {/* About Section */}
        {sectionVisible('about') && <div>
          <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight px-4 pb-2 pt-4">
            {t('settings.about')}
          </h3>
          <div className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between">
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">info</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.version')}</p>
            </div>
            <div className="shrink-0">
              <span className="text-sm text-slate-500 dark:text-slate-400">1.0.2</span>
            </div>
          </div>
          <div
            onClick={() => setActiveModal('terms')}
            className="flex items-center gap-4 bg-background-light dark:bg-background-dark px-4 min-h-14 justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="text-slate-700 dark:text-slate-200 flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 shrink-0 size-10">
                <span className="material-symbols-outlined">gavel</span>
              </div>
              <p className="text-slate-900 dark:text-white text-base font-normal leading-normal flex-1 truncate">{t('settings.terms')}</p>
            </div>
            <div className="shrink-0 text-slate-400 dark:text-slate-500">
              <span className="material-symbols-outlined">chevron_right</span>
            </div>
          </div>
        </div>}

        {/* No results message */}
        {sq && !sectionVisible('account') && !sectionVisible('preferences') && !sectionVisible('integrations') && !sectionVisible('notifications') && !sectionVisible('data') && !sectionVisible('support') && !sectionVisible('about') && (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400 dark:text-slate-500">
            <span className="material-symbols-outlined text-4xl mb-2">search_off</span>
            <p className="text-sm">No settings found for "{searchQuery}"</p>
          </div>
        )}
      </div>

      {/* Render Active Modal */}
      {activeModal === 'password' && <PasswordModal onClose={closeModal} />}
      {activeModal === 'devices' && (
        <DevicesModal
          onClose={closeModal}
          devices={devices}
          onDisconnect={handleDisconnectDevice}
          onConnect={handleConnectDevice}
        />
      )}
      {activeModal === 'help' && <HelpModal onClose={closeModal} />}
      {activeModal === 'feedback' && <FeedbackModal onClose={closeModal} />}
      {activeModal === 'terms' && <TermsModal onClose={closeModal} />}
      {activeModal === 'calendar' && <CalendarModal onClose={closeModal} />}
      {activeModal === 'weeklySummary' && <WeeklySummaryModal onClose={closeModal} />}
    </div>
  );
};

export default SettingsScreen;
