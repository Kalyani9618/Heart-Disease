import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';
import { useAuth } from '../hooks/useAuth';
import { apiClient } from '../services/apiClient';
import { useToast } from '../components/Toast';
import { Modal } from '../components/Modal';

interface AgentSettings {
    model: 'gemini-pro' | 'gpt-4' | 'claude-3';
    persona: 'medical' | 'friendly' | 'concise';
    responseLength: 'short' | 'medium' | 'long';
    voice: 'male' | 'female';
    temperature: number;
}

const DEFAULT_SETTINGS: AgentSettings = {
    model: 'gemini-pro',
    persona: 'medical',
    responseLength: 'medium',
    voice: 'female',
    temperature: 0.7
};

const AgentSettingsScreen: React.FC = () => {
    const navigate = useNavigate();
    const { t } = useLanguage();
    const { user } = useAuth();

    const [loading, setLoading] = useState(true);
    const [settings, setSettings] = useState<AgentSettings>(() => {
        const saved = localStorage.getItem('agent_settings');
        return saved ? JSON.parse(saved) : DEFAULT_SETTINGS;
    });

    const [saved, setSaved] = useState(false);
    const [showResetConfirm, setShowResetConfirm] = useState(false);
    const { showToast } = useToast();

    // Load settings from backend on mount
    useEffect(() => {
        const loadSettings = async () => {
            if (!user) {
                setLoading(false);
                return;
            }
            try {
                const prefs = await apiClient.getPreferences(user.id);
                if (prefs.agent_settings) {
                    setSettings(prev => ({
                        ...prev,
                        ...prefs.agent_settings as any // Cast because of string literals vs string
                    }));
                }
            } catch (error) {
                console.error('Failed to load agent settings from backend', error);
                // Fallback to local storage (already loaded in initial state) is fine
            } finally {
                setLoading(false);
            }
        };
        loadSettings();
    }, [user]);

    // Save to local storage whenever settings change (for offline/backup)
    useEffect(() => {
        localStorage.setItem('agent_settings', JSON.stringify(settings));
    }, [settings]);

    const handleSave = async () => {
        // Validate temperature
        if (settings.temperature < 0 || settings.temperature > 1 || isNaN(settings.temperature)) {
            showToast('Temperature must be between 0 and 1', 'error');
            setSettings(prev => ({ ...prev, temperature: Math.max(0, Math.min(1, prev.temperature || 0.7)) }));
            return;
        }

        setSaved(true);
        // Dispatch storage event so ChatScreen picks up changes immediately
        window.dispatchEvent(new StorageEvent('storage', {
            key: 'agent_settings',
            newValue: JSON.stringify(settings),
        }));

        if (user) {
            try {
                await apiClient.updatePreferences(user.id, {
                    agent_settings: settings
                });
                showToast('Agent settings saved successfully!', 'success');
            } catch (error) {
                console.error('Failed to sync agent settings to backend', error);
                showToast('Settings saved locally. Backend sync failed.', 'info');
            }
        } else {
            showToast('Settings saved locally.', 'success');
        }
        setTimeout(() => setSaved(false), 2000);
    };

    const handleReset = () => {
        setShowResetConfirm(true);
    };

    const confirmReset = () => {
        setSettings(DEFAULT_SETTINGS);
        localStorage.setItem('agent_settings', JSON.stringify(DEFAULT_SETTINGS));
        window.dispatchEvent(new StorageEvent('storage', {
            key: 'agent_settings',
            newValue: JSON.stringify(DEFAULT_SETTINGS),
        }));
        setShowResetConfirm(false);
        showToast('Settings reset to defaults.', 'success');
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-background-light dark:bg-background-dark flex items-center justify-center">
                <span className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></span>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background-light dark:bg-background-dark pb-24">
            {/* Header */}
            <div className="flex items-center p-4 bg-white dark:bg-card-dark sticky top-0 z-10 border-b border-slate-100 dark:border-slate-800 shadow-sm">
                <button
                    onClick={() => navigate(-1)}
                    className="p-2 -ml-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-900 dark:text-white transition-colors"
                >
                    <span className="material-symbols-outlined">arrow_back</span>
                </button>
                <h2 className="flex-1 text-center font-bold text-lg dark:text-white">Agent Settings</h2>
                <button
                    onClick={handleReset}
                    className="p-2 -mr-2 text-xs font-bold text-slate-500 hover:text-red-500 transition-colors"
                >
                    RESET
                </button>
            </div>

            <div className="p-4 space-y-6">
                {/* Model Selection */}
                <div>
                    <h3 className="text-sm font-bold text-slate-500 uppercase mb-3">AI Model</h3>
                    <div className="bg-white dark:bg-card-dark rounded-2xl p-4 shadow-sm space-y-3">
                        {[
                            { id: 'gemini-pro', name: 'Gemini Pro', desc: 'Fast & capable (Recommended)' },
                            { id: 'gpt-4', name: 'GPT-4', desc: 'Most accurate, slower' },
                            { id: 'claude-3', name: 'Claude 3', desc: 'Natural conversation' }
                        ].map(model => (
                            <label key={model.id} className="flex items-center justify-between p-3 rounded-xl border border-slate-100 dark:border-slate-700 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                <div className="flex items-center gap-3">
                                    <div className={`w-10 h-10 rounded-full flex items-center justify-center ${settings.model === model.id ? 'bg-primary/10 text-primary' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'}`}>
                                        <span className="material-symbols-outlined">smart_toy</span>
                                    </div>
                                    <div>
                                        <p className="font-medium dark:text-white">{model.name}</p>
                                        <p className="text-xs text-slate-500">{model.desc}</p>
                                    </div>
                                </div>
                                <input
                                    type="radio"
                                    name="model"
                                    checked={settings.model === model.id}
                                    onChange={() => setSettings(prev => ({ ...prev, model: model.id as any }))}
                                    className="w-5 h-5 text-primary focus:ring-primary border-gray-300"
                                />
                            </label>
                        ))}
                    </div>
                </div>

                {/* Persona */}
                <div>
                    <h3 className="text-sm font-bold text-slate-500 uppercase mb-3">Persona</h3>
                    <div className="bg-white dark:bg-card-dark rounded-2xl p-4 shadow-sm">
                        <div className="grid grid-cols-3 gap-3">
                            {[
                                { id: 'medical', icon: 'stethoscope', label: 'Medical' },
                                { id: 'friendly', icon: 'sentiment_satisfied', label: 'Friendly' },
                                { id: 'concise', icon: 'bolt', label: 'Concise' }
                            ].map(persona => (
                                <button
                                    key={persona.id}
                                    onClick={() => setSettings(prev => ({ ...prev, persona: persona.id as any }))}
                                    className={`flex flex-col items-center justify-center p-3 rounded-xl border transition-all ${settings.persona === persona.id
                                        ? 'border-primary bg-primary/5 text-primary'
                                        : 'border-slate-100 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 text-slate-500'
                                        }`}
                                >
                                    <span className="material-symbols-outlined mb-1">{persona.icon}</span>
                                    <span className="text-xs font-medium">{persona.label}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Response Settings */}
                <div>
                    <h3 className="text-sm font-bold text-slate-500 uppercase mb-3">Response Settings</h3>
                    <div className="bg-white dark:bg-card-dark rounded-2xl p-4 shadow-sm space-y-6">
                        {/* Length */}
                        <div>
                            <div className="flex justify-between mb-2">
                                <label className="text-sm font-medium dark:text-white">Response Length</label>
                                <span className="text-xs text-slate-500 capitalize">{settings.responseLength}</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="2"
                                step="1"
                                value={settings.responseLength === 'short' ? 0 : settings.responseLength === 'medium' ? 1 : 2}
                                onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    setSettings(prev => ({
                                        ...prev,
                                        responseLength: val === 0 ? 'short' : val === 1 ? 'medium' : 'long'
                                    }));
                                }}
                                className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary"
                            />
                            <div className="flex justify-between mt-1 text-xs text-slate-400">
                                <span>Short</span>
                                <span>Medium</span>
                                <span>Long</span>
                            </div>
                        </div>

                        {/* Temperature */}
                        <div>
                            <div className="flex justify-between mb-2">
                                <label className="text-sm font-medium dark:text-white">Creativity (Temperature)</label>
                                <span className="text-xs text-slate-500">{settings.temperature}</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={settings.temperature}
                                onChange={(e) => setSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                                className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary"
                            />
                            <div className="flex justify-between mt-1 text-xs text-slate-400">
                                <span>Precise</span>
                                <span>Balanced</span>
                                <span>Creative</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Voice Settings */}
                <div>
                    <h3 className="text-sm font-bold text-slate-500 uppercase mb-3">Voice Output</h3>
                    <div className="bg-white dark:bg-card-dark rounded-2xl p-4 shadow-sm">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600">
                                    <span className="material-symbols-outlined">record_voice_over</span>
                                </div>
                                <div>
                                    <p className="font-medium dark:text-white">Voice Preference</p>
                                    <p className="text-xs text-slate-500">For text-to-speech</p>
                                </div>
                            </div>
                            <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
                                <button
                                    onClick={() => setSettings(prev => ({ ...prev, voice: 'male' }))}
                                    className={`px-3 py-1 rounded-md text-sm font-medium transition-all ${settings.voice === 'male'
                                        ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-white'
                                        : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                        }`}
                                >
                                    Male
                                </button>
                                <button
                                    onClick={() => setSettings(prev => ({ ...prev, voice: 'female' }))}
                                    className={`px-3 py-1 rounded-md text-sm font-medium transition-all ${settings.voice === 'female'
                                        ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-white'
                                        : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                        }`}
                                >
                                    Female
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <button
                    onClick={handleSave}
                    className="w-full py-4 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 flex items-center justify-center gap-2 hover:bg-primary-dark transition-colors"
                >
                    {saved ? (
                        <>
                            <span className="material-symbols-outlined">check</span>
                            Settings Saved
                        </>
                    ) : (
                        'Save Configuration'
                    )}
                </button>
            </div>

            {/* Reset Confirmation Modal */}
            {showResetConfirm && (
                <Modal isOpen={true} onClose={() => setShowResetConfirm(false)} title="Reset Settings">
                    <p className="text-slate-600 dark:text-slate-300 text-sm mb-6">
                        Are you sure you want to reset all agent settings to their defaults? This cannot be undone.
                    </p>
                    <div className="flex gap-3">
                        <button
                            onClick={() => setShowResetConfirm(false)}
                            className="flex-1 py-3 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold rounded-xl hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={confirmReset}
                            className="flex-1 py-3 bg-red-500 text-white font-bold rounded-xl hover:bg-red-600 transition-colors"
                        >
                            Reset
                        </button>
                    </div>
                </Modal>
            )}
        </div>
    );
};

export default AgentSettingsScreen;
