import React, { useState, useEffect } from 'react';
import { apiClient } from '../services/apiClient';
import { useToast } from '../components/Toast';
import { useConfirm } from '../components/ConfirmDialog';
import ScreenHeader from '../components/ScreenHeader';

export default function ConsentScreen() {
    const { showToast } = useToast();
    const confirm = useConfirm();
    const [loading, setLoading] = useState(true);
    const [consents, setConsents] = useState<any>({});
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadConsents();
    }, []);

    const loadConsents = async () => {
        try {
            setLoading(true);
            const data = await apiClient.getConsent('current_user');
            setConsents(data);
            setError(null);
        } catch (err) {
            console.error('Failed to load consents:', err);
            setError('Failed to load consent settings.');
        } finally {
            setLoading(false);
        }
    };

    const handleConsentToggle = async (consentType: string, value: boolean) => {
        try {
            const updated = { ...consents, [consentType]: value };
            await apiClient.updateConsent('current_user', updated);
            setConsents(updated);
            showToast('Consent preferences updated', 'success');
        } catch (err) {
            console.error('Failed to update consent:', err);
            showToast('Failed to update consent. Please try again.', 'error');
        }
    };

    const consentItems = [
        {
            key: 'data_processing',
            title: 'Data Processing',
            description: 'Allow processing of your health data for personalized insights and recommendations',
            icon: 'dns',
            required: true,
        },
        {
            key: 'ai_analysis',
            title: 'AI Analysis',
            description: 'Enable AI-powered analysis of your health records and vitals',
            icon: 'auto_awesome',
            required: false,
        },
        {
            key: 'data_sharing',
            title: 'Data Sharing',
            description: 'Share anonymized data for research and improving healthcare AI',
            icon: 'share',
            required: false,
        },
        {
            key: 'marketing',
            title: 'Marketing Communications',
            description: 'Receive health tips, updates, and promotional content',
            icon: 'mail',
            required: false,
        },
        {
            key: 'analytics',
            title: 'Usage Analytics',
            description: 'Allow collection of app usage data to improve user experience',
            icon: 'analytics',
            required: false,
        },
        {
            key: 'third_party',
            title: 'Third-Party Integration',
            description: 'Enable integration with third-party health apps and devices',
            icon: 'link',
            required: false,
        },
    ];

    const handleRevokeAll = async () => {
        const confirmed = await confirm({
            title: 'Revoke All Consents',
            message: 'This will disable all optional features. Are you sure?',
            confirmText: 'Revoke',
            variant: 'danger',
        });
        if (confirmed) {
            const updated = { ...consents };
            consentItems.filter(i => !i.required).forEach(i => {
                updated[i.key] = false;
            });
            await apiClient.updateConsent('current_user', updated);
            setConsents(updated);
            showToast('All optional consents revoked', 'success');
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader title="Privacy & Consent" subtitle="Manage Your Data Preferences" />

            <div className="p-4 space-y-4">
                {loading ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <div className="w-8 h-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading consents...</p>
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <span className="material-symbols-outlined text-5xl text-red-500 mb-3">error</span>
                        <p className="text-red-500 text-sm mb-4">{error}</p>
                        <button
                            onClick={loadConsents}
                            className="px-5 py-2.5 bg-indigo-600 text-white rounded-xl text-sm font-semibold hover:bg-indigo-700 transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                ) : (
                    <>
                        {/* Info Card */}
                        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-2xl p-5 flex flex-col items-center text-center">
                            <span className="material-symbols-outlined text-4xl text-indigo-600 dark:text-indigo-400 filled">verified_user</span>
                            <h3 className="text-lg font-bold text-slate-800 dark:text-white mt-2 mb-1">Your Data, Your Control</h3>
                            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                                We respect your privacy. Manage your consent preferences below. You can change these settings at any time.
                            </p>
                        </div>

                        <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Consent Preferences</h2>

                        {/* Consent Items */}
                        <div className="space-y-3">
                            {consentItems.map((item) => {
                                const isOn = consents[item.key] || false;
                                return (
                                    <div
                                        key={item.key}
                                        className="bg-white dark:bg-card-dark rounded-xl p-4 shadow-sm border border-slate-100 dark:border-slate-800"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
                                                isOn ? 'bg-indigo-100 dark:bg-indigo-900/30' : 'bg-slate-100 dark:bg-slate-800'
                                            }`}>
                                                <span className={`material-symbols-outlined text-xl ${
                                                    isOn ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400 dark:text-slate-500'
                                                }`}>
                                                    {item.icon}
                                                </span>
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-2">
                                                    <h4 className="text-sm font-semibold text-slate-800 dark:text-white">{item.title}</h4>
                                                    {item.required && (
                                                        <span className="bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 text-[10px] font-bold px-2 py-0.5 rounded-full">
                                                            Required
                                                        </span>
                                                    )}
                                                </div>
                                                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed mt-0.5">
                                                    {item.description}
                                                </p>
                                            </div>
                                            {/* Toggle Switch */}
                                            <button
                                                onClick={() => !item.required && handleConsentToggle(item.key, !isOn)}
                                                disabled={item.required}
                                                className={`relative w-11 h-6 rounded-full transition-colors shrink-0 ${
                                                    item.required ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                                                } ${isOn ? 'bg-indigo-600' : 'bg-slate-300 dark:bg-slate-600'}`}
                                            >
                                                <span className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                                                    isOn ? 'translate-x-[22px]' : 'translate-x-0.5'
                                                }`} />
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* GDPR Info */}
                        <div className="flex items-start gap-2.5 bg-white dark:bg-card-dark rounded-xl p-3 border border-slate-100 dark:border-slate-800">
                            <span className="material-symbols-outlined text-lg text-slate-400 shrink-0 mt-0.5">info</span>
                            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                                In compliance with GDPR and HIPAA regulations. Your data is encrypted and securely stored.
                            </p>
                        </div>

                        {/* Revoke All */}
                        <button
                            onClick={handleRevokeAll}
                            className="w-full flex items-center justify-center gap-2 bg-white dark:bg-card-dark border border-red-300 dark:border-red-800 rounded-xl py-3 text-red-500 font-semibold text-sm hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                        >
                            <span className="material-symbols-outlined text-lg">cancel</span>
                            Revoke All Optional Consents
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}
