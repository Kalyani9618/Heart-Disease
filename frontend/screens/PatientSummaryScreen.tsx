import React, { useState, useEffect } from 'react';
import { apiClient } from '../services/apiClient';
import ScreenHeader from '../components/ScreenHeader';

export default function PatientSummaryScreen() {
    const [loading, setLoading] = useState(true);
    const [summary, setSummary] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadSummary();
    }, []);

    const loadSummary = async () => {
        try {
            setLoading(true);
            const data = await apiClient.getPatientSummary('current_user');
            setSummary(data);
            setError(null);
        } catch (err) {
            console.error('Failed to load patient summary:', err);
            setError('Failed to load patient summary. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity?.toLowerCase()) {
            case 'critical': return { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-600 dark:text-red-400' };
            case 'high': return { bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-600 dark:text-orange-400' };
            case 'moderate': return { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-600 dark:text-amber-400' };
            case 'low': return { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-600 dark:text-emerald-400' };
            default: return { bg: 'bg-slate-100 dark:bg-slate-800', text: 'text-slate-600 dark:text-slate-400' };
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader
                title="Patient Summary"
                subtitle="Health Record Overview"
                rightIcon="refresh"
                onRightAction={loadSummary}
            />

            <div className="p-4 space-y-4">
                {loading ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <div className="w-8 h-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading summary...</p>
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center py-20">
                        <span className="material-symbols-outlined text-5xl text-red-500 mb-3">error</span>
                        <p className="text-red-500 text-sm mb-4">{error}</p>
                        <button
                            onClick={loadSummary}
                            className="px-5 py-2.5 bg-indigo-600 text-white rounded-xl text-sm font-semibold hover:bg-indigo-700 transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                ) : summary ? (
                    <>
                        {/* AI Summary */}
                        {summary.ai_summary && (
                            <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-4 border-l-4 border-indigo-500">
                                <div className="flex items-center gap-2 mb-2">
                                    <span className="material-symbols-outlined text-lg text-indigo-600 dark:text-indigo-400">auto_awesome</span>
                                    <h4 className="text-sm font-bold text-slate-800 dark:text-white">AI-Generated Summary</h4>
                                </div>
                                <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{summary.ai_summary}</p>
                            </div>
                        )}

                        {/* Demographics */}
                        {summary.demographics && (
                            <>
                                <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Demographics</h2>
                                <div className="bg-white dark:bg-card-dark rounded-xl p-4 shadow-sm border border-slate-100 dark:border-slate-800 divide-y divide-slate-100 dark:divide-slate-700">
                                    <div className="flex justify-between py-2.5">
                                        <span className="text-sm text-slate-500 dark:text-slate-400 font-medium">Age</span>
                                        <span className="text-sm text-slate-800 dark:text-white font-semibold">{summary.demographics.age}</span>
                                    </div>
                                    <div className="flex justify-between py-2.5">
                                        <span className="text-sm text-slate-500 dark:text-slate-400 font-medium">Gender</span>
                                        <span className="text-sm text-slate-800 dark:text-white font-semibold">{summary.demographics.gender}</span>
                                    </div>
                                    {summary.demographics.blood_type && (
                                        <div className="flex justify-between py-2.5">
                                            <span className="text-sm text-slate-500 dark:text-slate-400 font-medium">Blood Type</span>
                                            <span className="text-sm text-slate-800 dark:text-white font-semibold">{summary.demographics.blood_type}</span>
                                        </div>
                                    )}
                                </div>
                            </>
                        )}

                        {/* Medical Conditions */}
                        {summary.conditions && summary.conditions.length > 0 && (
                            <>
                                <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Medical Conditions</h2>
                                <div className="space-y-2.5">
                                    {summary.conditions.map((condition: any, index: number) => {
                                        const sevColors = condition.severity ? getSeverityColor(condition.severity) : null;
                                        return (
                                            <div key={index} className="bg-white dark:bg-card-dark rounded-xl p-4 border-l-4 border-pink-500 shadow-sm">
                                                <div className="flex items-center gap-2.5 mb-1">
                                                    <span className="material-symbols-outlined text-lg text-pink-500">medical_services</span>
                                                    <h4 className="text-sm font-semibold text-slate-800 dark:text-white flex-1">{condition.name || condition.condition}</h4>
                                                </div>
                                                {condition.diagnosed_date && (
                                                    <p className="text-xs text-slate-500 dark:text-slate-400 ml-8 mb-1">Diagnosed: {new Date(condition.diagnosed_date).toLocaleDateString()}</p>
                                                )}
                                                {sevColors && (
                                                    <span className={`inline-block ml-8 mt-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase ${sevColors.bg} ${sevColors.text}`}>
                                                        {condition.severity}
                                                    </span>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            </>
                        )}

                        {/* Current Medications */}
                        {summary.medications && summary.medications.length > 0 && (
                            <>
                                <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Current Medications</h2>
                                <div className="space-y-2.5">
                                    {summary.medications.map((medication: any, index: number) => (
                                        <div key={index} className="bg-white dark:bg-card-dark rounded-xl p-4 border-l-4 border-blue-500 shadow-sm">
                                            <div className="flex items-center gap-2.5 mb-1">
                                                <span className="material-symbols-outlined text-lg text-blue-500">medication</span>
                                                <h4 className="text-sm font-semibold text-slate-800 dark:text-white">{medication.name}</h4>
                                            </div>
                                            <p className="text-sm text-slate-600 dark:text-slate-300 ml-8">
                                                {medication.dosage} — {medication.frequency || 'As needed'}
                                            </p>
                                            {medication.prescriber && (
                                                <p className="text-xs text-slate-400 dark:text-slate-500 ml-8 mt-1">Prescribed by: {medication.prescriber}</p>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </>
                        )}

                        {/* Allergies */}
                        {summary.allergies && summary.allergies.length > 0 && (
                            <>
                                <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Allergies</h2>
                                <div className="flex flex-wrap gap-2">
                                    {summary.allergies.map((allergy: string, index: number) => (
                                        <span key={index} className="inline-flex items-center gap-1.5 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-3 py-1.5 rounded-full text-sm font-medium">
                                            <span className="material-symbols-outlined text-sm">warning</span>
                                            {allergy}
                                        </span>
                                    ))}
                                </div>
                            </>
                        )}

                        {/* Risk Factors */}
                        {summary.risk_factors && summary.risk_factors.length > 0 && (
                            <>
                                <h2 className="text-lg font-bold text-slate-800 dark:text-white pt-2">Risk Factors</h2>
                                <div className="space-y-2">
                                    {summary.risk_factors.map((factor: string, index: number) => (
                                        <div key={index} className="flex items-center gap-2.5 bg-white dark:bg-card-dark rounded-xl p-3 shadow-sm border border-slate-100 dark:border-slate-800">
                                            <span className="material-symbols-outlined text-base text-amber-500">error</span>
                                            <p className="text-sm text-slate-600 dark:text-slate-300 flex-1">{factor}</p>
                                        </div>
                                    ))}
                                </div>
                            </>
                        )}
                    </>
                ) : (
                    <div className="flex flex-col items-center py-20 text-slate-400 dark:text-slate-500">
                        <span className="material-symbols-outlined text-5xl mb-3">person_off</span>
                        <p className="text-base italic">No patient data available</p>
                    </div>
                )}
            </div>
        </div>
    );
}
