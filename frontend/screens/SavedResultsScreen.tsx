import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import ScreenHeader from '../components/ScreenHeader';
import StructuredInterpretation from '../components/StructuredInterpretation';
import {
    getSavedAssessments,
    deleteAssessment,
    clearAllAssessments,
    shareViaWhatsApp,
    shareViaLink,
    downloadAsPDF,
    shareAsPDF,
    SavedAssessment,
} from '../services/assessmentStorage';

const SavedResultsScreen: React.FC = () => {
    const navigate = useNavigate();
    const [assessments, setAssessments] = useState<SavedAssessment[]>([]);
    const [expandedId, setExpandedId] = useState<string | null>(null);
    const [confirmClearAll, setConfirmClearAll] = useState(false);
    const [toast, setToast] = useState<string | null>(null);
    const [shareOpenId, setShareOpenId] = useState<string | null>(null);

    const refresh = useCallback(() => {
        setAssessments(getSavedAssessments());
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const showToast = (msg: string) => {
        setToast(msg);
        setTimeout(() => setToast(null), 3000);
    };

    const handleDelete = (id: string) => {
        deleteAssessment(id);
        if (expandedId === id) setExpandedId(null);
        refresh();
        showToast('Assessment deleted');
    };

    const handleClearAll = () => {
        if (!confirmClearAll) {
            setConfirmClearAll(true);
            setTimeout(() => setConfirmClearAll(false), 4000);
            return;
        }
        clearAllAssessments();
        setExpandedId(null);
        refresh();
        setConfirmClearAll(false);
        showToast('All assessments cleared');
    };

    const riskColor = (riskLevel: string | undefined, prediction: number) => {
        const rl = (riskLevel || '').toLowerCase();
        if (rl === 'critical' || (prediction === 1 && !riskLevel)) return { bg: 'bg-red-50 dark:bg-red-950/20', text: 'text-red-700 dark:text-red-300', badge: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300', border: 'border-red-200 dark:border-red-800' };
        if (rl === 'high') return { bg: 'bg-orange-50 dark:bg-orange-950/20', text: 'text-orange-700 dark:text-orange-300', badge: 'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300', border: 'border-orange-200 dark:border-orange-800' };
        if (rl === 'moderate') return { bg: 'bg-amber-50 dark:bg-amber-950/20', text: 'text-amber-700 dark:text-amber-300', badge: 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300', border: 'border-amber-200 dark:border-amber-800' };
        return { bg: 'bg-green-50 dark:bg-green-950/20', text: 'text-green-700 dark:text-green-300', badge: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300', border: 'border-green-200 dark:border-green-800' };
    };

    return (
        <div className="flex flex-col h-full bg-slate-50 dark:bg-black">
            <ScreenHeader
                title="Saved Results"
                subtitle="View your past heart risk assessments"
            />

            <div className="flex-1 overflow-y-auto pb-24">
                {/* Toast */}
                {toast && (
                    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 px-5 py-2.5 bg-slate-900 dark:bg-slate-200 text-white dark:text-slate-900 text-sm font-semibold rounded-xl shadow-lg animate-in fade-in slide-in-from-top-4 duration-300">
                        {toast}
                    </div>
                )}

                <div className="p-4 space-y-3">
                    {/* Header row */}
                    {assessments.length > 0 && (
                        <div className="flex items-center justify-between mb-2">
                            <p className="text-sm text-slate-500 dark:text-slate-400">
                                <strong className="text-slate-700 dark:text-slate-200">{assessments.length}</strong> saved assessment{assessments.length !== 1 ? 's' : ''}
                            </p>
                            <button
                                onClick={handleClearAll}
                                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                                    confirmClearAll
                                        ? 'bg-red-600 text-white hover:bg-red-700'
                                        : 'text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20'
                                }`}
                            >
                                {confirmClearAll ? 'Confirm Clear All?' : 'Clear All'}
                            </button>
                        </div>
                    )}

                    {/* Empty state */}
                    {assessments.length === 0 && (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <div className="w-20 h-20 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-6">
                                <span className="material-symbols-outlined text-4xl text-slate-300 dark:text-slate-600">folder_open</span>
                            </div>
                            <h3 className="text-lg font-bold text-slate-700 dark:text-slate-200 mb-2">No Saved Results</h3>
                            <p className="text-sm text-slate-400 max-w-xs leading-relaxed mb-6">
                                Complete a heart risk assessment and save the results to view them here anytime.
                            </p>
                            <button
                                onClick={() => navigate('/assessment')}
                                className="px-6 py-3 bg-primary text-white rounded-xl font-bold text-sm hover:bg-primary-dark transition-colors flex items-center gap-2"
                            >
                                <span className="material-symbols-outlined text-lg">ecg_heart</span>
                                Start Assessment
                            </button>
                        </div>
                    )}

                    {/* Assessment list */}
                    {assessments.map((a) => {
                        const colors = riskColor(a.result.risk_level, a.result.prediction);
                        const isExpanded = expandedId === a.id;
                        const prob = (a.result.probability * 100).toFixed(1);
                        const date = new Date(a.timestamp);
                        const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
                        const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

                        return (
                            <div
                                key={a.id}
                                className={`rounded-2xl border overflow-hidden transition-all duration-300 ${colors.border} ${isExpanded ? 'shadow-lg' : 'shadow-sm'}`}
                            >
                                {/* Card Header - always visible */}
                                <button
                                    onClick={() => setExpandedId(isExpanded ? null : a.id)}
                                    className={`w-full p-4 flex items-center gap-3 text-left transition-colors ${isExpanded ? colors.bg : 'bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800/80'}`}
                                >
                                    {/* Risk indicator */}
                                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${colors.badge}`}>
                                        <span className="material-symbols-outlined text-xl">
                                            {a.result.prediction === 1 ? 'warning' : 'verified'}
                                        </span>
                                    </div>

                                    {/* Info */}
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 mb-0.5">
                                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide ${colors.badge}`}>
                                                {a.result.risk_level || (a.result.prediction === 1 ? 'High' : 'Low')} Risk
                                            </span>
                                            <span className="text-xs text-slate-400">{prob}%</span>
                                        </div>
                                        <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
                                            {dateStr} at {timeStr} • Age {a.input.age}, {a.input.sex === 1 ? 'Male' : 'Female'}
                                        </p>
                                    </div>

                                    {/* Expand arrow */}
                                    <span className={`material-symbols-outlined text-slate-400 transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`}>
                                        expand_more
                                    </span>
                                </button>

                                {/* Expanded Detail */}
                                {isExpanded && (
                                    <div className="bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800 animate-in fade-in slide-in-from-top-2 duration-300">
                                        {/* Key Metrics */}
                                        <div className="p-4">
                                            <div className="grid grid-cols-3 gap-2 mb-4">
                                                <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 text-center">
                                                    <p className="text-[10px] uppercase tracking-wide font-bold text-slate-400 mb-0.5">Probability</p>
                                                    <p className={`text-lg font-black ${colors.text}`}>{prob}%</p>
                                                </div>
                                                <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 text-center">
                                                    <p className="text-[10px] uppercase tracking-wide font-bold text-slate-400 mb-0.5">Confidence</p>
                                                    <p className="text-lg font-black text-blue-600 dark:text-blue-400">
                                                        {a.result.confidence ? `${(a.result.confidence * 100).toFixed(0)}%` : 'N/A'}
                                                    </p>
                                                </div>
                                                <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 text-center">
                                                    <p className="text-[10px] uppercase tracking-wide font-bold text-slate-400 mb-0.5">Triage</p>
                                                    <p className="text-sm font-bold text-slate-700 dark:text-slate-200">
                                                        {a.result.triage_level || '—'}
                                                    </p>
                                                </div>
                                            </div>

                                            {/* Patient Info Grid */}
                                            <div className="mb-4">
                                                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                                    <span className="material-symbols-outlined text-sm">person</span>
                                                    Patient Data
                                                </h4>
                                                <div className="grid grid-cols-2 gap-1.5">
                                                    {([
                                                        ['Age', `${a.input.age} yrs`],
                                                        ['Sex', a.input.sex === 1 ? 'Male' : 'Female'],
                                                        ['Resting BP', `${a.input.resting_bp_s} mm Hg`],
                                                        ['Cholesterol', `${a.input.cholesterol} mg/dl`],
                                                        ['Max HR', `${a.input.max_heart_rate} bpm`],
                                                        ['Oldpeak', `${a.input.oldpeak}`],
                                                    ] as const).map(([label, val], idx) => (
                                                        <div key={idx} className="flex justify-between items-center bg-slate-50 dark:bg-slate-800/40 rounded-lg px-3 py-2">
                                                            <span className="text-[11px] text-slate-400">{label}</span>
                                                            <span className="text-[11px] font-bold text-slate-700 dark:text-slate-200">{val}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* Clinical Interpretation */}
                                            {a.result.clinical_interpretation && (
                                                <div className="mb-4">
                                                    <StructuredInterpretation
                                                        interpretation={a.result.clinical_interpretation}
                                                        isGrounded={a.result.is_grounded}
                                                    />
                                                </div>
                                            )}

                                            {/* Test Results */}
                                            {a.result.test_results && a.result.test_results.length > 0 && (
                                                <div className="mb-4">
                                                    <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                                        <span className="material-symbols-outlined text-sm text-blue-500">lab_panel</span>
                                                        Test Results
                                                    </h4>
                                                    <div className="space-y-1.5">
                                                        {a.result.test_results.map((t, idx) => (
                                                            <div key={idx} className="bg-slate-50 dark:bg-slate-800/40 rounded-lg px-3 py-2 flex items-center justify-between">
                                                                <span className="text-[11px] font-semibold text-slate-700 dark:text-slate-200">{t.test_name}</span>
                                                                <div className="flex items-center gap-1.5">
                                                                    <span className="text-[10px] text-slate-400">{t.value}</span>
                                                                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                                                                        t.status === 'Normal' ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' :
                                                                        t.status === 'Borderline' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300' :
                                                                        t.status === 'Critical' ? 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300' :
                                                                        'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300'
                                                                    }`}>{t.status}</span>
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        {/* Actions bar */}
                                        <div className="px-4 pb-4 space-y-2">
                                            {/* Share menu */}
                                                <div className="relative">
                                                <button
                                                    onClick={() => setShareOpenId(shareOpenId === a.id ? null : a.id)}
                                                    className="w-full py-2 bg-slate-100 dark:bg-slate-800 rounded-xl text-xs font-semibold text-slate-600 dark:text-slate-300 flex items-center justify-center gap-1.5 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                                                >
                                                    <span className="material-symbols-outlined text-sm">share</span>
                                                    Share Report
                                                    <span className="material-symbols-outlined text-xs">{shareOpenId === a.id ? 'expand_less' : 'expand_more'}</span>
                                                </button>

                                                {shareOpenId === a.id && (
                                                    <div className="mt-1 space-y-2 animate-in fade-in slide-in-from-top-2 duration-200">
                                                        <div className="grid grid-cols-2 gap-2">
                                                            <button
                                                                onClick={() => { shareViaWhatsApp(a); setShareOpenId(null); }}
                                                                className="py-2 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-xl text-[10px] font-bold flex items-center justify-center gap-1 hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
                                                            >
                                                                WhatsApp
                                                            </button>
                                                            <button
                                                                onClick={async () => { await shareViaLink(a); showToast('Report text copied!'); setShareOpenId(null); }}
                                                                className="py-2 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-xl text-[10px] font-bold flex items-center justify-center gap-1 hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
                                                            >
                                                                <span className="material-symbols-outlined text-xs">content_copy</span>
                                                                Copy Text
                                                            </button>
                                                            <button
                                                                onClick={() => { downloadAsPDF(a); setShareOpenId(null); }}
                                                                className="py-2 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 rounded-xl text-[10px] font-bold flex items-center justify-center gap-1 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
                                                            >
                                                                <span className="material-symbols-outlined text-xs">picture_as_pdf</span>
                                                                Download PDF
                                                            </button>
                                                            <button
                                                                onClick={async () => { await shareAsPDF(a); setShareOpenId(null); }}
                                                                className="py-2 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-xl text-[10px] font-bold flex items-center justify-center gap-1 hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
                                                            >
                                                                <span className="material-symbols-outlined text-xs">share</span>
                                                                Share PDF
                                                            </button>
                                                        </div>
                                                        <p className="text-[9px] text-slate-400 text-center">Share as text or PDF — select any text above to copy</p>
                                                    </div>
                                                )}
                                            </div>

                                            <div className="grid grid-cols-2 gap-2">
                                                {/* Chat about this */}
                                                <button
                                                    onClick={() => {
                                                        const prob = (a.result.probability * 100).toFixed(1);
                                                        const conf = a.result.confidence ? `${(a.result.confidence * 100).toFixed(0)}%` : 'N/A';
                                                        const chestPainMap: Record<number, string> = { 1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic' };
                                                        const ecgMap: Record<number, string> = { 0: 'Normal', 1: 'ST Abnormality', 2: 'LVH' };
                                                        const slopeMap: Record<number, string> = { 1: 'Upsloping', 2: 'Flat', 3: 'Downsloping' };

                                                        let msg = `Based on my heart risk assessment results:\n`;
                                                        msg += `\n--- Patient Data ---\n`;
                                                        msg += `- Age: ${a.input.age}, Sex: ${a.input.sex === 1 ? 'Male' : 'Female'}\n`;
                                                        msg += `- Resting BP: ${a.input.resting_bp_s} mm Hg\n`;
                                                        msg += `- Cholesterol: ${a.input.cholesterol} mg/dl\n`;
                                                        msg += `- Max Heart Rate: ${a.input.max_heart_rate} bpm\n`;
                                                        msg += `- Fasting Blood Sugar: ${a.input.fasting_blood_sugar === 1 ? '> 120' : '≤ 120'} mg/dl\n`;
                                                        msg += `- Chest Pain Type: ${chestPainMap[a.input.chest_pain_type] || a.input.chest_pain_type}\n`;
                                                        msg += `- Resting ECG: ${ecgMap[a.input.resting_ecg] || a.input.resting_ecg}\n`;
                                                        msg += `- Exercise Angina: ${a.input.exercise_angina === 1 ? 'Yes' : 'No'}\n`;
                                                        msg += `- Oldpeak: ${a.input.oldpeak}\n`;
                                                        msg += `- ST Slope: ${slopeMap[a.input.st_slope] || a.input.st_slope}\n`;
                                                        msg += `\n--- Model Prediction ---\n`;
                                                        msg += `- Risk Level: ${a.result.risk_level || (a.result.prediction === 1 ? 'High' : 'Low')}\n`;
                                                        msg += `- Probability: ${prob}%\n`;
                                                        msg += `- Confidence: ${conf}\n`;
                                                        msg += `- Prediction: ${a.result.prediction === 1 ? 'Heart Disease Likely' : 'Heart Disease Unlikely'}\n`;
                                                        if (a.result.message) msg += `- Summary: ${a.result.message}\n`;
                                                        if (a.result.clinical_interpretation) msg += `\n--- Clinical Interpretation ---\n${a.result.clinical_interpretation}\n`;
                                                        msg += `\nWhat lifestyle changes and medical steps should I take to reduce my cardiovascular risk? Please provide specific, actionable recommendations based on my data above.`;
                                                        navigate('/chat', { state: { autoSend: msg } });
                                                    }}
                                                    className="py-2.5 bg-indigo-600 text-white rounded-xl text-xs font-bold flex items-center justify-center gap-1.5 hover:bg-indigo-700 transition-colors"
                                                >
                                                    <span className="material-symbols-outlined text-sm">chat</span>
                                                    Chat About This
                                                </button>

                                                {/* Delete */}
                                                <button
                                                    onClick={() => handleDelete(a.id)}
                                                    className="py-2.5 text-red-500 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 hover:bg-red-50 dark:hover:bg-red-900/10 transition-colors"
                                                >
                                                    <span className="material-symbols-outlined text-sm">delete</span>
                                                    Delete
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Float button to new assessment */}
                {assessments.length > 0 && (
                    <div className="fixed bottom-24 right-4 z-40">
                        <button
                            onClick={() => navigate('/assessment')}
                            className="w-14 h-14 rounded-full bg-primary text-white shadow-lg shadow-primary/30 flex items-center justify-center hover:bg-primary-dark transition-colors active:scale-95"
                        >
                            <span className="material-symbols-outlined text-2xl">add</span>
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SavedResultsScreen;
