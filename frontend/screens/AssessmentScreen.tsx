import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../services/apiClient';
import { HeartDiseasePredictionRequest, HeartDiseasePredictionResponse, TestResultDetail } from '../services/api.types';
import ScreenHeader from '../components/ScreenHeader';
import StructuredInterpretation from '../components/StructuredInterpretation';
import { saveAssessment, shareViaWhatsApp, shareViaLink, downloadAsPDF, shareAsPDF, SavedAssessment } from '../services/assessmentStorage';

const AssessmentScreen: React.FC = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<HeartDiseasePredictionResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [touched, setTouched] = useState<Record<string, boolean>>({});

    interface AssessmentFormState {
        age: number | string;
        sex: number;
        chest_pain_type: number;
        resting_bp_s: number | string;
        cholesterol: number | string;
        fasting_blood_sugar: number;
        resting_ecg: number;
        max_heart_rate: number | string;
        exercise_angina: number;
        oldpeak: number | string;
        st_slope: number;
    }

    const [savedNotice, setSavedNotice] = useState<string | null>(null);
    const [shareMenuOpen, setShareMenuOpen] = useState(false);

    const [formData, setFormData] = useState<AssessmentFormState>({
        age: 50,
        sex: 1,
        chest_pain_type: 1,
        resting_bp_s: 120,
        cholesterol: 200,
        fasting_blood_sugar: 0,
        resting_ecg: 0,
        max_heart_rate: 150,
        exercise_angina: 0,
        oldpeak: 0.0,
        st_slope: 1,
    });

    const handleChange = useCallback((field: keyof AssessmentFormState, value: string | number) => {
        setTouched(prev => ({ ...prev, [field]: true }));
        if (value === '') {
            setFormData(prev => ({ ...prev, [field]: '' }));
            return;
        }
        setFormData(prev => ({ ...prev, [field]: value }));
    }, []);

    const markTouched = useCallback((field: string) => {
        setTouched(prev => ({ ...prev, [field]: true }));
    }, []);

    /* ---- Inline validation helpers ---- */
    const validate = useCallback(() => {
        const errors: Record<string, string> = {};
        const age = Number(formData.age);
        if (formData.age === '' || isNaN(age)) errors.age = 'Required';
        else if (age < 1 || age > 120) errors.age = 'Enter 1–120';

        const bp = Number(formData.resting_bp_s);
        if (formData.resting_bp_s === '' || isNaN(bp)) errors.resting_bp_s = 'Required';
        else if (bp < 60 || bp > 250) errors.resting_bp_s = 'Enter 60–250';

        const chol = Number(formData.cholesterol);
        if (formData.cholesterol === '' || isNaN(chol)) errors.cholesterol = 'Required';
        else if (chol < 50 || chol > 600) errors.cholesterol = 'Enter 50–600';

        const hr = Number(formData.max_heart_rate);
        if (formData.max_heart_rate === '' || isNaN(hr)) errors.max_heart_rate = 'Required';
        else if (hr < 50 || hr > 250) errors.max_heart_rate = 'Enter 50–250';

        const op = Number(formData.oldpeak);
        if (formData.oldpeak === '' || isNaN(op)) errors.oldpeak = 'Required';
        else if (op < -5 || op > 10) errors.oldpeak = 'Enter -5 to 10';

        return errors;
    }, [formData]);

    const validationErrors = validate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        // Mark everything touched so errors show
        const allFields = Object.keys(formData);
        setTouched(Object.fromEntries(allFields.map(f => [f, true])));

        if (Object.keys(validationErrors).length > 0) {
            setError('Please fix the highlighted fields before submitting.');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        const requestData: HeartDiseasePredictionRequest = {
            age: Number(formData.age),
            sex: Number(formData.sex),
            chest_pain_type: Number(formData.chest_pain_type),
            resting_bp_s: Number(formData.resting_bp_s),
            cholesterol: Number(formData.cholesterol),
            fasting_blood_sugar: Number(formData.fasting_blood_sugar),
            resting_ecg: Number(formData.resting_ecg),
            max_heart_rate: Number(formData.max_heart_rate),
            exercise_angina: Number(formData.exercise_angina),
            oldpeak: Number(formData.oldpeak),
            st_slope: Number(formData.st_slope),
        };

        const hasInvalid = Object.values(requestData).some(val => isNaN(val));
        if (hasInvalid) {
            setError('Please fill in all fields with valid numbers.');
            setLoading(false);
            return;
        }

        setTimeout(() => {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }, 100);

        try {
            const data = await apiClient.predictHeartDisease(requestData);
            setResult(data);
        } catch (err: any) {
            setError(err.message || 'Failed to get prediction. Ensure the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    /* ---- Small helper: inline error text ---- */
    const FieldError: React.FC<{ field: string }> = ({ field }) => {
        if (!touched[field] || !validationErrors[field]) return null;
        return (
            <p className="text-xs text-red-500 dark:text-red-400 mt-1 flex items-center gap-1">
                <span className="material-symbols-outlined text-sm">error</span>
                {validationErrors[field]}
            </p>
        );
    };

    /* ---- Shared input class builder ---- */
    const inputCls = (field: string, extra = '') =>
        `w-full p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border-2 outline-none focus:ring-4 focus:ring-primary/10 text-slate-900 dark:text-white font-medium transition-all placeholder:text-slate-400 ${extra} ${
            touched[field] && validationErrors[field]
                ? 'border-red-400 dark:border-red-500 focus:border-red-400'
                : 'border-transparent focus:border-primary/20'
        }`;

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader
                title="Heart Risk Assessment"
                subtitle="AI Clinical Prediction Model"
            />

            <div className="max-w-4xl mx-auto p-4 space-y-6">

                {/* Quick link to saved results */}
                <button
                    onClick={() => navigate('/saved-results')}
                    className="w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/80 transition-colors group"
                >
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
                            <span className="material-symbols-outlined text-indigo-600 dark:text-indigo-400 text-lg">history</span>
                        </div>
                        <div className="text-left">
                            <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">View Past Results</p>
                            <p className="text-[10px] text-slate-400">Access your saved assessments anytime</p>
                        </div>
                    </div>
                    <span className="material-symbols-outlined text-slate-300 group-hover:text-slate-500 transition-colors text-lg">chevron_right</span>
                </button>

                {/* Info Card */}
                <div className="bg-gradient-to-br from-blue-600 to-indigo-700 dark:from-blue-700 dark:to-indigo-800 text-white rounded-3xl p-6 shadow-xl shadow-blue-500/20 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-40 h-40 bg-white/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:scale-110 transition-transform duration-700"></div>
                    <div className="relative z-10 flex items-start gap-4">
                        <div className="p-3 bg-white/20 backdrop-blur-sm rounded-xl">
                            <span className="material-symbols-outlined text-2xl">cardiology</span>
                        </div>
                        <div>
                            <h3 className="font-bold text-lg mb-1">Clinical Assessment</h3>
                            <p className="text-blue-100 text-sm leading-relaxed">
                                This tool uses a machine learning model trained on the Cleveland Heart Disease dataset to estimate risk factors.
                                <span className="block mt-2 font-medium opacity-90">Please fill out all vitals accurately.</span>
                            </p>
                        </div>
                    </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">

                    {/* ═══════ Section 1: Patient Vitals ═══════ */}
                    <section className="bg-white dark:bg-card-dark rounded-3xl p-6 shadow-sm border border-slate-100 dark:border-slate-800 hover:shadow-md transition-shadow duration-300">
                        <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-50 dark:border-slate-800">
                            <div className="w-10 h-10 rounded-xl bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 flex items-center justify-center">
                                <span className="material-symbols-outlined">vital_signs</span>
                            </div>
                            <h3 className="font-bold text-lg text-slate-800 dark:text-white">Patient Vitals</h3>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Age */}
                            <div className="space-y-2">
                                <label htmlFor="field-age" className="text-xs font-bold text-slate-500 uppercase tracking-wide">Age <span className="text-red-400">*</span></label>
                                <input
                                    id="field-age"
                                    type="number"
                                    min={1} max={120}
                                    value={formData.age}
                                    onChange={(e) => handleChange('age', e.target.value)}
                                    onBlur={() => markTouched('age')}
                                    className={inputCls('age')}
                                    required
                                />
                                <FieldError field="age" />
                            </div>

                            {/* Sex — toggle */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-500 uppercase tracking-wide">Sex</label>
                                <div className="flex bg-slate-50 dark:bg-slate-800 p-1.5 rounded-xl" role="radiogroup" aria-label="Sex">
                                    {[
                                        { label: 'Male', value: 1, icon: 'male' },
                                        { label: 'Female', value: 0, icon: 'female' }
                                    ].map(opt => (
                                        <button
                                            key={opt.value}
                                            type="button"
                                            role="radio"
                                            aria-checked={formData.sex === opt.value}
                                            onClick={() => handleChange('sex', opt.value)}
                                            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-bold transition-all ${formData.sex === opt.value
                                                ? 'bg-white dark:bg-slate-700 shadow-sm text-primary dark:text-white'
                                                : 'text-slate-500 hover:bg-white/50 dark:hover:bg-slate-700/50'
                                                }`}
                                        >
                                            <span className="material-symbols-outlined text-lg">{opt.icon}</span>
                                            {opt.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Resting BP */}
                            <div className="space-y-2">
                                <label htmlFor="field-bp" className="text-xs font-bold text-slate-500 uppercase tracking-wide">Resting BP (mm Hg) <span className="text-red-400">*</span></label>
                                <div className="relative">
                                    <input
                                        id="field-bp"
                                        type="number"
                                        min={60} max={250}
                                        value={formData.resting_bp_s}
                                        onChange={(e) => handleChange('resting_bp_s', e.target.value)}
                                        onBlur={() => markTouched('resting_bp_s')}
                                        className={inputCls('resting_bp_s', 'pl-12')}
                                        required
                                    />
                                    <span className="material-symbols-outlined absolute left-4 top-3.5 text-slate-400">blood_pressure</span>
                                </div>
                                <FieldError field="resting_bp_s" />
                            </div>

                            {/* Cholesterol */}
                            <div className="space-y-2">
                                <label htmlFor="field-chol" className="text-xs font-bold text-slate-500 uppercase tracking-wide">Cholesterol (mg/dl) <span className="text-red-400">*</span></label>
                                <div className="relative">
                                    <input
                                        id="field-chol"
                                        type="number"
                                        min={50} max={600}
                                        value={formData.cholesterol}
                                        onChange={(e) => handleChange('cholesterol', e.target.value)}
                                        onBlur={() => markTouched('cholesterol')}
                                        className={inputCls('cholesterol', 'pl-12')}
                                        required
                                    />
                                    <span className="material-symbols-outlined absolute left-4 top-3.5 text-slate-400">water_drop</span>
                                </div>
                                <FieldError field="cholesterol" />
                            </div>

                            {/* Fasting Blood Sugar — Yes/No toggle instead of raw number */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-500 uppercase tracking-wide">Fasting Blood Sugar &gt; 120 mg/dl?</label>
                                <div className="flex bg-slate-50 dark:bg-slate-800 p-1.5 rounded-xl" role="radiogroup" aria-label="Fasting Blood Sugar">
                                    {[
                                        { label: 'Yes', value: 1, color: 'text-amber-600 dark:text-amber-400' },
                                        { label: 'No', value: 0, color: 'text-green-600 dark:text-green-400' }
                                    ].map(opt => (
                                        <button
                                            key={opt.value}
                                            type="button"
                                            role="radio"
                                            aria-checked={formData.fasting_blood_sugar === opt.value}
                                            onClick={() => handleChange('fasting_blood_sugar', opt.value)}
                                            className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${formData.fasting_blood_sugar === opt.value
                                                ? `bg-white dark:bg-slate-700 shadow-sm ${opt.color}`
                                                : 'text-slate-500 hover:bg-white/50 dark:hover:bg-slate-700/50'
                                                }`}
                                        >
                                            {opt.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Max Heart Rate */}
                            <div className="space-y-2">
                                <label htmlFor="field-hr" className="text-xs font-bold text-slate-500 uppercase tracking-wide">Max Heart Rate <span className="text-red-400">*</span></label>
                                <div className="relative">
                                    <input
                                        id="field-hr"
                                        type="number"
                                        min={50} max={250}
                                        value={formData.max_heart_rate}
                                        onChange={(e) => handleChange('max_heart_rate', e.target.value)}
                                        onBlur={() => markTouched('max_heart_rate')}
                                        className={inputCls('max_heart_rate', 'pl-12')}
                                        required
                                    />
                                    <span className="material-symbols-outlined absolute left-4 top-3.5 text-slate-400">favorite</span>
                                </div>
                                <FieldError field="max_heart_rate" />
                            </div>
                        </div>
                    </section>

                    {/* ═══════ Section 2: Clinical Factors ═══════ */}
                    <section className="bg-white dark:bg-card-dark rounded-3xl p-6 shadow-sm border border-slate-100 dark:border-slate-800 hover:shadow-md transition-shadow duration-300">
                        <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-50 dark:border-slate-800">
                            <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
                                <span className="material-symbols-outlined">stethoscope</span>
                            </div>
                            <h3 className="font-bold text-lg text-slate-800 dark:text-white">Clinical Factors</h3>
                        </div>

                        <div className="space-y-6">
                            {/* Chest Pain Type — card selector (unchanged, already good) */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-slate-500 uppercase tracking-wide">Chest Pain Type</label>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3" role="radiogroup" aria-label="Chest Pain Type">
                                    {[
                                        { val: 1, label: 'Typical Angina', desc: 'Exertional chest pain', icon: 'cardiology' },
                                        { val: 2, label: 'Atypical Angina', desc: 'Non-specific pain', icon: 'help_clinic' },
                                        { val: 3, label: 'Non-Anginal', desc: 'Not heart related', icon: 'sentiment_neutral' },
                                        { val: 4, label: 'Asymptomatic', desc: 'No pain present', icon: 'check_circle' }
                                    ].map(opt => (
                                        <button
                                            key={opt.val}
                                            type="button"
                                            role="radio"
                                            aria-checked={formData.chest_pain_type === opt.val}
                                            onClick={() => handleChange('chest_pain_type', opt.val)}
                                            className={`p-4 rounded-xl border-2 text-left transition-all hover:shadow-md ${formData.chest_pain_type === opt.val
                                                ? 'bg-primary/5 border-primary shadow-sm'
                                                : 'bg-slate-50 dark:bg-slate-800 border-transparent hover:bg-white dark:hover:bg-slate-700'
                                                }`}
                                        >
                                            <div className="flex items-center justify-between mb-1">
                                                <div className="flex items-center gap-2">
                                                    <span className={`material-symbols-outlined text-lg ${formData.chest_pain_type === opt.val ? 'text-primary' : 'text-slate-400'}`}>{opt.icon}</span>
                                                    <span className={`font-bold ${formData.chest_pain_type === opt.val ? 'text-primary' : 'text-slate-900 dark:text-white'}`}>
                                                        {opt.label}
                                                    </span>
                                                </div>
                                                {formData.chest_pain_type === opt.val && (
                                                    <span className="material-symbols-outlined text-primary text-sm">check_circle</span>
                                                )}
                                            </div>
                                            <p className="text-xs text-slate-500 dark:text-slate-400 ml-7">{opt.desc}</p>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Resting ECG — 3-option card selector */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-slate-500 uppercase tracking-wide">Resting ECG</label>
                                <div className="grid grid-cols-3 gap-3" role="radiogroup" aria-label="Resting ECG">
                                    {[
                                        { val: 0, label: 'Normal', icon: 'check_circle', color: 'text-green-600 dark:text-green-400' },
                                        { val: 1, label: 'ST Abnormality', icon: 'show_chart', color: 'text-amber-600 dark:text-amber-400' },
                                        { val: 2, label: 'LVH', icon: 'ecg_heart', color: 'text-red-600 dark:text-red-400' }
                                    ].map(opt => (
                                        <button
                                            key={opt.val}
                                            type="button"
                                            role="radio"
                                            aria-checked={formData.resting_ecg === opt.val}
                                            onClick={() => handleChange('resting_ecg', opt.val)}
                                            className={`p-3 rounded-xl border-2 text-center transition-all hover:shadow-md ${formData.resting_ecg === opt.val
                                                ? 'bg-primary/5 border-primary shadow-sm'
                                                : 'bg-slate-50 dark:bg-slate-800 border-transparent hover:bg-white dark:hover:bg-slate-700'
                                                }`}
                                        >
                                            <span className={`material-symbols-outlined text-2xl mb-1 block ${formData.resting_ecg === opt.val ? 'text-primary' : opt.color}`}>{opt.icon}</span>
                                            <span className={`text-xs font-bold block ${formData.resting_ecg === opt.val ? 'text-primary' : 'text-slate-700 dark:text-slate-300'}`}>
                                                {opt.label}
                                            </span>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Exercise Angina */}
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wide">Exercise Angina?</label>
                                    <div className="flex bg-slate-50 dark:bg-slate-800 p-1.5 rounded-xl" role="radiogroup" aria-label="Exercise Angina">
                                        {[
                                            { label: 'Yes', value: 1, color: 'text-red-500' },
                                            { label: 'No', value: 0, color: 'text-green-500' }
                                        ].map(opt => (
                                            <button
                                                key={opt.value}
                                                type="button"
                                                role="radio"
                                                aria-checked={formData.exercise_angina === opt.value}
                                                onClick={() => handleChange('exercise_angina', opt.value)}
                                                className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${formData.exercise_angina === opt.value
                                                    ? `bg-white dark:bg-slate-700 shadow-sm ${opt.color}`
                                                    : 'text-slate-500 hover:bg-white/50 dark:hover:bg-slate-700/50'
                                                    }`}
                                            >
                                                {opt.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Oldpeak */}
                                <div className="space-y-2">
                                    <label htmlFor="field-oldpeak" className="text-xs font-bold text-slate-500 uppercase tracking-wide">Oldpeak (ST Depression) <span className="text-red-400">*</span></label>
                                    <input
                                        id="field-oldpeak"
                                        type="number"
                                        step="0.1"
                                        min={-5} max={10}
                                        value={formData.oldpeak}
                                        onChange={(e) => handleChange('oldpeak', e.target.value)}
                                        onBlur={() => markTouched('oldpeak')}
                                        className={inputCls('oldpeak')}
                                        required
                                    />
                                    <FieldError field="oldpeak" />
                                </div>
                            </div>

                            {/* ST Slope — 3-option card selector */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-slate-500 uppercase tracking-wide">ST Slope</label>
                                <div className="grid grid-cols-3 gap-3" role="radiogroup" aria-label="ST Slope">
                                    {[
                                        { val: 1, label: 'Up', icon: 'trending_up', color: 'text-green-600 dark:text-green-400' },
                                        { val: 2, label: 'Flat', icon: 'trending_flat', color: 'text-amber-600 dark:text-amber-400' },
                                        { val: 3, label: 'Down', icon: 'trending_down', color: 'text-red-600 dark:text-red-400' }
                                    ].map(opt => (
                                        <button
                                            key={opt.val}
                                            type="button"
                                            role="radio"
                                            aria-checked={formData.st_slope === opt.val}
                                            onClick={() => handleChange('st_slope', opt.val)}
                                            className={`p-3 rounded-xl border-2 text-center transition-all hover:shadow-md ${formData.st_slope === opt.val
                                                ? 'bg-primary/5 border-primary shadow-sm'
                                                : 'bg-slate-50 dark:bg-slate-800 border-transparent hover:bg-white dark:hover:bg-slate-700'
                                                }`}
                                        >
                                            <span className={`material-symbols-outlined text-2xl mb-1 block ${formData.st_slope === opt.val ? 'text-primary' : opt.color}`}>{opt.icon}</span>
                                            <span className={`text-xs font-bold block ${formData.st_slope === opt.val ? 'text-primary' : 'text-slate-700 dark:text-slate-300'}`}>
                                                {opt.label}
                                            </span>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Submit — standard flow (no sticky) */}
                    <div className="pt-2 pb-4">
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-4 bg-primary hover:bg-primary-dark text-white rounded-2xl font-bold text-lg shadow-xl shadow-primary/30 transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-70 disabled:hover:scale-100 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <span className="w-5 h-5 border-3 border-white/30 border-t-white rounded-full animate-spin"></span>
                                    <span>Analyzing Risk Profile...</span>
                                </>
                            ) : (
                                <>
                                    <span className="material-symbols-outlined">analytics</span>
                                    <span>Calculate Heart Risk</span>
                                </>
                            )}
                        </button>
                    </div>
                </form>

                {error && (
                    <div className="animate-in slide-in-from-bottom-5 fade-in p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl text-red-600 dark:text-red-300 flex items-center gap-3">
                        <span className="material-symbols-outlined rounded-full bg-red-100 dark:bg-red-900/50 p-1">error</span>
                        <p className="font-medium">{error}</p>
                    </div>
                )}

                {/* Result Modal */}
                {result && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
                        <div className="bg-white dark:bg-slate-900 rounded-3xl max-w-lg w-full max-h-[90vh] overflow-y-auto shadow-2xl relative animate-in zoom-in-95 duration-300 border border-slate-200 dark:border-slate-800">

                            {/* Close Button */}
                            <button
                                onClick={() => setResult(null)}
                                className="absolute top-4 right-4 p-2 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors z-10"
                            >
                                <span className="material-symbols-outlined">close</span>
                            </button>

                            {/* Header Section */}
                            <div className={`p-8 text-center relative overflow-hidden ${result.prediction === 1
                                ? 'bg-gradient-to-br from-red-50 via-orange-50 to-rose-50 dark:from-red-950/30 dark:via-orange-950/15 dark:to-rose-950/10'
                                : 'bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 dark:from-green-950/30 dark:via-emerald-950/15 dark:to-teal-950/10'
                                }`}>

                                {/* Decorative background elements */}
                                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                                    <div className={`absolute -top-10 -right-10 w-40 h-40 rounded-full opacity-[0.07] ${result.prediction === 1 ? 'bg-red-500' : 'bg-green-500'}`}></div>
                                    <div className={`absolute -bottom-8 -left-8 w-32 h-32 rounded-full opacity-[0.05] ${result.prediction === 1 ? 'bg-orange-500' : 'bg-emerald-500'}`}></div>
                                    <span className="material-symbols-outlined absolute top-6 left-6 text-5xl opacity-[0.06]">ecg_heart</span>
                                    <span className="material-symbols-outlined absolute bottom-6 right-6 text-5xl opacity-[0.06]">monitor_heart</span>
                                </div>

                                {/* Animated circle icon */}
                                <div className="relative z-10 inline-block mb-5">
                                    <div className={`absolute inset-0 rounded-full animate-ping opacity-20 ${result.prediction === 1 ? 'bg-red-400' : 'bg-green-400'}`} style={{ animationDuration: '2s' }}></div>
                                    <div className={`relative inline-flex p-5 rounded-full shadow-xl ${
                                        result.prediction === 1
                                            ? 'bg-gradient-to-br from-red-500 to-rose-600 shadow-red-500/25'
                                            : 'bg-gradient-to-br from-green-500 to-emerald-600 shadow-green-500/25'
                                    }`}>
                                        <span className="material-symbols-outlined text-white text-4xl">
                                            {result.prediction === 1 ? 'warning' : 'verified'}
                                        </span>
                                    </div>
                                </div>

                                <h2 className={`text-2xl font-black mb-2 relative z-10 ${result.prediction === 1 ? 'text-red-700 dark:text-red-400' : 'text-green-700 dark:text-green-400'
                                    }`}>
                                    {result.prediction === 1 ? 'Elevated Risk Detected' : 'Low Risk Detected'}
                                </h2>

                                {result.risk_level && (
                                    <div className="relative z-10 mb-4">
                                        <span className={`inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider shadow-sm ${
                                            result.risk_level === 'Critical' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200 shadow-red-200/50' :
                                            result.risk_level === 'High' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200 shadow-orange-200/50' :
                                            result.risk_level === 'Moderate' ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200 shadow-amber-200/50' :
                                            'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200 shadow-green-200/50'
                                        }`}>
                                            <span className={`w-2 h-2 rounded-full ${
                                                result.risk_level === 'Critical' ? 'bg-red-500' :
                                                result.risk_level === 'High' ? 'bg-orange-500' :
                                                result.risk_level === 'Moderate' ? 'bg-amber-500' : 'bg-green-500'
                                            }`}></span>
                                            {result.risk_level} Risk
                                        </span>
                                    </div>
                                )}

                                <p className="text-slate-600 dark:text-slate-300 text-sm mb-6 max-w-lg mx-auto leading-relaxed relative z-10">
                                    {result.message}
                                </p>

                                {/* Metrics Grid — Redesigned */}
                                <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto relative z-10">
                                    {/* Probability */}
                                    <div className="bg-white/90 dark:bg-slate-900/60 backdrop-blur-sm p-4 rounded-2xl border border-white/60 dark:border-slate-700/40 shadow-sm">
                                        <p className="text-[10px] uppercase tracking-widest font-bold text-slate-400 dark:text-slate-500 mb-2 flex items-center justify-center gap-1">
                                            <span className="material-symbols-outlined text-[12px]">target</span>
                                            Probability
                                        </p>
                                        <p className={`text-3xl font-black tracking-tight ${result.prediction === 1 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                                            {(result.probability * 100).toFixed(1)}<span className="text-lg">%</span>
                                        </p>
                                        <div className="h-2 w-full bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden mt-2.5">
                                            <div
                                                className={`h-full rounded-full transition-all duration-1000 ease-out ${result.prediction === 1 ? 'bg-gradient-to-r from-orange-400 to-red-500' : 'bg-gradient-to-r from-emerald-400 to-green-500'}`}
                                                style={{ width: `${result.probability * 100}%` }}
                                            ></div>
                                        </div>
                                    </div>

                                    {/* Confidence */}
                                    <div className="bg-white/90 dark:bg-slate-900/60 backdrop-blur-sm p-4 rounded-2xl border border-white/60 dark:border-slate-700/40 shadow-sm">
                                        <p className="text-[10px] uppercase tracking-widest font-bold text-slate-400 dark:text-slate-500 mb-2 flex items-center justify-center gap-1">
                                            <span className="material-symbols-outlined text-[12px]">speed</span>
                                            Confidence
                                        </p>
                                        <p className="text-3xl font-black text-blue-600 dark:text-blue-400 tracking-tight">
                                            {result.confidence ? `${(result.confidence * 100).toFixed(0)}` : 'N/A'}<span className="text-lg">%</span>
                                        </p>
                                        {result.quality_score !== undefined && (
                                            <p className="text-[10px] text-slate-400 mt-2 flex items-center justify-center gap-1">
                                                <span className="material-symbols-outlined text-[11px]">verified</span>
                                                Quality: {(result.quality_score * 100).toFixed(0)}%
                                            </p>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Clinical Interpretation — Structured */}
                            {result.clinical_interpretation && (
                                <div className="px-6 pt-5">
                                    <StructuredInterpretation
                                        interpretation={result.clinical_interpretation}
                                        isGrounded={result.is_grounded}
                                    />
                                </div>
                            )}

                            {/* Triage & Actions */}
                            {result.triage_level && (
                                <div className="px-6 pt-5">
                                    <div className={`rounded-2xl border overflow-hidden ${
                                        result.triage_level === 'Emergency' ? 'border-red-200 dark:border-red-800/40' :
                                        result.triage_level === 'Urgent' ? 'border-orange-200 dark:border-orange-800/40' :
                                        'border-blue-200 dark:border-blue-800/40'
                                    }`}>
                                        <div className={`flex items-center gap-3 px-4 py-3 border-l-[3px] ${
                                            result.triage_level === 'Emergency' ? 'bg-red-50 dark:bg-red-950/20 border-l-red-500' :
                                            result.triage_level === 'Urgent' ? 'bg-orange-50 dark:bg-orange-950/20 border-l-orange-500' :
                                            'bg-blue-50 dark:bg-blue-950/20 border-l-blue-500'
                                        }`}>
                                            <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 ${
                                                result.triage_level === 'Emergency' ? 'bg-red-100 dark:bg-red-900/30' :
                                                result.triage_level === 'Urgent' ? 'bg-orange-100 dark:bg-orange-900/30' :
                                                'bg-blue-100 dark:bg-blue-900/30'
                                            }`}>
                                                <span className={`material-symbols-outlined text-[18px] ${
                                                    result.triage_level === 'Emergency' ? 'text-red-500' :
                                                    result.triage_level === 'Urgent' ? 'text-orange-500' : 'text-blue-500'
                                                }`}>emergency</span>
                                            </div>
                                            <div className="flex-1">
                                                <h3 className="font-semibold text-[13px] text-slate-800 dark:text-slate-100">Triage Assessment</h3>
                                            </div>
                                            <span className={`px-2.5 py-1 text-[10px] font-bold rounded-full ${
                                                result.triage_level === 'Emergency' ? 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300' :
                                                result.triage_level === 'Urgent' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300' :
                                                'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300'
                                            }`}>{result.triage_level}</span>
                                        </div>
                                        {result.triage_actions && result.triage_actions.length > 0 && (
                                            <div className="p-4 space-y-2 bg-white/60 dark:bg-slate-900/30">
                                                {result.triage_actions.map((action, idx) => (
                                                    <div key={idx} className="flex items-start gap-2.5">
                                                        <div className={`w-5 h-5 rounded-md flex items-center justify-center shrink-0 mt-0.5 text-[10px] font-bold ${
                                                            result.triage_level === 'Emergency' ? 'bg-red-50 text-red-500 dark:bg-red-900/20 dark:text-red-400' :
                                                            result.triage_level === 'Urgent' ? 'bg-orange-50 text-orange-500 dark:bg-orange-900/20 dark:text-orange-400' :
                                                            'bg-blue-50 text-blue-500 dark:bg-blue-900/20 dark:text-blue-400'
                                                        }`}>{idx + 1}</div>
                                                        <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">{action}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Test Results Breakdown */}
                            {result.test_results && result.test_results.length > 0 && (
                                <div className="px-6 pt-5">
                                    <div className="flex items-center gap-3 mb-4">
                                        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500/10 to-cyan-500/5 flex items-center justify-center border border-blue-100 dark:border-blue-800/30">
                                            <span className="material-symbols-outlined text-blue-500 text-[18px]">biotech</span>
                                        </div>
                                        <div>
                                            <h3 className="font-bold text-sm text-slate-800 dark:text-white">Detailed Test Results</h3>
                                            <p className="text-[10px] text-slate-400">{result.test_results.length} parameters analyzed</p>
                                        </div>
                                    </div>
                                    <div className="space-y-2.5">
                                        {result.test_results.map((test: TestResultDetail, idx: number) => (
                                            <div key={idx} className="bg-white dark:bg-slate-800/70 rounded-xl p-3.5 border border-slate-100 dark:border-slate-700/40 hover:shadow-md hover:border-slate-200 dark:hover:border-slate-600/50 transition-all duration-200">
                                                <div className="flex items-start justify-between gap-2 mb-2">
                                                    <div className="flex items-center gap-2.5">
                                                        <div className={`w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-bold ${
                                                            test.status === 'Normal' ? 'bg-green-50 text-green-600 dark:bg-green-900/30 dark:text-green-300' :
                                                            test.status === 'Borderline' ? 'bg-amber-50 text-amber-600 dark:bg-amber-900/30 dark:text-amber-300' :
                                                            test.status === 'Critical' ? 'bg-red-50 text-red-600 dark:bg-red-900/30 dark:text-red-300' :
                                                            'bg-orange-50 text-orange-600 dark:bg-orange-900/30 dark:text-orange-300'
                                                        }`}>{idx + 1}</div>
                                                        <span className="font-semibold text-[13px] text-slate-800 dark:text-slate-100">{test.test_name}</span>
                                                    </div>
                                                    <div className="flex items-center gap-1.5 shrink-0">
                                                        <span className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                                            test.status === 'Normal' ? 'bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-300' :
                                                            test.status === 'Borderline' ? 'bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' :
                                                            test.status === 'Critical' ? 'bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300' :
                                                            'bg-orange-50 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300'
                                                        }`}>
                                                            <span className={`w-1.5 h-1.5 rounded-full ${
                                                                test.status === 'Normal' ? 'bg-green-500' :
                                                                test.status === 'Borderline' ? 'bg-amber-500' :
                                                                test.status === 'Critical' ? 'bg-red-500' : 'bg-orange-500'
                                                            }`}></span>
                                                            {test.status}
                                                        </span>
                                                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                                            test.risk_contribution === 'Low' ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-300' :
                                                            test.risk_contribution === 'Moderate' ? 'bg-yellow-50 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-300' :
                                                            'bg-red-50 text-red-600 dark:bg-red-900/20 dark:text-red-300'
                                                        }`}>{test.risk_contribution} Risk</span>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-4 text-xs text-slate-500 dark:text-slate-400 ml-8.5 pl-0.5">
                                                    <span className="flex items-center gap-1"><span className="material-symbols-outlined text-[12px]">lab_profile</span> <strong className="text-slate-700 dark:text-slate-200">{test.value}</strong></span>
                                                    <span className="flex items-center gap-1"><span className="material-symbols-outlined text-[12px]">straighten</span> Normal: {test.normal_range}</span>
                                                </div>
                                                {test.explanation && (
                                                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 ml-8.5 pl-0.5 leading-relaxed">{test.explanation}</p>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Guidelines */}
                            {result.guidelines_cited && result.guidelines_cited.length > 0 && (
                                <div className="px-6 pt-4">
                                    <div className="flex items-center gap-2 mb-2.5">
                                        <span className="material-symbols-outlined text-slate-400 text-sm">menu_book</span>
                                        <h4 className="font-bold text-[11px] text-slate-500 dark:text-slate-400 uppercase tracking-wider">Referenced Guidelines</h4>
                                    </div>
                                    <div className="flex flex-wrap gap-2">
                                        {result.guidelines_cited.map((g, idx) => (
                                            <span key={idx} className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-50 dark:bg-slate-800/60 text-[10px] text-slate-600 dark:text-slate-400 rounded-lg border border-slate-100 dark:border-slate-700/40 font-medium">
                                                <span className="material-symbols-outlined text-[11px] text-slate-400">clinical_notes</span>
                                                {g}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Action Required Warning */}
                            {(result.prediction === 1 || result.needs_medical_attention) && (
                                <div className="px-6 pt-5">
                                    <div className="bg-gradient-to-r from-red-50 to-rose-50 dark:from-red-950/20 dark:to-rose-950/15 p-4 rounded-2xl border border-red-200/70 dark:border-red-800/40 flex items-start gap-3 shadow-sm shadow-red-100/50 dark:shadow-none">
                                        <div className="w-10 h-10 rounded-xl bg-red-100 dark:bg-red-900/30 flex items-center justify-center shrink-0">
                                            <span className="material-symbols-outlined text-red-600 dark:text-red-400 text-xl">medical_services</span>
                                        </div>
                                        <div>
                                            <p className="font-bold text-red-800 dark:text-red-300 text-sm">Medical Attention Recommended</p>
                                            <p className="text-red-700/80 dark:text-red-400/80 text-xs mt-1 leading-relaxed">
                                                Please consult a cardiologist for a thorough evaluation. This AI analysis is for informational purposes only and does not replace professional medical advice.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Action Bar & Footer */}
                            <div className="p-6 mt-2 space-y-4">
                                {/* Saved notice */}
                                {savedNotice && (
                                    <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 animate-in fade-in duration-300">
                                        <span className="material-symbols-outlined text-green-600 text-sm">check_circle</span>
                                        <p className="text-xs font-semibold text-green-700 dark:text-green-300">{savedNotice}</p>
                                    </div>
                                )}

                                {/* Processing time */}
                                {result.processing_time_ms && (
                                    <div className="flex items-center justify-center">
                                        <span className="text-[10px] text-slate-400 flex items-center gap-1">
                                            <span className="material-symbols-outlined text-xs">timer</span>
                                            {result.processing_time_ms.toFixed(0)}ms
                                        </span>
                                    </div>
                                )}

                                {/* Primary actions row */}
                                <div className="grid grid-cols-2 gap-3">
                                    {/* Save Results */}
                                    <button
                                        onClick={() => {
                                            const input: HeartDiseasePredictionRequest = {
                                                age: Number(formData.age) || 0,
                                                sex: formData.sex,
                                                chest_pain_type: formData.chest_pain_type,
                                                resting_bp_s: Number(formData.resting_bp_s) || 0,
                                                cholesterol: Number(formData.cholesterol) || 0,
                                                fasting_blood_sugar: formData.fasting_blood_sugar,
                                                resting_ecg: formData.resting_ecg,
                                                max_heart_rate: Number(formData.max_heart_rate) || 0,
                                                exercise_angina: formData.exercise_angina,
                                                oldpeak: Number(formData.oldpeak) || 0,
                                                st_slope: formData.st_slope,
                                            };
                                            saveAssessment(input, result);

                                            // Also save dashboard format
                                            const rl = (result.risk_level || '').toLowerCase();
                                            const riskLabel = (rl === 'high' || rl === 'critical') ? 'High Risk'
                                                : rl === 'moderate' ? 'Moderate Risk' : 'Low Risk';
                                            const scoreVal = result.probability != null
                                                ? Math.round((1 - result.probability) * 100)
                                                : ((rl === 'high' || rl === 'critical') ? 30 : rl === 'moderate' ? 60 : 85);
                                            localStorage.setItem('last_assessment', JSON.stringify({
                                                date: new Date().toISOString(),
                                                score: scoreVal,
                                                risk: riskLabel,
                                                details: result.clinical_interpretation || result.message || 'Assessment completed.',
                                                vitals: {
                                                    systolic: Number(formData.resting_bp_s) || 120,
                                                    cholesterol: Number(formData.cholesterol) || 200,
                                                },
                                            }));

                                            setSavedNotice('Results saved successfully!');
                                            setTimeout(() => setSavedNotice(null), 3000);
                                        }}
                                        className="py-3.5 bg-gradient-to-r from-slate-800 to-slate-900 dark:from-slate-600 dark:to-slate-700 text-white rounded-xl font-bold text-sm hover:from-slate-700 hover:to-slate-800 dark:hover:from-slate-500 dark:hover:to-slate-600 transition-all shadow-sm flex items-center justify-center gap-2"
                                    >
                                        <span className="material-symbols-outlined text-lg">save</span>
                                        Save Results
                                    </button>

                                    {/* Chat about results */}
                                    <button
                                        onClick={() => {
                                            const prob = (result.probability * 100).toFixed(1);
                                            const conf = result.confidence ? `${(result.confidence * 100).toFixed(0)}%` : 'N/A';
                                            const chestPainMap: Record<number, string> = { 1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic' };
                                            const ecgMap: Record<number, string> = { 0: 'Normal', 1: 'ST Abnormality', 2: 'LVH' };
                                            const slopeMap: Record<number, string> = { 1: 'Upsloping', 2: 'Flat', 3: 'Downsloping' };

                                            let msg = `Based on my heart risk assessment results:\n`;
                                            msg += `\n--- Patient Data ---\n`;
                                            msg += `- Age: ${formData.age}, Sex: ${formData.sex === 1 ? 'Male' : 'Female'}\n`;
                                            msg += `- Resting BP: ${formData.resting_bp_s} mm Hg\n`;
                                            msg += `- Cholesterol: ${formData.cholesterol} mg/dl\n`;
                                            msg += `- Max Heart Rate: ${formData.max_heart_rate} bpm\n`;
                                            msg += `- Fasting Blood Sugar: ${formData.fasting_blood_sugar === 1 ? '> 120' : '≤ 120'} mg/dl\n`;
                                            msg += `- Chest Pain Type: ${chestPainMap[formData.chest_pain_type] || formData.chest_pain_type}\n`;
                                            msg += `- Resting ECG: ${ecgMap[formData.resting_ecg] || formData.resting_ecg}\n`;
                                            msg += `- Exercise Angina: ${formData.exercise_angina === 1 ? 'Yes' : 'No'}\n`;
                                            msg += `- Oldpeak: ${formData.oldpeak}\n`;
                                            msg += `- ST Slope: ${slopeMap[formData.st_slope] || formData.st_slope}\n`;
                                            msg += `\n--- Model Prediction ---\n`;
                                            msg += `- Risk Level: ${result.risk_level || (result.prediction === 1 ? 'High' : 'Low')}\n`;
                                            msg += `- Probability: ${prob}%\n`;
                                            msg += `- Confidence: ${conf}\n`;
                                            msg += `- Prediction: ${result.prediction === 1 ? 'Heart Disease Likely' : 'Heart Disease Unlikely'}\n`;
                                            if (result.message) msg += `- Summary: ${result.message}\n`;
                                            if (result.clinical_interpretation) msg += `\n--- Clinical Interpretation ---\n${result.clinical_interpretation}\n`;
                                            msg += `\nWhat lifestyle changes and medical steps should I take to reduce my cardiovascular risk? Please provide specific, actionable recommendations based on my data above.`;
                                            setResult(null);
                                            navigate('/chat', { state: { autoSend: msg } });
                                        }}
                                        className="py-3.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-bold text-sm hover:from-indigo-700 hover:to-purple-700 transition-all shadow-sm shadow-indigo-500/20 flex items-center justify-center gap-2"
                                    >
                                        <span className="material-symbols-outlined text-lg">chat</span>
                                        Chat About This
                                    </button>
                                </div>

                                {/* Share row */}
                                <div className="relative">
                                    <button
                                        onClick={() => setShareMenuOpen(!shareMenuOpen)}
                                        className="w-full py-2.5 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 rounded-xl font-bold text-sm hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors flex items-center justify-center gap-2"
                                    >
                                        <span className="material-symbols-outlined text-lg">share</span>
                                        Share Results
                                        <span className="material-symbols-outlined text-sm">{shareMenuOpen ? 'expand_less' : 'expand_more'}</span>
                                    </button>

                                    {shareMenuOpen && (
                                        <div className="mt-2 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                                            <button
                                                onClick={() => {
                                                    const input: HeartDiseasePredictionRequest = {
                                                        age: Number(formData.age) || 0, sex: formData.sex,
                                                        chest_pain_type: formData.chest_pain_type,
                                                        resting_bp_s: Number(formData.resting_bp_s) || 0,
                                                        cholesterol: Number(formData.cholesterol) || 0,
                                                        fasting_blood_sugar: formData.fasting_blood_sugar,
                                                        resting_ecg: formData.resting_ecg,
                                                        max_heart_rate: Number(formData.max_heart_rate) || 0,
                                                        exercise_angina: formData.exercise_angina,
                                                        oldpeak: Number(formData.oldpeak) || 0,
                                                        st_slope: formData.st_slope,
                                                    };
                                                    const saved: SavedAssessment = { id: 'temp', timestamp: new Date().toISOString(), input, result };
                                                    shareViaWhatsApp(saved);
                                                    setShareMenuOpen(false);
                                                }}
                                                className="w-full px-4 py-3 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-left"
                                            >
                                                <span className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                                                    <span className="text-green-600 text-sm font-bold">W</span>
                                                </span>
                                                <div>
                                                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">WhatsApp</p>
                                                    <p className="text-[10px] text-slate-400">Share via WhatsApp message</p>
                                                </div>
                                            </button>

                                            <button
                                                onClick={async () => {
                                                    const input: HeartDiseasePredictionRequest = {
                                                        age: Number(formData.age) || 0, sex: formData.sex,
                                                        chest_pain_type: formData.chest_pain_type,
                                                        resting_bp_s: Number(formData.resting_bp_s) || 0,
                                                        cholesterol: Number(formData.cholesterol) || 0,
                                                        fasting_blood_sugar: formData.fasting_blood_sugar,
                                                        resting_ecg: formData.resting_ecg,
                                                        max_heart_rate: Number(formData.max_heart_rate) || 0,
                                                        exercise_angina: formData.exercise_angina,
                                                        oldpeak: Number(formData.oldpeak) || 0,
                                                        st_slope: formData.st_slope,
                                                    };
                                                    const saved: SavedAssessment = { id: 'temp', timestamp: new Date().toISOString(), input, result };
                                                    await shareViaLink(saved);
                                                    setSavedNotice('Report copied to clipboard!');
                                                    setTimeout(() => setSavedNotice(null), 3000);
                                                    setShareMenuOpen(false);
                                                }}
                                                className="w-full px-4 py-3 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-left border-t border-slate-100 dark:border-slate-700"
                                            >
                                                <span className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                                                    <span className="material-symbols-outlined text-blue-600 text-sm">content_copy</span>
                                                </span>
                                                <div>
                                                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">Copy to Clipboard</p>
                                                    <p className="text-[10px] text-slate-400">Copy report text to share anywhere</p>
                                                </div>
                                            </button>

                                            <button
                                                onClick={() => {
                                                    const input: HeartDiseasePredictionRequest = {
                                                        age: Number(formData.age) || 0, sex: formData.sex,
                                                        chest_pain_type: formData.chest_pain_type,
                                                        resting_bp_s: Number(formData.resting_bp_s) || 0,
                                                        cholesterol: Number(formData.cholesterol) || 0,
                                                        fasting_blood_sugar: formData.fasting_blood_sugar,
                                                        resting_ecg: formData.resting_ecg,
                                                        max_heart_rate: Number(formData.max_heart_rate) || 0,
                                                        exercise_angina: formData.exercise_angina,
                                                        oldpeak: Number(formData.oldpeak) || 0,
                                                        st_slope: formData.st_slope,
                                                    };
                                                    const saved: SavedAssessment = { id: 'temp', timestamp: new Date().toISOString(), input, result };
                                                    downloadAsPDF(saved);
                                                    setShareMenuOpen(false);
                                                }}
                                                className="w-full px-4 py-3 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-left border-t border-slate-100 dark:border-slate-700"
                                            >
                                                <span className="w-8 h-8 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                                                    <span className="material-symbols-outlined text-red-600 text-sm">picture_as_pdf</span>
                                                </span>
                                                <div>
                                                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">Download PDF</p>
                                                    <p className="text-[10px] text-slate-400">Generate a printable PDF report</p>
                                                </div>
                                            </button>

                                            <button
                                                onClick={async () => {
                                                    const input: HeartDiseasePredictionRequest = {
                                                        age: Number(formData.age) || 0, sex: formData.sex,
                                                        chest_pain_type: formData.chest_pain_type,
                                                        resting_bp_s: Number(formData.resting_bp_s) || 0,
                                                        cholesterol: Number(formData.cholesterol) || 0,
                                                        fasting_blood_sugar: formData.fasting_blood_sugar,
                                                        resting_ecg: formData.resting_ecg,
                                                        max_heart_rate: Number(formData.max_heart_rate) || 0,
                                                        exercise_angina: formData.exercise_angina,
                                                        oldpeak: Number(formData.oldpeak) || 0,
                                                        st_slope: formData.st_slope,
                                                    };
                                                    const saved: SavedAssessment = { id: 'temp', timestamp: new Date().toISOString(), input, result };
                                                    await shareAsPDF(saved);
                                                    setShareMenuOpen(false);
                                                }}
                                                className="w-full px-4 py-3 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-left border-t border-slate-100 dark:border-slate-700"
                                            >
                                                <span className="w-8 h-8 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                                                    <span className="material-symbols-outlined text-purple-600 text-sm">share</span>
                                                </span>
                                                <div>
                                                    <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">Share PDF</p>
                                                    <p className="text-[10px] text-slate-400">Share report as PDF via any app</p>
                                                </div>
                                            </button>
                                        </div>
                                    )}
                                </div>

                                {/* View Saved Results & Close */}
                                <div className="grid grid-cols-2 gap-2">
                                    <button
                                        onClick={() => {
                                            setResult(null);
                                            navigate('/saved-results');
                                        }}
                                        className="py-2.5 text-slate-600 dark:text-slate-300 rounded-xl font-semibold text-xs hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors flex items-center justify-center gap-1.5"
                                    >
                                        <span className="material-symbols-outlined text-sm">history</span>
                                        View Past Results
                                    </button>
                                    <button
                                        onClick={() => {
                                            setResult(null);
                                            navigate('/dashboard');
                                        }}
                                        className="py-2.5 text-slate-600 dark:text-slate-300 rounded-xl font-semibold text-xs hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors flex items-center justify-center gap-1.5"
                                    >
                                        <span className="material-symbols-outlined text-sm">home</span>
                                        Go to Dashboard
                                    </button>
                                </div>

                                <p className="text-[10px] text-slate-400 text-center px-4 leading-relaxed">
                                    *This AI model uses clinical data and machine learning to estimate risk. It should not replace professional medical advice, diagnosis, or treatment.
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AssessmentScreen;
