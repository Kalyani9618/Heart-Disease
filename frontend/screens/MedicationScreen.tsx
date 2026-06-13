import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiClient, APIError } from '../services/apiClient';
import { visionService, VisionServiceError } from '../services/visionService';
import { ParsedMedication } from '../services/api.types';
import { Medication } from '../types';
import { useMedications } from '../hooks/useMedications';
import ScreenHeader from '../components/ScreenHeader';
import { useConfirm } from '../components/ConfirmDialog';
import { scheduleLocalNotification, requestNotificationPermission } from '../services/nativeNotificationService';
import { useToast } from '../components/Toast';

/**
 * Convert service errors to user-friendly messages
 */
function getUserFriendlyError(error: unknown): string {
    if (error instanceof VisionServiceError) {
        return error.userMessage;
    }
    if (error instanceof Error) {
        console.error('[MedicationScreen] Scan error:', error);
        return 'Could not read the label. Please try a clearer photo.';
    }
    return 'Something went wrong. Please try again.';
}

const MedicationScreen: React.FC = () => {
    const navigate = useNavigate();
    const confirm = useConfirm();
    const { showToast } = useToast();
    const { medications, loading: isLoading, addMedication, updateMedication, deleteMedication: removeMedication } = useMedications();
    const [showAddModal, setShowAddModal] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isScanning, setIsScanning] = useState(false);
    const [analysisResult, setAnalysisResult] = useState<string | null>(null);
    const [scanError, setScanError] = useState<string | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Form State
    const [newMed, setNewMed] = useState({
        name: '',
        dosage: '',
        frequency: 'Daily',
        times: ['08:00'],
        instructions: '',
        quantity: '30'
    });

    // Add another time slot for the same medication
    const addTimeSlot = () => {
        setNewMed(prev => ({ ...prev, times: [...prev.times, '08:00'] }));
    };

    const removeTimeSlot = (index: number) => {
        if (newMed.times.length <= 1) return;
        setNewMed(prev => ({ ...prev, times: prev.times.filter((_, i) => i !== index) }));
    };

    const updateTimeSlot = (index: number, value: string) => {
        setNewMed(prev => {
            const newTimes = [...prev.times];
            newTimes[index] = value;
            return { ...prev, times: newTimes };
        });
    };

    /**
     * Schedule a daily medication reminder notification
     */
    const scheduleMedicationReminder = async (medName: string, medDosage: string, timeStr: string) => {
        try {
            // Request notification permission first
            const hasPermission = await requestNotificationPermission();
            if (!hasPermission) {
                console.log('[MedicationScreen] Notification permission denied');
                return false;
            }

            // Parse time string (HH:MM) and create scheduled date for today
            const [hours, minutes] = timeStr.split(':').map(Number);
            const scheduledAt = new Date();
            scheduledAt.setHours(hours, minutes, 0, 0);

            // If time already passed today, schedule for tomorrow
            if (scheduledAt <= new Date()) {
                scheduledAt.setDate(scheduledAt.getDate() + 1);
            }

            // Schedule the notification with daily repeat
            const success = await scheduleLocalNotification({
                id: Date.now(),
                title: `Time to take ${medName}`,
                body: `Take your ${medDosage} dose of ${medName}`,
                scheduledAt: scheduledAt,
                repeats: true,
                repeatInterval: 'day',
                channelId: 'cardio-reminders',
                extra: { type: 'medication', medName },
            });

            if (success) {
                console.log(`[MedicationScreen] Scheduled daily reminder for ${medName} at ${timeStr}`);
            }
            return success;
        } catch (err) {
            console.error('[MedicationScreen] Failed to schedule reminder:', err);
            return false;
        }
    };

    const handleAddMedication = async (): Promise<boolean> => {
        if (!newMed.name || !newMed.dosage) return false;

        const medData: Medication = {
            id: `med_${Date.now()}`,
            name: newMed.name,
            dosage: newMed.dosage,
            frequency: newMed.frequency,
            times: newMed.times,
            takenToday: newMed.times.map(() => false),
            instructions: newMed.instructions,
            quantity: parseInt(newMed.quantity) || 30,
        } as any;

        try {
            await addMedication(medData);

            // Schedule reminder notifications for each medication time
            let anyRemindersScheduled = false;
            for (const time of newMed.times) {
                const reminderScheduled = await scheduleMedicationReminder(newMed.name, newMed.dosage, time);
                if (reminderScheduled) anyRemindersScheduled = true;
            }
            if (anyRemindersScheduled) {
                showToast(`Reminders set for ${newMed.times.join(', ')}`, 'success');
            }

            // Reset form but keep modal open so user can add more medications
            setNewMed({ name: '', dosage: '', frequency: 'Daily', times: ['08:00'], instructions: '', quantity: '30' });
            showToast(`${medData.name} added successfully!`, 'success');
            return true;
        } catch (error) {
            console.error('[MedicationScreen] Failed to add medication:', error);
            showToast('Failed to save medication. Please try again.', 'error');
            return false;
        }
    };

    const toggleTaken = async (medId: string, timeIndex: number) => {
        const medIndex = medications.findIndex(m => m.id === medId);
        if (medIndex === -1) return;

        const med = medications[medIndex];
        const isTaking = !med.takenToday[timeIndex];
        const newTaken = [...med.takenToday];
        newTaken[timeIndex] = isTaking;

        let newQuantity = med.quantity || 0;
        if (isTaking && newQuantity > 0) newQuantity -= 1;
        else if (!isTaking) newQuantity += 1;

        const updatedMed = { ...med, takenToday: newTaken, quantity: newQuantity };
        try {
            await updateMedication(updatedMed);
        } catch (error) {
            console.error('[MedicationScreen] Failed to update medication:', error);
            showToast('Failed to update. Please try again.', 'error');
        }
    };

    const checkInteractions = async () => {
        if (medications.length < 2) {
            setAnalysisResult("Add at least two medications to check for interactions.");
            return;
        }

        setIsAnalyzing(true);
        setAnalysisResult(null);

        try {
            const response = await apiClient.medicationInsights({
                medications: medications.map(m => ({ name: m.name, dosage: m.dosage })),
                supplements: [],
                recent_vitals: {}
            });

            setAnalysisResult(response.insights || "No interactions found.");

        } catch (error) {
            console.error("API Error:", error);
            if (error instanceof APIError) {
                setAnalysisResult(`Error: ${error.message}`);
            } else {
                setAnalysisResult("Failed to analyze interactions. Please try again.");
            }
        } finally {
            setIsAnalyzing(false);
        }
    };

    const deleteMed = async (id: string) => {
        const confirmed = await confirm({
            title: "Delete Medication",
            message: "Are you sure you want to delete this medication?",
            confirmText: "Delete",
            variant: "danger"
        });
        if (confirmed) {
            removeMedication(id);
        }
    };

    // --- AI Label Scanner Logic ---
    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64String = reader.result as string;
                scanLabel(base64String.split(',')[1]);
            };
            reader.readAsDataURL(file);
        }
    };

    const scanLabel = async (base64Data: string) => {
        setIsScanning(true);
        setScanError(null);

        try {
            const result = await visionService.analyzeVision(
                base64Data,
                'document',
                'medication_label'
            );

            const medData: ParsedMedication = visionService.parseMedicationFromVisionResult(result);

            if (!medData.name || medData.name === 'Unknown Medication') {
                throw new Error('Could not extract medication name from image');
            }

            const newMed: Medication = {
                id: `med_${Date.now()}`,
                name: medData.name,
                dosage: medData.dosage || 'See label',
                frequency: medData.frequency || 'As directed',
                times: ['08:00'],
                takenToday: [false],
                instructions: medData.instructions,
                quantity: medData.quantity || 30,
            };

            await addMedication(newMed);
            // Reset form state and close modal
            setNewMed({ name: '', dosage: '', frequency: 'Daily', times: ['08:00'], instructions: '', quantity: '30' });
            setShowAddModal(false);
            showToast(`${newMed.name} added from scan!`, 'success');

        } catch (error) {
            console.error('[MedicationScreen] Scan error:', error);
            setScanError(getUserFriendlyError(error));
        } finally {
            setIsScanning(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-background-dark pb-24 font-sans overflow-x-hidden">
            <ScreenHeader
                title="Medicine Cabinet"
                subtitle="Track & Manage Prescriptions"
                rightIcon="add"
                onRightAction={() => setShowAddModal(true)}
            />

            <div className="max-w-4xl mx-auto p-4 space-y-6">

                {/* AI Safety Card */}
                <div className="bg-gradient-to-br from-indigo-600 to-violet-600 rounded-3xl p-6 text-white shadow-xl shadow-indigo-500/20 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl -mr-20 -mt-20"></div>

                    <div className="relative z-10">
                        <div className="flex items-start gap-4 mb-4">
                            <div className="w-12 h-12 bg-white/20 backdrop-blur-md rounded-2xl flex items-center justify-center shrink-0">
                                <span className="material-symbols-outlined text-2xl">science</span>
                            </div>
                            <div>
                                <h3 className="font-bold text-lg">AI Interaction Check</h3>
                                <p className="text-indigo-100 text-sm">Scan your medication list for potential drug interactions and safety warnings.</p>
                            </div>
                        </div>

                        {analysisResult ? (
                            <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl border border-white/20 animate-in fade-in slide-in-from-top-2">
                                <p className="text-sm leading-relaxed whitespace-pre-wrap">{analysisResult}</p>
                                <button
                                    onClick={() => setAnalysisResult(null)}
                                    className="mt-3 text-xs font-bold text-indigo-200 hover:text-white flex items-center gap-1"
                                >
                                    <span className="material-symbols-outlined text-sm">refresh</span> Check Again
                                </button>
                            </div>
                        ) : (
                            <button
                                onClick={checkInteractions}
                                disabled={isAnalyzing}
                                className="bg-white text-indigo-700 px-5 py-3 rounded-xl text-sm font-bold flex items-center gap-2 hover:bg-indigo-50 transition-colors disabled:opacity-70 shadow-sm"
                            >
                                {isAnalyzing ? (
                                    <>
                                        <span className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin"></span>
                                        Analyzing Safety...
                                    </>
                                ) : (
                                    <>
                                        <span className="material-symbols-outlined">health_and_safety</span>
                                        Check Interactions
                                    </>
                                )}
                            </button>
                        )}
                    </div>
                </div>

                {/* Medication List */}
                <div>
                    <div className="flex justify-between items-center mb-4 px-1">
                        <div className="flex items-center gap-2">
                            <h3 className="font-bold text-lg text-slate-800 dark:text-white">Active Medications</h3>
                            {isLoading && <span className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></span>}
                        </div>
                        <span className="text-xs font-medium px-2 py-1 bg-slate-100 dark:bg-slate-800 rounded-lg text-slate-500">{medications.length} total</span>
                    </div>

                    {!isLoading && medications.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {[...medications].sort((a, b) => (a.times?.[0] || '').localeCompare(b.times?.[0] || '')).map((med) => (
                                <div key={med.id} className="bg-white dark:bg-card-dark p-5 rounded-3xl border border-slate-100 dark:border-slate-800 shadow-sm hover:shadow-lg hover:scale-[1.01] transition-all duration-300 relative group">
                                    <button
                                        onClick={() => deleteMed(med.id)}
                                        className="absolute top-4 right-4 text-slate-300 hover:text-red-500 p-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white dark:bg-slate-800 rounded-full shadow-sm"
                                    >
                                        <span className="material-symbols-outlined text-lg">delete</span>
                                    </button>

                                    <div className="flex items-start gap-4 mb-4">
                                        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 shadow-inner ${parseInt(med.times[0]) < 12 ? 'bg-orange-50 text-orange-500 dark:bg-orange-900/10 dark:text-orange-400' :
                                            parseInt(med.times[0]) < 18 ? 'bg-blue-50 text-blue-500 dark:bg-blue-900/10 dark:text-blue-400' :
                                                'bg-indigo-50 text-indigo-500 dark:bg-indigo-900/10 dark:text-indigo-400'
                                            }`}>
                                            <span className="material-symbols-outlined text-3xl">medication_liquid</span>
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <h4 className="font-bold text-slate-900 dark:text-white text-lg truncate">{med.name}</h4>
                                            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{med.dosage}</p>
                                            <p className="text-xs text-slate-400 mt-1 truncate">{med.instructions || "Take as directed"}</p>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between pt-4 border-t border-slate-50 dark:border-slate-800/50">
                                        <div className="flex gap-2">
                                            {med.times.map((time, idx) => (
                                                <button
                                                    key={idx}
                                                    onClick={() => toggleTaken(med.id, idx)}
                                                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${med.takenToday[idx]
                                                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 ring-1 ring-green-600/20'
                                                        : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                                                        }`}
                                                >
                                                    <span className={`w-2 h-2 rounded-full ${med.takenToday[idx] ? 'bg-green-500' : 'bg-slate-400'}`}></span>
                                                    {time}
                                                </button>
                                            ))}
                                        </div>

                                        <div className={`text-xs font-bold px-2.5 py-1 rounded-full ${(med.quantity || 0) < 5
                                            ? 'bg-red-100 text-red-600 dark:bg-red-900/20 dark:text-red-400'
                                            : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
                                            }`}>
                                            {(med.quantity || 0) < 5 ? 'Low Stock' : `${med.quantity} left`}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : !isLoading && (
                        <div className="text-center py-16 bg-white dark:bg-card-dark rounded-3xl border border-dashed border-slate-200 dark:border-slate-800">
                            <div className="w-20 h-20 bg-slate-50 dark:bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                                <span className="material-symbols-outlined text-4xl text-slate-300">medication</span>
                            </div>
                            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">Cabinet is Empty</h3>
                            <p className="text-slate-500 dark:text-slate-400 text-sm max-w-xs mx-auto">Add your medications manually or scan a pill bottle to get started.</p>
                            <button
                                onClick={() => setShowAddModal(true)}
                                className="mt-6 text-primary font-bold text-sm hover:underline"
                            >
                                + Add First Medication
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Floating Action Button */}
            <button
                onClick={() => setShowAddModal(true)}
                className="fixed bottom-6 right-6 w-16 h-16 bg-gradient-to-r from-primary to-indigo-600 text-white rounded-full shadow-lg shadow-primary/30 flex items-center justify-center transition-transform hover:scale-105 hover:rotate-90 active:scale-95 z-20"
                aria-label="Add Medication"
            >
                <span className="material-symbols-outlined text-3xl">add</span>
            </button>

            {/* Add Modal */}
            {showAddModal && (
                <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4 sm:p-6 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={() => setShowAddModal(false)}>
                    <div className="bg-white dark:bg-slate-900 rounded-3xl w-full max-w-md shadow-2xl overflow-hidden animate-in slide-in-from-bottom-10" onClick={e => e.stopPropagation()}>
                        <div className="p-6 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center bg-slate-50/50 dark:bg-slate-800/50">
                            <h3 className="text-xl font-black text-slate-900 dark:text-white">Add Medication</h3>
                            <button onClick={() => setShowAddModal(false)} className="p-2 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-full transition-colors">
                                <span className="material-symbols-outlined text-slate-500">close</span>
                            </button>
                        </div>

                        <div className="p-6 max-h-[80vh] overflow-y-auto">
                            {/* AI Scan Button */}
                            <div className="mb-6">
                                <button
                                    onClick={() => fileInputRef.current?.click()}
                                    disabled={isScanning}
                                    className="w-full py-4 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-300 rounded-2xl border-2 border-indigo-200 dark:border-indigo-800 border-dashed flex flex-col items-center justify-center gap-2 hover:bg-indigo-100 dark:hover:bg-indigo-900/30 hover:border-indigo-300 transition-all group relative overflow-hidden"
                                >
                                    {isScanning ? (
                                        <>
                                            <span className="w-6 h-6 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin"></span>
                                            <span className="font-bold">Analyzing Label...</span>
                                        </>
                                    ) : (
                                        <>
                                            <div className="w-12 h-12 bg-white dark:bg-indigo-900 rounded-full flex items-center justify-center shadow-sm group-hover:scale-110 transition-transform">
                                                <span className="material-symbols-outlined text-2xl">document_scanner</span>
                                            </div>
                                            <span className="font-bold">Scan Pill Bottle</span>
                                            <span className="text-xs text-indigo-400">Auto-fill details from photo</span>
                                        </>
                                    )}
                                </button>
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    accept="image/*"
                                    capture="environment"
                                    className="hidden"
                                    onChange={handleFileChange}
                                />

                                {scanError && (
                                    <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3">
                                        <span className="material-symbols-outlined text-red-500 text-sm mt-0.5">error</span>
                                        <div className="flex-1">
                                            <p className="text-red-700 dark:text-red-400 text-sm font-medium">{scanError}</p>
                                            <button
                                                onClick={() => { setScanError(null); fileInputRef.current?.click(); }}
                                                className="text-red-600 dark:text-red-400 text-xs font-bold mt-1 hover:underline"
                                            >
                                                Try Another Photo
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="space-y-4">
                                <div>
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wide ml-1">Medication Name</label>
                                    <input
                                        type="text"
                                        className="w-full mt-1 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border-2 border-transparent focus:border-indigo-500/20 outline-none focus:ring-4 focus:ring-indigo-500/10 dark:text-white font-semibold placeholder:font-normal transition-all"
                                        placeholder="e.g. Metoprolol"
                                        value={newMed.name}
                                        onChange={e => setNewMed({ ...newMed, name: e.target.value })}
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-xs font-bold text-slate-500 uppercase tracking-wide ml-1">Dosage</label>
                                        <input
                                            type="text"
                                            className="w-full mt-1 p-3.5 rounded-xl bg-slate-100 dark:bg-slate-800 border-2 border-transparent focus:border-indigo-500/20 outline-none focus:ring-0 dark:text-white font-semibold placeholder:font-normal transition-all"
                                            placeholder="50mg"
                                            value={newMed.dosage}
                                            onChange={e => setNewMed({ ...newMed, dosage: e.target.value })}
                                        />
                                    </div>
                                    <div>
                                        <label className="text-xs font-bold text-slate-500 uppercase tracking-wide ml-1">Qty (Pills)</label>
                                        <input
                                            type="number"
                                            className="w-full mt-1 p-3.5 rounded-xl bg-slate-100 dark:bg-slate-800 border-2 border-transparent focus:border-indigo-500/20 outline-none focus:ring-0 dark:text-white font-semibold placeholder:font-normal transition-all"
                                            placeholder="30"
                                            value={newMed.quantity}
                                            onChange={e => setNewMed({ ...newMed, quantity: e.target.value })}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wide ml-1">Reminder Times</label>
                                    <div className="space-y-2 mt-1">
                                        {newMed.times.map((time, idx) => (
                                            <div key={idx} className="flex items-center gap-2">
                                                <input
                                                    type="time"
                                                    className="flex-1 p-3.5 rounded-xl bg-slate-100 dark:bg-slate-800 border-2 border-transparent focus:border-indigo-500/20 outline-none focus:ring-0 dark:text-white font-semibold transition-all"
                                                    value={time}
                                                    onChange={e => updateTimeSlot(idx, e.target.value)}
                                                />
                                                {newMed.times.length > 1 && (
                                                    <button
                                                        onClick={() => removeTimeSlot(idx)}
                                                        className="p-2 text-red-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-xl transition-colors"
                                                    >
                                                        <span className="material-symbols-outlined text-lg">remove_circle</span>
                                                    </button>
                                                )}
                                            </div>
                                        ))}
                                        <button
                                            onClick={addTimeSlot}
                                            className="w-full py-2.5 border-2 border-dashed border-indigo-200 dark:border-indigo-800 text-indigo-500 rounded-xl text-sm font-bold flex items-center justify-center gap-1.5 hover:bg-indigo-50 dark:hover:bg-indigo-900/10 transition-colors"
                                        >
                                            <span className="material-symbols-outlined text-sm">add</span>
                                            Add Another Time
                                        </button>
                                    </div>
                                </div>
                                <div>
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wide ml-1">Instructions (Optional)</label>
                                    <input
                                        type="text"
                                        className="w-full mt-1 p-3.5 rounded-xl bg-slate-100 dark:bg-slate-800 border-2 border-transparent focus:border-indigo-500/20 outline-none focus:ring-0 dark:text-white font-semibold placeholder:font-normal transition-all"
                                        placeholder="e.g. Take with food"
                                        value={newMed.instructions}
                                        onChange={e => setNewMed({ ...newMed, instructions: e.target.value })}
                                    />
                                </div>
                            </div>

                            <div className="flex flex-col gap-3 mt-8">
                                <button onClick={handleAddMedication} className="w-full py-3.5 bg-primary hover:bg-primary-dark text-white font-bold rounded-xl shadow-lg shadow-primary/30 transition-all active:scale-95 flex items-center justify-center gap-2">
                                    <span className="material-symbols-outlined text-lg">add</span>
                                    Save & Add Another
                                </button>
                                <button onClick={async () => { const saved = await handleAddMedication(); if (saved) setShowAddModal(false); }} className="w-full py-3.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-lg shadow-indigo-500/30 transition-all active:scale-95 flex items-center justify-center gap-2">
                                    <span className="material-symbols-outlined text-lg">check</span>
                                    Save & Done
                                </button>
                                <button onClick={() => setShowAddModal(false)} className="w-full py-3 text-slate-500 font-bold hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors">Cancel</button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MedicationScreen;
