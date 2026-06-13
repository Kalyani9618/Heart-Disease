import { useState, useEffect, useCallback } from 'react';
import { Medication } from '../types';
import { useAuth } from './useAuth';
import { useOfflineStatus } from './useOfflineStatus';
import { apiClient } from '../services/apiClient';

// Offline queue types
interface QueuedAction {
    id: string;
    type: 'add' | 'update' | 'delete';
    data: any;
    timestamp: number;
}

const QUEUE_KEY = 'medications_offline_queue';

// Get offline queue from localStorage
const getOfflineQueue = (): QueuedAction[] => {
    try {
        return JSON.parse(localStorage.getItem(QUEUE_KEY) || '[]');
    } catch {
        return [];
    }
};

// Save offline queue to localStorage
const saveOfflineQueue = (queue: QueuedAction[]) => {
    localStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
};

export const useMedications = () => {
    const { user } = useAuth();
    const { isOnline } = useOfflineStatus();
    const [medications, setMedications] = useState<Medication[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [pendingSync, setPendingSync] = useState(false);

    // Load medications from API
    useEffect(() => {
        if (!user) {
            setMedications([]);
            setLoading(false);
            return;
        }

        const loadMedications = async () => {
            setLoading(true);
            setError(null);

            try {
                const data = await apiClient.getMedications(user.id);
                // Map API response to Medication type with UI-only fields
                const meds: Medication[] = data.map(med => ({
                    ...med,
                    times: med.schedule || [],
                    takenToday: (med.schedule || []).map(() => false),
                }));
                setMedications(meds);
            } catch (err: any) {
                console.error('[Medications] Load failed:', err);
                setError(err?.message || 'Failed to load medications');
                setMedications([]);
            } finally {
                setLoading(false);
            }
        };

        loadMedications();
    }, [user]);

    // Process offline queue when coming back online
    const processOfflineQueue = useCallback(async () => {
        if (!user || !isOnline) return;

        const queue = getOfflineQueue();
        if (queue.length === 0) return;

        setPendingSync(true);
        console.log(`[Medications] Processing ${queue.length} queued actions...`);

        const failedActions: QueuedAction[] = [];

        for (const action of queue) {
            try {
                if (action.type === 'add') {
                    await apiClient.addMedication(user.id, action.data);
                } else if (action.type === 'update') {
                    await apiClient.updateMedication(user.id, action.data.id, action.data);
                } else if (action.type === 'delete') {
                    await apiClient.deleteMedication(user.id, action.data.id);
                }
                console.log(`[Medications] Synced ${action.type} action`);
            } catch (err) {
                console.error(`[Medications] Failed to sync ${action.type}:`, err);
                failedActions.push(action);
            }
        }

        // Keep only failed actions in queue
        saveOfflineQueue(failedActions);
        setPendingSync(false);

        // Refresh data from server
        if (failedActions.length < queue.length) {
            const data = await apiClient.getMedications(user.id);
            const meds: Medication[] = data.map(med => ({
                ...med,
                times: med.schedule || [],
                takenToday: (med.schedule || []).map(() => false),
            }));
            setMedications(meds);
        }
    }, [user, isOnline]);

    // Auto-sync when coming back online
    useEffect(() => {
        if (isOnline && user) {
            processOfflineQueue();
        }
    }, [isOnline, user, processOfflineQueue]);

    /**
     * Add medication with optimistic update
     */
    const addMedication = async (med: Omit<Medication, 'id'>) => {
        if (!user) {
            throw new Error('User not authenticated');
        }

        // Generate temporary ID for optimistic update
        const tempId = `temp-${Date.now()}`;
        const optimisticMed = { ...med, id: tempId } as Medication;

        // Optimistic update: add to UI immediately
        setMedications(prev => [...prev, optimisticMed]);

        // Prepare API payload (include quantity for persistence)
        const apiPayload = {
            name: med.name,
            dosage: med.dosage,
            schedule: med.times || [],
            frequency: med.frequency,
            notes: med.instructions,
            quantity: (med as Medication).quantity || 30,
        };

        // If offline, queue for later sync
        if (!isOnline) {
            const queue = getOfflineQueue();
            queue.push({
                id: tempId,
                type: 'add',
                data: apiPayload,
                timestamp: Date.now(),
            });
            saveOfflineQueue(queue);
            console.log('[Medications] Queued add for offline sync');
            return optimisticMed;
        }

        try {
            // Call API
            const savedMed = await apiClient.addMedication(user.id, apiPayload);

            // Map response back to Medication type
            const fullMed: Medication = {
                ...savedMed,
                times: savedMed.schedule || [],
                takenToday: (savedMed.schedule || []).map(() => false),
            };

            // Replace temporary with real data
            setMedications(prev =>
                prev.map(m => m.id === tempId ? fullMed : m)
            );

            console.log('[Medications] Added successfully:', fullMed);
            return fullMed;
        } catch (err: any) {
            // Rollback on error: remove optimistic entry
            setMedications(prev => prev.filter(m => m.id !== tempId));
            console.error('[Medications] Add failed:', err);
            throw err;
        }
    };

    /**
     * Update medication with optimistic update
     */
    const updateMedication = async (updatedMed: Medication) => {
        if (!user) {
            throw new Error('User not authenticated');
        }

        // Store backup for rollback
        const backup = medications.find(m => m.id === updatedMed.id);
        if (!backup) {
            throw new Error('Medication not found');
        }

        // Optimistic update
        setMedications(prev =>
            prev.map(m => m.id === updatedMed.id ? updatedMed : m)
        );

        // Prepare API payload (include quantity for persistence)
        const apiPayload = {
            name: updatedMed.name,
            dosage: updatedMed.dosage,
            schedule: updatedMed.times || [],
            frequency: updatedMed.frequency,
            notes: updatedMed.instructions,
            quantity: updatedMed.quantity,
        };

        // If offline, queue for later sync
        if (!isOnline) {
            const queue = getOfflineQueue();
            queue.push({
                id: updatedMed.id,
                type: 'update',
                data: { id: updatedMed.id, ...apiPayload },
                timestamp: Date.now(),
            });
            saveOfflineQueue(queue);
            console.log('[Medications] Queued update for offline sync');
            return updatedMed;
        }

        try {
            // Call API
            const saved = await apiClient.updateMedication(user.id, updatedMed.id, apiPayload);

            // Map response but preserve UI-only fields (takenToday, quantity)
            const fullMed: Medication = {
                ...saved,
                times: saved.schedule || [],
                // Preserve takenToday from the updated med instead of resetting
                takenToday: updatedMed.takenToday,
                // Preserve quantity from the updated med
                quantity: updatedMed.quantity ?? saved.quantity,
            };

            // Update with server response
            setMedications(prev =>
                prev.map(m => m.id === updatedMed.id ? fullMed : m)
            );

            console.log('[Medications] Updated successfully:', fullMed);
            return fullMed;
        } catch (err: any) {
            // Rollback on error: restore backup
            setMedications(prev =>
                prev.map(m => m.id === updatedMed.id ? backup : m)
            );
            console.error('[Medications] Update failed:', err);
            throw err;
        }
    };

    /**
     * Delete medication with optimistic update
     */
    const deleteMedication = async (id: string) => {
        if (!user) {
            throw new Error('User not authenticated');
        }

        // Store backup for rollback
        const backup = medications;

        // Optimistic delete: remove from UI immediately
        setMedications(prev => prev.filter(m => m.id !== id));

        // If offline, queue for later sync
        if (!isOnline) {
            const queue = getOfflineQueue();
            queue.push({
                id,
                type: 'delete',
                data: { id },
                timestamp: Date.now(),
            });
            saveOfflineQueue(queue);
            console.log('[Medications] Queued delete for offline sync');
            return;
        }

        try {
            // Call API
            await apiClient.deleteMedication(user.id, id);
            console.log('[Medications] Deleted successfully');
        } catch (err: any) {
            // Rollback on error: restore backup
            setMedications(backup);
            console.error('[Medications] Delete failed:', err);
            throw err;
        }
    };

    /**
     * Force reload medications from server
     */
    const refreshMedications = async () => {
        if (!user) return;

        setLoading(true);
        setError(null);

        try {
            const data = await apiClient.getMedications(user.id);
            // Map API response
            const meds: Medication[] = data.map(med => ({
                ...med,
                times: med.schedule || [],
                takenToday: (med.schedule || []).map(() => false),
            }));
            setMedications(meds);
        } catch (err: any) {
            console.error('[Medications] Refresh failed:', err);
            setError(err?.message || 'Failed to refresh medications');
        } finally {
            setLoading(false);
        }
    };

    return {
        medications,
        loading,
        error,
        pendingSync,
        isOnline,
        addMedication,
        updateMedication,
        deleteMedication,
        refreshMedications,
    };
};
