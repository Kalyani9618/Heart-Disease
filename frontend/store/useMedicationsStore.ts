/**
 * Medications Store - Zustand slice for medication management
 *
 * Centralizes:
 * - Active medications
 * - Dosing schedules
 * - Refill tracking
 * - Adherence tracking
 * - Drug interactions
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// ============================================================================
// Types
// ============================================================================

export type MedicationFrequency =
  | 'once_daily'
  | 'twice_daily'
  | 'three_times_daily'
  | 'four_times_daily'
  | 'every_other_day'
  | 'weekly'
  | 'as_needed'
  | 'custom';

export interface Medication {
  id: string;
  name: string;
  genericName?: string;
  brandName?: string;

  // Dosing
  dosage: string;
  unit: string;
  frequency: MedicationFrequency;
  schedules: MedicationSchedule[];

  // Instructions
  instructions?: string;
  withFood?: boolean;

  // Prescription info
  prescribedBy?: string;
  prescribedDate?: string;
  pharmacy?: string;
  refillsRemaining?: number;
  lastRefillDate?: string;
  nextRefillDate?: string;

  // Status
  isActive: boolean;
  startDate: string;
  endDate?: string;

  // Categorization
  category?: string;
  purpose?: string;

  // Metadata
  notes?: string;
  sideEffects?: string[];
  interactions?: DrugInteraction[];

  // Timestamps
  createdAt: string;
  updatedAt: string;
}

export interface MedicationSchedule {
  id: string;
  time: string; // HH:MM
  daysOfWeek?: number[]; // 0-6, Sunday = 0
  reminderEnabled: boolean;
  reminderMinutesBefore: number;
}

export interface DrugInteraction {
  medicationId: string;
  medicationName: string;
  severity: 'mild' | 'moderate' | 'severe' | 'contraindicated';
  description: string;
}

export interface AdherenceLog {
  id: string;
  medicationId: string;
  scheduledTime: string;
  status: 'taken' | 'missed' | 'skipped' | 'late';
  actualTime?: string;
  notes?: string;
}

export interface MedicationsState {
  // Data
  medications: Medication[];
  adherenceLogs: AdherenceLog[];

  // State
  isLoading: boolean;
  error: string | null;

  // Actions - Medications
  addMedication: (medication: Omit<Medication, 'id' | 'createdAt' | 'updatedAt'>) => string;
  updateMedication: (id: string, updates: Partial<Medication>) => void;
  removeMedication: (id: string) => void;
  deactivateMedication: (id: string, reason?: string) => void;
  reactivateMedication: (id: string) => void;

  // Actions - Schedules
  addSchedule: (medicationId: string, schedule: Omit<MedicationSchedule, 'id'>) => void;
  updateSchedule: (medicationId: string, scheduleId: string, updates: Partial<MedicationSchedule>) => void;
  removeSchedule: (medicationId: string, scheduleId: string) => void;

  // Actions - Adherence
  logAdherence: (log: Omit<AdherenceLog, 'id'>) => void;
  getTodaysSchedule: () => Array<{ medication: Medication; schedule: MedicationSchedule; status?: string }>;
  getAdherenceRate: (medicationId: string, days?: number) => number;

  // Actions - Query
  getActiveMedications: () => Medication[];
  getMedicationById: (id: string) => Medication | null;
  checkInteractions: (medicationIds: string[]) => DrugInteraction[];

  // Actions - Refills
  updateRefillInfo: (id: string, info: { refillsRemaining?: number; lastRefillDate?: string; nextRefillDate?: string }) => void;
  getMedicationsNeedingRefill: (daysAhead?: number) => Medication[];

  // State setters
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

// ============================================================================
// Helpers
// ============================================================================

const generateId = () => `med_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

// ============================================================================
// Initial State
// ============================================================================

const initialState = {
  medications: [],
  adherenceLogs: [],
  isLoading: false,
  error: null,
};

// ============================================================================
// Store
// ============================================================================

export const useMedicationsStore = create<MedicationsState>()(
  devtools(
    persist(
      immer((set, get) => ({
        ...initialState,

        // Medication Actions
        addMedication: (medication) => {
          const id = generateId();
          const now = new Date().toISOString();

          set((state) => {
            state.medications.push({
              ...medication,
              id,
              createdAt: now,
              updatedAt: now,
            });
          });

          return id;
        },

        updateMedication: (id, updates) => set((state) => {
          const index = state.medications.findIndex(m => m.id === id);
          if (index !== -1) {
            state.medications[index] = {
              ...state.medications[index],
              ...updates,
              updatedAt: new Date().toISOString(),
            };
          }
        }),

        removeMedication: (id) => set((state) => {
          state.medications = state.medications.filter(m => m.id !== id);
          state.adherenceLogs = state.adherenceLogs.filter(l => l.medicationId !== id);
        }),

        deactivateMedication: (id, reason) => set((state) => {
          const med = state.medications.find(m => m.id === id);
          if (med) {
            med.isActive = false;
            med.endDate = new Date().toISOString();
            if (reason) {
              med.notes = (med.notes || '') + `\nDeactivated: ${reason}`;
            }
            med.updatedAt = new Date().toISOString();
          }
        }),

        reactivateMedication: (id) => set((state) => {
          const med = state.medications.find(m => m.id === id);
          if (med) {
            med.isActive = true;
            med.endDate = undefined;
            med.updatedAt = new Date().toISOString();
          }
        }),

        // Schedule Actions
        addSchedule: (medicationId, schedule) => set((state) => {
          const med = state.medications.find(m => m.id === medicationId);
          if (med) {
            med.schedules.push({
              ...schedule,
              id: `sched_${Date.now()}`,
            });
            med.updatedAt = new Date().toISOString();
          }
        }),

        updateSchedule: (medicationId, scheduleId, updates) => set((state) => {
          const med = state.medications.find(m => m.id === medicationId);
          if (med) {
            const schedIndex = med.schedules.findIndex(s => s.id === scheduleId);
            if (schedIndex !== -1) {
              med.schedules[schedIndex] = { ...med.schedules[schedIndex], ...updates };
              med.updatedAt = new Date().toISOString();
            }
          }
        }),

        removeSchedule: (medicationId, scheduleId) => set((state) => {
          const med = state.medications.find(m => m.id === medicationId);
          if (med) {
            med.schedules = med.schedules.filter(s => s.id !== scheduleId);
            med.updatedAt = new Date().toISOString();
          }
        }),

        // Adherence Actions
        logAdherence: (log) => set((state) => {
          state.adherenceLogs.unshift({
            ...log,
            id: `adh_${Date.now()}`,
          });

          // Keep last 1000 logs
          if (state.adherenceLogs.length > 1000) {
            state.adherenceLogs = state.adherenceLogs.slice(0, 1000);
          }
        }),

        getTodaysSchedule: () => {
          const state = get();
          const today = new Date();
          const dayOfWeek = today.getDay();

          const schedule: Array<{ medication: Medication; schedule: MedicationSchedule; status?: string }> = [];

          for (const med of state.medications.filter(m => m.isActive)) {
            for (const sched of med.schedules) {
              // Check if this schedule applies today
              const appliestoday = !sched.daysOfWeek || sched.daysOfWeek.includes(dayOfWeek);

              if (appliestoday) {
                // Check if already logged
                const todayStr = today.toISOString().split('T')[0];
                const log = state.adherenceLogs.find(
                  l => l.medicationId === med.id &&
                       l.scheduledTime.startsWith(todayStr) &&
                       l.scheduledTime.includes(sched.time)
                );

                schedule.push({
                  medication: med,
                  schedule: sched,
                  status: log?.status,
                });
              }
            }
          }

          // Sort by time
          return schedule.sort((a, b) => a.schedule.time.localeCompare(b.schedule.time));
        },

        getAdherenceRate: (medicationId, days = 30) => {
          const state = get();
          const cutoff = new Date();
          cutoff.setDate(cutoff.getDate() - days);

          const logs = state.adherenceLogs.filter(
            l => l.medicationId === medicationId && new Date(l.scheduledTime) >= cutoff
          );

          if (logs.length === 0) return 100;

          const taken = logs.filter(l => l.status === 'taken' || l.status === 'late').length;
          return Math.round((taken / logs.length) * 100);
        },

        // Query Actions
        getActiveMedications: () => {
          return get().medications.filter(m => m.isActive);
        },

        getMedicationById: (id) => {
          return get().medications.find(m => m.id === id) || null;
        },

        checkInteractions: (medicationIds) => {
          const state = get();
          const interactions: DrugInteraction[] = [];

          for (const med of state.medications.filter(m => medicationIds.includes(m.id))) {
            if (med.interactions) {
              for (const interaction of med.interactions) {
                if (medicationIds.includes(interaction.medicationId)) {
                  interactions.push(interaction);
                }
              }
            }
          }

          return interactions;
        },

        // Refill Actions
        updateRefillInfo: (id, info) => set((state) => {
          const med = state.medications.find(m => m.id === id);
          if (med) {
            Object.assign(med, info);
            med.updatedAt = new Date().toISOString();
          }
        }),

        getMedicationsNeedingRefill: (daysAhead = 7) => {
          const state = get();
          const cutoff = new Date();
          cutoff.setDate(cutoff.getDate() + daysAhead);

          return state.medications.filter(m => {
            if (!m.isActive || !m.nextRefillDate) return false;
            return new Date(m.nextRefillDate) <= cutoff;
          });
        },

        // State setters
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),
      })),
      {
        name: 'medications-store',
        partialize: (state) => ({
          medications: state.medications,
          adherenceLogs: state.adherenceLogs.slice(0, 500),
        }),
      }
    ),
    { name: 'MedicationsStore' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectMedications = (state: MedicationsState) => state.medications;
export const selectActiveMedications = (state: MedicationsState) =>
  state.medications.filter(m => m.isActive);
export const selectAdherenceLogs = (state: MedicationsState) => state.adherenceLogs;

// ============================================================================
// Hooks
// ============================================================================

export const useActiveMedications = () => {
  const medications = useMedicationsStore(selectActiveMedications);
  return medications;
};

export const useTodaysMedicationSchedule = () => {
  const getTodaysSchedule = useMedicationsStore(state => state.getTodaysSchedule);
  const logAdherence = useMedicationsStore(state => state.logAdherence);

  return {
    schedule: getTodaysSchedule(),
    logTaken: (medicationId: string, scheduledTime: string) => logAdherence({
      medicationId,
      scheduledTime,
      status: 'taken',
      actualTime: new Date().toISOString(),
    }),
    logMissed: (medicationId: string, scheduledTime: string) => logAdherence({
      medicationId,
      scheduledTime,
      status: 'missed',
    }),
  };
};

export const useMedicationRefills = () => {
  const getMedicationsNeedingRefill = useMedicationsStore(state => state.getMedicationsNeedingRefill);
  const updateRefillInfo = useMedicationsStore(state => state.updateRefillInfo);

  return {
    needsRefill: getMedicationsNeedingRefill(),
    recordRefill: (id: string, refillsRemaining: number) => updateRefillInfo(id, {
      refillsRemaining,
      lastRefillDate: new Date().toISOString(),
      nextRefillDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days
    }),
  };
};
