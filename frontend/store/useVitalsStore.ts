/**
 * Vitals Store - Zustand slice for health vitals management
 *
 * Centralizes:
 * - Blood pressure readings
 * - Heart rate data
 * - Blood glucose
 * - Weight/BMI tracking
 * - Sleep data
 * - Activity data
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// ============================================================================
// Types
// ============================================================================

export type VitalType =
  | 'blood_pressure'
  | 'heart_rate'
  | 'blood_glucose'
  | 'weight'
  | 'temperature'
  | 'oxygen_saturation'
  | 'respiratory_rate'
  | 'steps'
  | 'sleep'
  | 'water_intake';

export interface VitalReading {
  id: string;
  type: VitalType;
  timestamp: string;
  value: number | { systolic: number; diastolic: number }; // BP has compound value
  unit: string;
  source: 'manual' | 'device' | 'imported';
  deviceId?: string;
  notes?: string;
  tags?: string[];
  context?: {
    activity?: 'resting' | 'post_exercise' | 'post_meal';
    position?: 'sitting' | 'standing' | 'lying';
    arm?: 'left' | 'right';
    mood?: string;
  };
}

export interface VitalStats {
  type: VitalType;
  average: number;
  min: number;
  max: number;
  trend: 'improving' | 'stable' | 'declining' | 'insufficient_data';
  lastReading?: VitalReading;
  readingCount: number;
}

export interface VitalGoal {
  type: VitalType;
  target: number | { systolic: number; diastolic: number };
  frequency: 'daily' | 'weekly' | 'monthly';
  enabled: boolean;
}

export interface VitalsState {
  // Data
  readings: VitalReading[];
  goals: VitalGoal[];

  // Computed/cached stats
  stats: Record<VitalType, VitalStats>;

  // State
  isLoading: boolean;
  isSyncing: boolean;
  lastSyncedAt: string | null;
  error: string | null;

  // Actions - CRUD
  addReading: (reading: Omit<VitalReading, 'id' | 'timestamp'>) => void;
  updateReading: (id: string, updates: Partial<VitalReading>) => void;
  deleteReading: (id: string) => void;
  bulkAddReadings: (readings: Omit<VitalReading, 'id'>[]) => void;

  // Actions - Goals
  setGoal: (goal: VitalGoal) => void;
  removeGoal: (type: VitalType) => void;

  // Actions - Query
  getReadings: (type: VitalType, days?: number) => VitalReading[];
  getLatestReading: (type: VitalType) => VitalReading | null;
  getStats: (type: VitalType) => VitalStats | null;

  // Actions - Sync
  syncFromDevice: (deviceId: string) => Promise<void>;
  exportData: (types: VitalType[], format: 'json' | 'csv') => string;

  // State setters
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Internal
  _recalculateStats: (type: VitalType) => void;
}

// ============================================================================
// Helpers
// ============================================================================

const generateId = () => `vital_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

const calculateStats = (readings: VitalReading[], type: VitalType): VitalStats => {
  const typeReadings = readings.filter(r => r.type === type);

  if (typeReadings.length === 0) {
    return {
      type,
      average: 0,
      min: 0,
      max: 0,
      trend: 'insufficient_data',
      readingCount: 0,
    };
  }

  // Extract numeric values (handle BP specially)
  const values = typeReadings.map(r => {
    if (type === 'blood_pressure' && typeof r.value === 'object') {
      return (r.value.systolic + r.value.diastolic) / 2; // Use MAP approximation
    }
    return r.value as number;
  });

  const sum = values.reduce((a, b) => a + b, 0);
  const average = sum / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);

  // Calculate trend (compare recent vs older readings)
  let trend: VitalStats['trend'] = 'insufficient_data';
  if (typeReadings.length >= 5) {
    const recent = values.slice(0, Math.floor(values.length / 2));
    const older = values.slice(Math.floor(values.length / 2));
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;

    const change = ((recentAvg - olderAvg) / olderAvg) * 100;

    if (Math.abs(change) < 5) {
      trend = 'stable';
    } else if (change > 0) {
      // Higher is not always better (depends on vital type)
      const higherIsBetter = ['steps', 'oxygen_saturation'].includes(type);
      trend = higherIsBetter ? 'improving' : 'declining';
    } else {
      const lowerIsBetter = ['blood_pressure', 'weight', 'blood_glucose'].includes(type);
      trend = lowerIsBetter ? 'improving' : 'declining';
    }
  }

  return {
    type,
    average: Math.round(average * 10) / 10,
    min,
    max,
    trend,
    lastReading: typeReadings[0],
    readingCount: typeReadings.length,
  };
};

// ============================================================================
// Initial State
// ============================================================================

const initialState = {
  readings: [],
  goals: [],
  stats: {} as Record<VitalType, VitalStats>,
  isLoading: false,
  isSyncing: false,
  lastSyncedAt: null,
  error: null,
};

// ============================================================================
// Store
// ============================================================================

export const useVitalsStore = create<VitalsState>()(
  devtools(
    persist(
      immer((set, get) => ({
        ...initialState,

        // CRUD Actions
        addReading: (reading) => set((state) => {
          const newReading: VitalReading = {
            ...reading,
            id: generateId(),
            timestamp: new Date().toISOString(),
          };

          // Insert at beginning (sorted by time desc)
          state.readings.unshift(newReading);

          // Recalculate stats for this type
          state.stats[reading.type] = calculateStats(state.readings, reading.type);
        }),

        updateReading: (id, updates) => set((state) => {
          const index = state.readings.findIndex(r => r.id === id);
          if (index !== -1) {
            const oldType = state.readings[index].type;
            state.readings[index] = { ...state.readings[index], ...updates };

            // Recalculate stats
            state.stats[oldType] = calculateStats(state.readings, oldType);
            if (updates.type && updates.type !== oldType) {
              state.stats[updates.type] = calculateStats(state.readings, updates.type);
            }
          }
        }),

        deleteReading: (id) => set((state) => {
          const reading = state.readings.find(r => r.id === id);
          if (reading) {
            state.readings = state.readings.filter(r => r.id !== id);
            state.stats[reading.type] = calculateStats(state.readings, reading.type);
          }
        }),

        bulkAddReadings: (readings) => set((state) => {
          const newReadings = readings.map(r => ({
            ...r,
            id: generateId(),
          }));

          state.readings = [...newReadings, ...state.readings].sort(
            (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
          );

          // Recalculate stats for affected types
          const types = new Set(readings.map(r => r.type));
          types.forEach(type => {
            state.stats[type] = calculateStats(state.readings, type);
          });
        }),

        // Goal Actions
        setGoal: (goal) => set((state) => {
          const index = state.goals.findIndex(g => g.type === goal.type);
          if (index !== -1) {
            state.goals[index] = goal;
          } else {
            state.goals.push(goal);
          }
        }),

        removeGoal: (type) => set((state) => {
          state.goals = state.goals.filter(g => g.type !== type);
        }),

        // Query Actions
        getReadings: (type, days = 30) => {
          const state = get();
          const cutoff = new Date();
          cutoff.setDate(cutoff.getDate() - days);

          return state.readings.filter(
            r => r.type === type && new Date(r.timestamp) >= cutoff
          );
        },

        getLatestReading: (type) => {
          const state = get();
          return state.readings.find(r => r.type === type) || null;
        },

        getStats: (type) => {
          const state = get();
          return state.stats[type] || null;
        },

        // Sync Actions
        syncFromDevice: async (deviceId) => {
          set({ isSyncing: true, error: null });
          try {
            // In production, call device sync API
            // const readings = await deviceService.syncReadings(deviceId);
            // get().bulkAddReadings(readings);

            await new Promise(r => setTimeout(r, 1000)); // Simulate

            set((state) => {
              state.isSyncing = false;
              state.lastSyncedAt = new Date().toISOString();
            });
          } catch (error) {
            set({
              error: error instanceof Error ? error.message : 'Sync failed',
              isSyncing: false,
            });
          }
        },

        exportData: (types, format) => {
          const state = get();
          const data = state.readings.filter(r => types.includes(r.type));

          if (format === 'json') {
            return JSON.stringify(data, null, 2);
          }

          // CSV format
          const headers = ['id', 'type', 'timestamp', 'value', 'unit', 'source', 'notes'];
          const rows = data.map(r => [
            r.id,
            r.type,
            r.timestamp,
            typeof r.value === 'object' ? `${r.value.systolic}/${r.value.diastolic}` : r.value,
            r.unit,
            r.source,
            r.notes || '',
          ]);

          return [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        },

        // State setters
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),

        // Internal
        _recalculateStats: (type) => set((state) => {
          state.stats[type] = calculateStats(state.readings, type);
        }),
      })),
      {
        name: 'vitals-store',
        partialize: (state) => ({
          readings: state.readings.slice(0, 1000), // Keep last 1000
          goals: state.goals,
        }),
      }
    ),
    { name: 'VitalsStore' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectReadings = (state: VitalsState) => state.readings;
export const selectGoals = (state: VitalsState) => state.goals;
export const selectStats = (state: VitalsState) => state.stats;
export const selectIsSyncing = (state: VitalsState) => state.isSyncing;

// Type-specific selectors
export const selectBloodPressureReadings = (state: VitalsState) =>
  state.readings.filter(r => r.type === 'blood_pressure');

export const selectHeartRateReadings = (state: VitalsState) =>
  state.readings.filter(r => r.type === 'heart_rate');

export const selectWeightReadings = (state: VitalsState) =>
  state.readings.filter(r => r.type === 'weight');

// ============================================================================
// Convenience Hooks
// ============================================================================

export const useBloodPressure = () => {
  const readings = useVitalsStore(selectBloodPressureReadings);
  const stats = useVitalsStore(state => state.stats.blood_pressure);
  const addReading = useVitalsStore(state => state.addReading);

  return {
    readings,
    stats,
    add: (systolic: number, diastolic: number, notes?: string) => addReading({
      type: 'blood_pressure',
      value: { systolic, diastolic },
      unit: 'mmHg',
      source: 'manual',
      notes,
    }),
  };
};

export const useHeartRate = () => {
  const readings = useVitalsStore(selectHeartRateReadings);
  const stats = useVitalsStore(state => state.stats.heart_rate);
  const addReading = useVitalsStore(state => state.addReading);

  return {
    readings,
    stats,
    add: (bpm: number, activity?: string, notes?: string) => addReading({
      type: 'heart_rate',
      value: bpm,
      unit: 'bpm',
      source: 'manual',
      notes,
      context: activity ? { activity: activity as any } : undefined,
    }),
  };
};

export const useWeight = () => {
  const readings = useVitalsStore(selectWeightReadings);
  const stats = useVitalsStore(state => state.stats.weight);
  const addReading = useVitalsStore(state => state.addReading);
  const preferences = { unit: 'lbs' }; // Would get from user store

  return {
    readings,
    stats,
    add: (weight: number, notes?: string) => addReading({
      type: 'weight',
      value: weight,
      unit: preferences.unit,
      source: 'manual',
      notes,
    }),
  };
};
