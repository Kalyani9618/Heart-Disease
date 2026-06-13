/**
 * Zustand Store - Central State Management
 *
 * This module provides a comprehensive state management solution using Zustand.
 * It replaces scattered localStorage usage with centralized, type-safe stores.
 *
 * Features:
 * - Modular store slices (chat, user, vitals, appointments, medications)
 * - Automatic persistence with zustand/persist
 * - DevTools integration for debugging
 * - Immer for immutable updates
 * - Type-safe selectors and hooks
 *
 * Usage:
 * ```tsx
 * import { useChatStore, useVitalsStore, useUserStore } from './store';
 *
 * function MyComponent() {
 *   // Use individual stores
 *   const messages = useChatStore(state => state.messages);
 *   const { readings, add } = useBloodPressure();
 *   const user = useUserStore(state => state.user);
 *
 *   // Or use the combined store
 *   const { user, vitals, appointments } = useStore();
 * }
 * ```
 */

// Chat Store - Messages, sessions, model selection
export {
  useChatStore,
  type ChatState,
  type Message,
  type ChatSession,
  type ModelType,
  type Citation,
  // Selectors
  selectMessages,
  selectIsLoading,
  selectIsStreaming,
  selectSelectedModel,
  selectSessions,
  selectCurrentSession,
  // Actions
  chatActions,
} from './useChatStore';

// User Store - Profile, auth, notifications, preferences
export {
  useUserStore,
  type UserState,
  type UserProfile,
  type EmergencyContact,
  type InsuranceInfo,
  type Notification,
  type UserPreferences,
  // Selectors
  selectUser,
  selectIsAuthenticated,
  selectNotifications,
  selectUnreadCount,
  selectPreferences,
  selectTheme,
  selectLanguage,
  // Hooks
  useNotifications,
  useUserPreferences,
} from './useUserStore';

// Vitals Store - Health measurements, stats, goals
export {
  useVitalsStore,
  type VitalsState,
  type VitalType,
  type VitalReading,
  type VitalStats,
  type VitalGoal,
  // Selectors
  selectReadings,
  selectGoals,
  selectStats,
  selectIsSyncing,
  selectBloodPressureReadings,
  selectHeartRateReadings,
  selectWeightReadings,
  // Hooks
  useBloodPressure,
  useHeartRate,
  useWeight,
} from './useVitalsStore';

// Appointments Store - Scheduling, providers, reminders
export {
  useAppointmentsStore,
  type AppointmentsState,
  type Appointment,
  type AppointmentStatus,
  type AppointmentType,
  type Provider,
  // Selectors
  selectAppointments,
  selectProviders,
  selectFilter,
  selectUpcomingAppointments,
  selectNextAppointment,
  selectFavoriteProviders,
  // Hooks
  useUpcomingAppointments,
  useNextAppointment,
  useProviders,
} from './useAppointmentsStore';

// Medications Store - Medications, schedules, adherence
export {
  useMedicationsStore,
  type MedicationsState,
  type Medication,
  type MedicationFrequency,
  type MedicationSchedule,
  type DrugInteraction,
  type AdherenceLog,
  // Selectors
  selectMedications,
  selectActiveMedications,
  selectAdherenceLogs,
  // Hooks
  useActiveMedications,
  useTodaysMedicationSchedule,
  useMedicationRefills,
} from './useMedicationsStore';

// ============================================================================
// Combined Store Hook (for components that need multiple stores)
// ============================================================================

import { useChatStore } from './useChatStore';
import { useUserStore } from './useUserStore';
import { useVitalsStore } from './useVitalsStore';
import { useAppointmentsStore } from './useAppointmentsStore';
import { useMedicationsStore } from './useMedicationsStore';

/**
 * Combined store hook for components that need access to multiple stores
 *
 * @example
 * ```tsx
 * const { user, vitals, appointments } = useStore();
 * ```
 */
export const useStore = () => {
  const user = useUserStore(state => state.user);
  const isAuthenticated = useUserStore(state => state.isAuthenticated);
  const preferences = useUserStore(state => state.preferences);

  const messages = useChatStore(state => state.messages);
  const isLoading = useChatStore(state => state.isLoading);
  const selectedModel = useChatStore(state => state.selectedModel);

  const vitals = useVitalsStore(state => state.readings);
  const vitalStats = useVitalsStore(state => state.stats);

  const appointments = useAppointmentsStore(state => state.appointments);
  const providers = useAppointmentsStore(state => state.providers);

  const medications = useMedicationsStore(state => state.medications);
  const adherenceLogs = useMedicationsStore(state => state.adherenceLogs);

  return {
    // User
    user,
    isAuthenticated,
    preferences,

    // Chat
    messages,
    isLoading,
    selectedModel,

    // Vitals
    vitals,
    vitalStats,

    // Appointments
    appointments,
    providers,

    // Medications
    medications,
    adherenceLogs,
  };
};

// ============================================================================
// Store Reset (for logout, testing, etc.)
// ============================================================================

/**
 * Reset all stores to initial state
 * Useful for logout or testing
 */
export const resetAllStores = () => {
  // Each store should have a reset method
  // Call persist's clearStorage if needed
  localStorage.removeItem('chat-store');
  localStorage.removeItem('user-store');
  localStorage.removeItem('vitals-store');
  localStorage.removeItem('appointments-store');
  localStorage.removeItem('medications-store');
  localStorage.removeItem('health-store');

  // Reload to reset in-memory state
  if (typeof window !== 'undefined') {
    window.location.reload();
  }
};

// ============================================================================
// Store Hydration Hook
// ============================================================================

import { useEffect, useState } from 'react';

/**
 * Hook to ensure stores are hydrated from localStorage
 * Use this at the app root to prevent hydration mismatches
 *
 * @example
 * ```tsx
 * function App() {
 *   const isHydrated = useStoreHydration();
 *
 *   if (!isHydrated) {
 *     return <LoadingScreen />;
 *   }
 *
 *   return <MainApp />;
 * }
 * ```
 */
export const useStoreHydration = () => {
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    // Zustand persist middleware hydrates automatically
    // This hook just provides a way to wait for it
    setIsHydrated(true);
  }, []);

  return isHydrated;
};

// ============================================================================
// Store DevTools Helper
// ============================================================================

/**
 * Get a snapshot of all store states (for debugging)
 */
export const getStoreSnapshot = () => {
  return {
    chat: useChatStore.getState(),
    user: useUserStore.getState(),
    vitals: useVitalsStore.getState(),
    appointments: useAppointmentsStore.getState(),
    medications: useMedicationsStore.getState(),
  };
};

// Make available globally for debugging
if (typeof window !== 'undefined') {
  (window as any).__STORE_SNAPSHOT__ = getStoreSnapshot;
}
