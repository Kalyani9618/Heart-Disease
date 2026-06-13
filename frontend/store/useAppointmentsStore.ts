/**
 * Appointments Store - Zustand slice for appointment management
 *
 * Centralizes:
 * - Appointment scheduling
 * - Doctor/provider management
 * - Reminders
 * - History
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// ============================================================================
// Types
// ============================================================================

export type AppointmentStatus = 'scheduled' | 'confirmed' | 'completed' | 'cancelled' | 'no_show';
export type AppointmentType = 'checkup' | 'follow_up' | 'specialist' | 'lab_work' | 'procedure' | 'telehealth';

export interface Appointment {
  id: string;
  title: string;
  type: AppointmentType;
  status: AppointmentStatus;

  // Scheduling
  dateTime: string;
  duration: number; // minutes

  // Provider
  provider: {
    id: string;
    name: string;
    specialty?: string;
    location?: string;
  };

  // Location
  location?: {
    name: string;
    address: string;
    phone?: string;
    directions?: string;
  };
  isVirtual?: boolean;
  virtualLink?: string;

  // Details
  reason?: string;
  notes?: string;

  // Preparation
  preparations?: string[];
  documentsNeeded?: string[];

  // Reminders
  reminders: {
    dayBefore: boolean;
    hourBefore: boolean;
    custom?: number; // minutes before
  };

  // Timestamps
  createdAt: string;
  updatedAt: string;
}

export interface Provider {
  id: string;
  name: string;
  specialty: string;
  phone?: string;
  email?: string;
  address?: string;
  notes?: string;
  isFavorite: boolean;
}

export interface AppointmentsState {
  // Data
  appointments: Appointment[];
  providers: Provider[];

  // State
  isLoading: boolean;
  error: string | null;

  // Filters/view
  filter: {
    status: AppointmentStatus | 'all';
    type: AppointmentType | 'all';
    dateRange: 'upcoming' | 'past' | 'all';
  };

  // Actions - Appointments
  addAppointment: (appointment: Omit<Appointment, 'id' | 'createdAt' | 'updatedAt'>) => string;
  updateAppointment: (id: string, updates: Partial<Appointment>) => void;
  cancelAppointment: (id: string, reason?: string) => void;
  completeAppointment: (id: string, notes?: string) => void;
  deleteAppointment: (id: string) => void;

  // Actions - Providers
  addProvider: (provider: Omit<Provider, 'id'>) => string;
  updateProvider: (id: string, updates: Partial<Provider>) => void;
  removeProvider: (id: string) => void;
  toggleFavoriteProvider: (id: string) => void;

  // Actions - Query
  getUpcoming: () => Appointment[];
  getPast: () => Appointment[];
  getByProvider: (providerId: string) => Appointment[];
  getByDate: (date: string) => Appointment[];

  // Actions - Filter
  setFilter: (filter: Partial<AppointmentsState['filter']>) => void;
  getFilteredAppointments: () => Appointment[];

  // State setters
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

// ============================================================================
// Helpers
// ============================================================================

const generateId = () => `appt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

const isUpcoming = (appointment: Appointment): boolean => {
  return new Date(appointment.dateTime) > new Date() &&
         appointment.status !== 'cancelled' &&
         appointment.status !== 'completed';
};

const isPast = (appointment: Appointment): boolean => {
  return new Date(appointment.dateTime) <= new Date() ||
         appointment.status === 'completed';
};

// ============================================================================
// Initial State
// ============================================================================

const initialState = {
  appointments: [],
  providers: [],
  isLoading: false,
  error: null,
  filter: {
    status: 'all' as const,
    type: 'all' as const,
    dateRange: 'upcoming' as const,
  },
};

// ============================================================================
// Store
// ============================================================================

export const useAppointmentsStore = create<AppointmentsState>()(
  devtools(
    persist(
      immer((set, get) => ({
        ...initialState,

        // Appointment Actions
        addAppointment: (appointment) => {
          const id = generateId();
          const now = new Date().toISOString();

          set((state) => {
            state.appointments.unshift({
              ...appointment,
              id,
              createdAt: now,
              updatedAt: now,
            });

            // Sort by date
            state.appointments.sort(
              (a, b) => new Date(a.dateTime).getTime() - new Date(b.dateTime).getTime()
            );
          });

          return id;
        },

        updateAppointment: (id, updates) => set((state) => {
          const index = state.appointments.findIndex(a => a.id === id);
          if (index !== -1) {
            state.appointments[index] = {
              ...state.appointments[index],
              ...updates,
              updatedAt: new Date().toISOString(),
            };
          }
        }),

        cancelAppointment: (id, reason) => set((state) => {
          const index = state.appointments.findIndex(a => a.id === id);
          if (index !== -1) {
            state.appointments[index].status = 'cancelled';
            state.appointments[index].notes = reason
              ? `Cancelled: ${reason}`
              : state.appointments[index].notes;
            state.appointments[index].updatedAt = new Date().toISOString();
          }
        }),

        completeAppointment: (id, notes) => set((state) => {
          const index = state.appointments.findIndex(a => a.id === id);
          if (index !== -1) {
            state.appointments[index].status = 'completed';
            if (notes) {
              state.appointments[index].notes =
                (state.appointments[index].notes || '') + '\n' + notes;
            }
            state.appointments[index].updatedAt = new Date().toISOString();
          }
        }),

        deleteAppointment: (id) => set((state) => {
          state.appointments = state.appointments.filter(a => a.id !== id);
        }),

        // Provider Actions
        addProvider: (provider) => {
          const id = `prov_${Date.now()}`;
          set((state) => {
            state.providers.push({ ...provider, id });
          });
          return id;
        },

        updateProvider: (id, updates) => set((state) => {
          const index = state.providers.findIndex(p => p.id === id);
          if (index !== -1) {
            state.providers[index] = { ...state.providers[index], ...updates };
          }
        }),

        removeProvider: (id) => set((state) => {
          state.providers = state.providers.filter(p => p.id !== id);
        }),

        toggleFavoriteProvider: (id) => set((state) => {
          const provider = state.providers.find(p => p.id === id);
          if (provider) {
            provider.isFavorite = !provider.isFavorite;
          }
        }),

        // Query Actions
        getUpcoming: () => {
          return get().appointments.filter(isUpcoming);
        },

        getPast: () => {
          return get().appointments.filter(isPast);
        },

        getByProvider: (providerId) => {
          return get().appointments.filter(a => a.provider.id === providerId);
        },

        getByDate: (date) => {
          const targetDate = new Date(date).toDateString();
          return get().appointments.filter(
            a => new Date(a.dateTime).toDateString() === targetDate
          );
        },

        // Filter Actions
        setFilter: (filter) => set((state) => {
          state.filter = { ...state.filter, ...filter };
        }),

        getFilteredAppointments: () => {
          const state = get();
          let filtered = [...state.appointments];

          // Status filter
          if (state.filter.status !== 'all') {
            filtered = filtered.filter(a => a.status === state.filter.status);
          }

          // Type filter
          if (state.filter.type !== 'all') {
            filtered = filtered.filter(a => a.type === state.filter.type);
          }

          // Date range filter
          if (state.filter.dateRange === 'upcoming') {
            filtered = filtered.filter(isUpcoming);
          } else if (state.filter.dateRange === 'past') {
            filtered = filtered.filter(isPast);
          }

          return filtered;
        },

        // State setters
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),
      })),
      {
        name: 'appointments-store',
        partialize: (state) => ({
          appointments: state.appointments,
          providers: state.providers,
        }),
      }
    ),
    { name: 'AppointmentsStore' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectAppointments = (state: AppointmentsState) => state.appointments;
export const selectProviders = (state: AppointmentsState) => state.providers;
export const selectFilter = (state: AppointmentsState) => state.filter;

export const selectUpcomingAppointments = (state: AppointmentsState) =>
  state.appointments.filter(isUpcoming);

export const selectNextAppointment = (state: AppointmentsState) =>
  state.appointments.filter(isUpcoming)[0] || null;

export const selectFavoriteProviders = (state: AppointmentsState) =>
  state.providers.filter(p => p.isFavorite);

// ============================================================================
// Hooks
// ============================================================================

export const useUpcomingAppointments = () => {
  const appointments = useAppointmentsStore(selectUpcomingAppointments);
  return appointments;
};

export const useNextAppointment = () => {
  const next = useAppointmentsStore(selectNextAppointment);
  return next;
};

export const useProviders = () => {
  const providers = useAppointmentsStore(selectProviders);
  const favorites = useAppointmentsStore(selectFavoriteProviders);
  const addProvider = useAppointmentsStore(state => state.addProvider);
  const toggleFavorite = useAppointmentsStore(state => state.toggleFavoriteProvider);

  return {
    providers,
    favorites,
    addProvider,
    toggleFavorite,
  };
};
