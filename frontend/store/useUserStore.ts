/**
 * User Store - Zustand slice for user profile and settings
 *
 * Centralizes:
 * - User profile data
 * - Authentication state
 * - User preferences/settings
 * - Notifications
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { apiClient } from '../services/apiClient';
import { authService } from '../services/authService';

// ============================================================================
// Types
// ============================================================================

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  dateOfBirth?: string;
  gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say';
  bloodType?: 'A+' | 'A-' | 'B+' | 'B-' | 'AB+' | 'AB-' | 'O+' | 'O-';
  height?: number; // in cm
  weight?: number; // in kg
  conditions?: string[];
  allergies?: string[];
  emergencyContact?: EmergencyContact;
  insuranceInfo?: InsuranceInfo;
  createdAt: string;
  updatedAt: string;
}

export interface EmergencyContact {
  name: string;
  relation: string;
  phone: string;
  email?: string;
}

export interface InsuranceInfo {
  provider: string;
  policyNumber: string;
  groupNumber?: string;
  expirationDate?: string;
}

export interface Notification {
  id: string;
  type: 'appointment' | 'medication' | 'vital' | 'alert' | 'info';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: 'en' | 'es' | 'fr' | 'te';
  units: {
    temperature: 'celsius' | 'fahrenheit';
    weight: 'kg' | 'lbs';
    height: 'cm' | 'ft';
    bloodGlucose: 'mg/dL' | 'mmol/L';
  };
  notifications: {
    enabled: boolean;
    appointments: boolean;
    medications: boolean;
    vitals: boolean;
    insights: boolean;
  };
  privacy: {
    shareDataForResearch: boolean;
    analyticsEnabled: boolean;
  };
}

export interface UserState {
  // Profile
  user: UserProfile | null;
  isAuthenticated: boolean;

  // Notifications
  notifications: Notification[];
  unreadCount: number;

  // Preferences
  preferences: UserPreferences;

  // State
  isLoading: boolean;
  error: string | null;

  // Profile actions
  setUser: (user: UserProfile) => void;
  updateUser: (updates: Partial<UserProfile>) => void;
  clearUser: () => void;

  // Auth actions
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;

  // Notification actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // Preference actions
  setPreferences: (preferences: Partial<UserPreferences>) => void;
  setTheme: (theme: UserPreferences['theme']) => void;
  setLanguage: (language: UserPreferences['language']) => void;
  setUnits: (units: Partial<UserPreferences['units']>) => void;

  // State setters
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

// ============================================================================
// Initial State
// ============================================================================

const generateId = () => `ntf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

const defaultPreferences: UserPreferences = {
  theme: 'system',
  language: 'en',
  units: {
    temperature: 'fahrenheit',
    weight: 'lbs',
    height: 'ft',
    bloodGlucose: 'mg/dL',
  },
  notifications: {
    enabled: true,
    appointments: true,
    medications: true,
    vitals: true,
    insights: true,
  },
  privacy: {
    shareDataForResearch: false,
    analyticsEnabled: true,
  },
};

const initialState = {
  user: null,
  isAuthenticated: false,
  notifications: [],
  unreadCount: 0,
  preferences: defaultPreferences,
  isLoading: false,
  error: null,
};

// ============================================================================
// Store
// ============================================================================

export const useUserStore = create<UserState>()(
  devtools(
    persist(
      immer((set, get) => ({
        ...initialState,

        // Profile actions
        setUser: (user) => set((state) => {
          state.user = user;
          state.isAuthenticated = true;
        }),

        updateUser: (updates) => set((state) => {
          if (state.user) {
            state.user = {
              ...state.user,
              ...updates,
              updatedAt: new Date().toISOString(),
            };
          }
        }),

        clearUser: () => set((state) => {
          state.user = null;
          state.isAuthenticated = false;
        }),

        // Auth actions
        login: async (email, password) => {
          set({ isLoading: true, error: null });
          try {
            const response = await apiClient.login(email, password);

            // Store auth tokens
            if (response.token) {
              authService.setToken(response.token);
            }
            if (response.refresh_token) {
              authService.setRefreshToken(response.refresh_token);
            }
            if (response.user) {
              authService.setUser(response.user);
            }

            const user: UserProfile = {
              id: response.user?.id || '',
              name: response.user?.name || email.split('@')[0],
              email: response.user?.email || email,
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            };

            set((state) => {
              state.user = user;
              state.isAuthenticated = true;
              state.isLoading = false;
            });

            return true;
          } catch (error) {
            set({
              error: error instanceof Error ? error.message : 'Login failed',
              isLoading: false,
            });
            return false;
          }
        },

        logout: () => set((state) => {
          state.user = null;
          state.isAuthenticated = false;
          state.notifications = [];
          state.unreadCount = 0;
        }),

        // Notification actions
        addNotification: (notification) => set((state) => {
          const newNotification: Notification = {
            ...notification,
            id: generateId(),
            timestamp: new Date().toISOString(),
            read: false,
          };
          state.notifications.unshift(newNotification);
          state.unreadCount += 1;
        }),

        markAsRead: (id) => set((state) => {
          const notification = state.notifications.find(n => n.id === id);
          if (notification && !notification.read) {
            notification.read = true;
            state.unreadCount = Math.max(0, state.unreadCount - 1);
          }
        }),

        markAllAsRead: () => set((state) => {
          state.notifications.forEach(n => { n.read = true; });
          state.unreadCount = 0;
        }),

        removeNotification: (id) => set((state) => {
          const notification = state.notifications.find(n => n.id === id);
          if (notification && !notification.read) {
            state.unreadCount = Math.max(0, state.unreadCount - 1);
          }
          state.notifications = state.notifications.filter(n => n.id !== id);
        }),

        clearNotifications: () => set((state) => {
          state.notifications = [];
          state.unreadCount = 0;
        }),

        // Preference actions
        setPreferences: (preferences) => set((state) => {
          state.preferences = { ...state.preferences, ...preferences };
        }),

        setTheme: (theme) => set((state) => {
          state.preferences.theme = theme;
        }),

        setLanguage: (language) => set((state) => {
          state.preferences.language = language;
        }),

        setUnits: (units) => set((state) => {
          state.preferences.units = { ...state.preferences.units, ...units };
        }),

        // State setters
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),
      })),
      {
        name: 'user-store',
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
          preferences: state.preferences,
          notifications: state.notifications.slice(0, 50), // Keep last 50
        }),
      }
    ),
    { name: 'UserStore' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectUser = (state: UserState) => state.user;
export const selectIsAuthenticated = (state: UserState) => state.isAuthenticated;
export const selectNotifications = (state: UserState) => state.notifications;
export const selectUnreadCount = (state: UserState) => state.unreadCount;
export const selectPreferences = (state: UserState) => state.preferences;
export const selectTheme = (state: UserState) => state.preferences.theme;
export const selectLanguage = (state: UserState) => state.preferences.language;

// ============================================================================
// Hooks for specific data
// ============================================================================

export const useNotifications = () => {
  const notifications = useUserStore(selectNotifications);
  const unreadCount = useUserStore(selectUnreadCount);
  const addNotification = useUserStore(state => state.addNotification);
  const markAsRead = useUserStore(state => state.markAsRead);
  const markAllAsRead = useUserStore(state => state.markAllAsRead);

  return {
    notifications,
    unreadCount,
    addNotification,
    markAsRead,
    markAllAsRead,
  };
};

export const useUserPreferences = () => {
  const preferences = useUserStore(selectPreferences);
  const setPreferences = useUserStore(state => state.setPreferences);
  const setTheme = useUserStore(state => state.setTheme);
  const setLanguage = useUserStore(state => state.setLanguage);
  const setUnits = useUserStore(state => state.setUnits);

  return {
    preferences,
    setPreferences,
    setTheme,
    setLanguage,
    setUnits,
  };
};
