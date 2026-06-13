/**
 * Notification Service
 *
 * Handles notification preferences and delivery (Push, Email, WhatsApp).
 * Also manages weekly summary preferences.
 *
 * Healthcare-grade error handling:
 * - No PHI or stack traces exposed to users
 * - All errors logged for developers
 * - User-friendly error messages
 */

import {
    DeviceRegistrationRequest,
    WeeklySummaryPreferences,
    NotificationPlatform,
} from './api.types';
import { authService } from './authService';

// Use environment variable or fallback to localhost
const API_BASE_URL = (import.meta as any).env?.VITE_NLP_SERVICE_URL || 'http://localhost:5001';

// ============================================================================
// Error Handling
// ============================================================================

export class NotificationServiceError extends Error {
    constructor(
        public code: string,
        public userMessage: string,
        public developerMessage: string,
        public statusCode?: number
    ) {
        super(userMessage);
        this.name = 'NotificationServiceError';
    }
}

/**
 * Convert API errors to user-friendly messages
 */
function handleError(error: unknown, context: string): never {
    console.error(`[NotificationService] ${context}:`, error);

    if (error instanceof NotificationServiceError) {
        throw error;
    }

    if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new NotificationServiceError(
            'NETWORK_ERROR',
            'Unable to connect. Please check your internet connection.',
            `Network error during ${context}: ${error.message}`
        );
    }

    if (error && typeof error === 'object' && 'status' in error) {
        const status = (error as { status: number }).status;
        const detail = (error as { detail?: string }).detail;

        switch (status) {
            case 400:
                throw new NotificationServiceError(
                    'INVALID_REQUEST',
                    'Invalid input. Please check your settings.',
                    detail || 'Bad request',
                    400
                );
            case 503:
                throw new NotificationServiceError(
                    'SERVICE_UNAVAILABLE',
                    'Notification service is temporarily unavailable. Please try again later.',
                    detail || 'Service unavailable',
                    503
                );
            default:
                throw new NotificationServiceError(
                    'SERVER_ERROR',
                    'Something went wrong. Please try again.',
                    detail || `HTTP ${status}`,
                    status
                );
        }
    }

    throw new NotificationServiceError(
        'UNKNOWN_ERROR',
        'Something went wrong. Please try again.',
        error instanceof Error ? error.message : String(error)
    );
}

// ============================================================================
// Push Notification Registration
// ============================================================================

/**
 * Register a device for push notifications
 *
 * @param userId - User identifier
 * @param deviceToken - Device push token from the browser/app
 * @param platform - Platform type
 * @returns Registration result
 */
export async function registerDevice(
    userId: string,
    deviceToken: string,
    platform: NotificationPlatform
): Promise<{ status: string; device_id: string }> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/notifications/register-device`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify({
                user_id: userId,
                device_token: deviceToken,
                platform,
            } as DeviceRegistrationRequest & { user_id: string }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'registerDevice');
    }
}

/**
 * Request browser push notification permission
 * Returns the subscription token if granted
 */
export async function requestPushPermission(): Promise<string | null> {
    if (!('Notification' in window)) {
        console.warn('[NotificationService] Push notifications not supported');
        return null;
    }

    if (!('serviceWorker' in navigator)) {
        console.warn('[NotificationService] Service workers not supported');
        return null;
    }

    try {
        const permission = await Notification.requestPermission();

        if (permission !== 'granted') {
            console.log('[NotificationService] Permission not granted');
            return null;
        }

        // Get service worker registration
        const registration = await navigator.serviceWorker.ready;

        // Subscribe to push (would need VAPID key from server in production)
        const subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            // In production, applicationServerKey should come from backend
            applicationServerKey: undefined,
        }).catch(() => null);

        if (subscription) {
            // Convert to a token representation
            return JSON.stringify(subscription.toJSON());
        }

        return null;
    } catch (error) {
        console.error('[NotificationService] Push permission error:', error);
        return null;
    }
}

// ============================================================================
// Weekly Summary Preferences
// ============================================================================

/**
 * Get weekly summary preferences for a user
 *
 * @param userId - User identifier
 * @returns Current preferences
 */
export async function getWeeklySummaryPreferences(
    userId: string
): Promise<WeeklySummaryPreferences> {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/weekly-summary/preferences?user_id=${userId}`,
            {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
                },
            }
        );

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'getWeeklySummaryPreferences');
    }
}

/**
 * Update weekly summary preferences
 *
 * @param userId - User identifier
 * @param preferences - New preferences
 * @returns Updated preferences
 */
export async function updateWeeklySummaryPreferences(
    userId: string,
    preferences: WeeklySummaryPreferences
): Promise<WeeklySummaryPreferences> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/weekly-summary/preferences`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify({
                user_id: userId,
                ...preferences,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'updateWeeklySummaryPreferences');
    }
}

/**
 * Manually trigger a weekly summary
 *
 * @param userId - User identifier
 * @param channels - Optional specific channels to use
 * @returns Trigger result
 */
export async function triggerWeeklySummary(
    userId: string,
    channels?: string[]
): Promise<{ status: string; delivery_channels: string[] }> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/weekly-summary/trigger`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify({
                user_id: userId,
                channels,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'triggerWeeklySummary');
    }
}

/**
 * Unsubscribe from weekly summaries
 *
 * @param userId - User identifier
 * @returns Unsubscribe result
 */
export async function unsubscribeWeeklySummary(
    userId: string
): Promise<{ status: string }> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/weekly-summary/unsubscribe`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify({ user_id: userId }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'unsubscribeWeeklySummary');
    }
}

// ============================================================================
// Notification Templates
// ============================================================================

/**
 * Get available notification templates
 *
 * @returns Array of template definitions
 */
export async function getNotificationTemplates(): Promise<
    Array<{ name: string; description: string; channels: string[] }>
> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/notifications/templates`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        const result = await response.json();
        return result.templates || [];
    } catch (error) {
        handleError(error, 'getNotificationTemplates');
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if push notifications are supported and enabled
 */
export function isPushSupported(): boolean {
    return 'Notification' in window && 'serviceWorker' in navigator;
}

/**
 * Check current notification permission status
 */
export function getNotificationPermission(): NotificationPermission | 'unsupported' {
    if (!('Notification' in window)) {
        return 'unsupported';
    }
    return Notification.permission;
}

// ============================================================================
// Exported Service Object
// ============================================================================

export const notificationService = {
    // Push Notifications
    registerDevice,
    requestPushPermission,
    isPushSupported,
    getNotificationPermission,

    // Weekly Summary
    getWeeklySummaryPreferences,
    updateWeeklySummaryPreferences,
    triggerWeeklySummary,
    unsubscribeWeeklySummary,

    // Templates
    getNotificationTemplates,

    // Supported channels
    SUPPORTED_CHANNELS: ['push', 'email', 'whatsapp'] as const,

    // Days of week for preferences
    DAYS_OF_WEEK: [
        { value: 0, label: 'Sunday' },
        { value: 1, label: 'Monday' },
        { value: 2, label: 'Tuesday' },
        { value: 3, label: 'Wednesday' },
        { value: 4, label: 'Thursday' },
        { value: 5, label: 'Friday' },
        { value: 6, label: 'Saturday' },
    ],
};

export default notificationService;
