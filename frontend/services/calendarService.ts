/**
 * Calendar Service
 *
 * Handles calendar integration with Google and Outlook.
 * OAuth credential management, calendar sync, and reminder scheduling.
 *
 * Healthcare-grade error handling:
 * - No PHI or stack traces exposed to users
 * - All errors logged for developers
 * - User-friendly error messages
 */

import {
    CalendarCredentialsRequest,
    CalendarSyncRequest,
    CalendarEvent,
    ReminderRequest,
    CalendarProvider,
} from './api.types';
import { authService } from './authService';

// Use environment variable or fallback to localhost
const API_BASE_URL = (import.meta as any).env?.VITE_NLP_SERVICE_URL || 'http://localhost:5001';

// Configuration
const REQUEST_TIMEOUT_MS = 30000;

// ============================================================================
// Error Handling
// ============================================================================

export class CalendarServiceError extends Error {
    constructor(
        public code: string,
        public userMessage: string,
        public developerMessage: string,
        public statusCode?: number
    ) {
        super(userMessage);
        this.name = 'CalendarServiceError';
    }
}

/**
 * Convert API errors to user-friendly messages
 */
function handleError(error: unknown, context: string): never {
    console.error(`[CalendarService] ${context}:`, error);

    if (error instanceof CalendarServiceError) {
        throw error;
    }

    if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new CalendarServiceError(
            'NETWORK_ERROR',
            'Unable to connect. Please check your internet connection.',
            `Network error during ${context}: ${error.message}`
        );
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
        throw new CalendarServiceError(
            'TIMEOUT',
            'Request timed out. Please try again.',
            `Timeout during ${context}`
        );
    }

    if (error && typeof error === 'object' && 'status' in error) {
        const status = (error as { status: number }).status;
        const detail = (error as { detail?: string }).detail;

        switch (status) {
            case 401:
                throw new CalendarServiceError(
                    'UNAUTHORIZED',
                    'Calendar authorization expired. Please reconnect.',
                    detail || 'Unauthorized',
                    401
                );
            case 403:
                throw new CalendarServiceError(
                    'FORBIDDEN',
                    'Access denied. Please check your calendar permissions.',
                    detail || 'Forbidden',
                    403
                );
            case 503:
                throw new CalendarServiceError(
                    'SERVICE_UNAVAILABLE',
                    'Calendar service is temporarily unavailable. Please try again later.',
                    detail || 'Service unavailable',
                    503
                );
            default:
                throw new CalendarServiceError(
                    'SERVER_ERROR',
                    'Something went wrong. Please try again.',
                    detail || `HTTP ${status}`,
                    status
                );
        }
    }

    throw new CalendarServiceError(
        'UNKNOWN_ERROR',
        'Something went wrong. Please try again.',
        error instanceof Error ? error.message : String(error)
    );
}

// ============================================================================
// API Methods
// ============================================================================

/**
 * Store OAuth credentials for a calendar provider
 *
 * @param userId - User identifier
 * @param credentials - OAuth credentials from the provider
 * @returns Success response
 */
export async function storeCalendarCredentials(
    userId: string,
    credentials: CalendarCredentialsRequest
): Promise<{ status: string }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/api/calendar/${userId}/credentials`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify(credentials),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        handleError(error, 'storeCalendarCredentials');
    }
}

/**
 * Revoke calendar credentials for a provider
 *
 * @param userId - User identifier
 * @param provider - Calendar provider to disconnect
 * @returns Success response
 */
export async function revokeCalendarCredentials(
    userId: string,
    provider: CalendarProvider
): Promise<{ status: string }> {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/calendar/${userId}/credentials/${provider}`,
            {
                method: 'DELETE',
                headers: {
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
        handleError(error, 'revokeCalendarCredentials');
    }
}

/**
 * Trigger calendar sync with provider
 *
 * @param userId - User identifier
 * @param request - Sync configuration
 * @returns Sync result with event count
 */
export async function syncCalendar(
    userId: string,
    request: CalendarSyncRequest
): Promise<{ status: string; events_synced: number }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS * 2); // Longer timeout for sync

    try {
        const response = await fetch(`${API_BASE_URL}/api/calendar/${userId}/sync`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify(request),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        handleError(error, 'syncCalendar');
    }
}

/**
 * Get calendar events from a provider
 *
 * @param userId - User identifier
 * @param provider - Calendar provider
 * @param startDate - Optional start date filter
 * @param endDate - Optional end date filter
 * @returns Array of calendar events
 */
export async function getCalendarEvents(
    userId: string,
    provider: CalendarProvider,
    startDate?: Date,
    endDate?: Date
): Promise<CalendarEvent[]> {
    try {
        const params = new URLSearchParams({ provider });
        if (startDate) params.append('start_date', startDate.toISOString());
        if (endDate) params.append('end_date', endDate.toISOString());

        const response = await fetch(
            `${API_BASE_URL}/api/calendar/${userId}/events?${params.toString()}`,
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

        const result = await response.json();
        return result.events || [];
    } catch (error) {
        handleError(error, 'getCalendarEvents');
    }
}

/**
 * Schedule a reminder for an event
 *
 * @param userId - User identifier
 * @param reminder - Reminder configuration
 * @returns Created reminder ID
 */
export async function scheduleReminder(
    userId: string,
    reminder: ReminderRequest
): Promise<{ reminder_id: string; status: string }> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/calendar/${userId}/reminder`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(authService.getAuthHeader() && { Authorization: authService.getAuthHeader()! }),
            },
            body: JSON.stringify(reminder),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw { status: response.status, detail: errorData.detail };
        }

        return await response.json();
    } catch (error) {
        handleError(error, 'scheduleReminder');
    }
}

/**
 * Get all reminders for a user
 *
 * @param userId - User identifier
 * @returns Array of active reminders
 */
export async function getReminders(userId: string): Promise<ReminderRequest[]> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/calendar/${userId}/reminders`, {
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
        return result.reminders || [];
    } catch (error) {
        handleError(error, 'getReminders');
    }
}

/**
 * Cancel a reminder
 *
 * @param userId - User identifier
 * @param reminderId - Reminder to cancel
 * @returns Success response
 */
export async function cancelReminder(
    userId: string,
    reminderId: string
): Promise<{ status: string }> {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/calendar/${userId}/reminder/${reminderId}`,
            {
                method: 'DELETE',
                headers: {
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
        handleError(error, 'cancelReminder');
    }
}

// ============================================================================
// Exported Service Object
// ============================================================================

export const calendarService = {
    storeCalendarCredentials,
    revokeCalendarCredentials,
    syncCalendar,
    getCalendarEvents,
    scheduleReminder,
    getReminders,
    cancelReminder,

    // Supported providers
    SUPPORTED_PROVIDERS: ['google', 'outlook'] as CalendarProvider[],
};

export default calendarService;
