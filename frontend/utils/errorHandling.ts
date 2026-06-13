/**
 * Error Handling Utilities
 * Centralized error classification, logging, and handling
 */

// Error types for classification
export enum ErrorType {
    NETWORK = 'NETWORK',
    AUTHENTICATION = 'AUTHENTICATION',
    AUTHORIZATION = 'AUTHORIZATION',
    VALIDATION = 'VALIDATION',
    NOT_FOUND = 'NOT_FOUND',
    SERVER = 'SERVER',
    TIMEOUT = 'TIMEOUT',
    UNKNOWN = 'UNKNOWN',
    OFFLINE = 'OFFLINE',
}

export interface AppError {
    type: ErrorType;
    message: string;
    originalError?: any;
    statusCode?: number;
    retryable: boolean;
    userMessage: string;
}

/**
 * Classify error based on error object
 */
export function classifyError(error: any): AppError {
    // Check for offline status
    if (typeof navigator !== 'undefined' && !navigator.onLine) {
        return {
            type: ErrorType.OFFLINE,
            message: 'Device is offline',
            originalError: error,
            retryable: true,
            userMessage: 'You are currently offline. Please check your internet connection and try again.',
        };
    }

    // Network errors
    if (error.message?.includes('Network') || error.message?.includes('fetch') || error.name === 'NetworkError') {
        return {
            type: ErrorType.NETWORK,
            message: error.message || 'Network error occurred',
            originalError: error,
            retryable: true,
            userMessage: 'Unable to connect to the server. Please check your internet connection and try again.',
        };
    }

    // HTTP status code errors
    const status = error.status || error.statusCode || error.response?.status;

    if (status === 401) {
        return {
            type: ErrorType.AUTHENTICATION,
            message: 'Authentication failed',
            originalError: error,
            statusCode: 401,
            retryable: false,
            userMessage: 'Your session has expired. Please log in again.',
        };
    }

    if (status === 403) {
        return {
            type: ErrorType.AUTHORIZATION,
            message: 'Authorization failed',
            originalError: error,
            statusCode: 403,
            retryable: false,
            userMessage: 'You do not have permission to access this resource.',
        };
    }

    if (status === 404) {
        return {
            type: ErrorType.NOT_FOUND,
            message: 'Resource not found',
            originalError: error,
            statusCode: 404,
            retryable: false,
            userMessage: 'The requested resource was not found.',
        };
    }

    if (status === 400 || status === 422) {
        return {
            type: ErrorType.VALIDATION,
            message: error.message || 'Validation error',
            originalError: error,
            statusCode: status,
            retryable: false,
            userMessage: error.message || 'Please check your input and try again.',
        };
    }

    if (status >= 500) {
        return {
            type: ErrorType.SERVER,
            message: 'Server error',
            originalError: error,
            statusCode: status,
            retryable: true,
            userMessage: 'A server error occurred. Please try again later.',
        };
    }

    // Timeout errors
    if (error.message?.includes('timeout') || error.name === 'TimeoutError') {
        return {
            type: ErrorType.TIMEOUT,
            message: 'Request timeout',
            originalError: error,
            retryable: true,
            userMessage: 'The request took too long. Please try again.',
        };
    }

    // Unknown errors
    return {
        type: ErrorType.UNKNOWN,
        message: error.message || 'An unknown error occurred',
        originalError: error,
        retryable: true,
        userMessage: 'An unexpected error occurred. Please try again.',
    };
}

/**
 * Retry with exponential backoff
 */
export async function retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    initialDelay: number = 1000
): Promise<T> {
    let lastError: any;

    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            const appError = classifyError(error);

            // Don't retry non-retryable errors
            if (!appError.retryable) {
                throw error;
            }

            // Don't retry on last attempt
            if (i === maxRetries - 1) {
                throw error;
            }

            // Wait with exponential backoff
            const delay = initialDelay * Math.pow(2, i);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }

    throw lastError;
}

/**
 * Log error to console and optionally to error tracking service
 */
export function logError(error: AppError, context?: any) {
    console.error('[Error]', {
        type: error.type,
        message: error.message,
        statusCode: error.statusCode,
        context,
        originalError: error.originalError,
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'N/A',
        timestamp: new Date().toISOString(),
        url: typeof window !== 'undefined' ? window.location?.href : 'N/A'
    });

    // In production, send to error tracking service (e.g., Sentry)
    if (process.env.NODE_ENV === 'production') {
        // Example: Sentry.captureException(error.originalError, { contexts: { app: context } });
    }
}

/**
 * Handle error with user feedback
 */
export function handleError(error: any, context?: any): AppError {
    const appError = classifyError(error);
    logError(appError, context);
    return appError;
}

/**
 * Get icon for error type
 */
export function getErrorIcon(errorType: ErrorType): string {
    switch (errorType) {
        case ErrorType.NETWORK:
        case ErrorType.OFFLINE:
            return 'cloud-offline';
        case ErrorType.AUTHENTICATION:
        case ErrorType.AUTHORIZATION:
            return 'lock-closed';
        case ErrorType.VALIDATION:
            return 'alert-circle';
        case ErrorType.NOT_FOUND:
            return 'search';
        case ErrorType.SERVER:
            return 'server';
        case ErrorType.TIMEOUT:
            return 'time';
        default:
            return 'warning';
    }
}

/**
 * Get color for error type
 */
export function getErrorColor(errorType: ErrorType): string {
    switch (errorType) {
        case ErrorType.NETWORK:
        case ErrorType.OFFLINE:
            return '#FF9800';
        case ErrorType.AUTHENTICATION:
        case ErrorType.AUTHORIZATION:
            return '#F9A825';
        case ErrorType.VALIDATION:
            return '#EA4335';
        case ErrorType.NOT_FOUND:
            return '#666';
        case ErrorType.SERVER:
        case ErrorType.TIMEOUT:
            return '#EA4335';
        default:
            return '#EA4335';
    }
}
