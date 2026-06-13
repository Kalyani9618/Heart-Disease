/**
 * API Client Service
 * All frontend API calls go through this service to the backend
 * NO API KEYS are stored in the frontend
 */

// Use empty string for API_BASE_URL when using Vite proxy (VITE_API_URL is empty or '/')
// When VITE_API_URL is empty, endpoints will be relative like /api/...
// When using proxy, /api paths are automatically forwarded to backend
// Updated to point to the new NLP service instead of Flask backend
const API_BASE_URL = (import.meta as any).env.VITE_NLP_SERVICE_URL && (import.meta as any).env.VITE_NLP_SERVICE_URL !== '/'
  ? (import.meta as any).env.VITE_NLP_SERVICE_URL
  : 'http://localhost:5001';

import { handleError, retryWithBackoff, ErrorType } from '../utils/errorHandling';
import { authService } from './authService';
import { HeartDiseasePredictionRequest, HeartDiseasePredictionResponse, TestResultDetail, DocumentDetails, AudioTranscriptionResponse, TextToSpeechResponse } from './api.types';

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: any;
  timeout?: number;
  skipAuth?: boolean; // Skip authentication for login/register endpoints
  skipDedup?: boolean; // Skip request deduplication
  retries?: number; // Number of retries for failed requests
}

class APIError extends Error {
  constructor(
    public status: number,
    public message: string,
    public data?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// ============================================================================
// Request Deduplication
// ============================================================================

// In-flight requests map for deduplication
const inFlightRequests = new Map<string, Promise<any>>();

/**
 * Generate a cache key for request deduplication
 */
function getRequestKey(endpoint: string, options: RequestOptions): string {
  return `${options.method || 'GET'}:${endpoint}:${JSON.stringify(options.body || {})}`;
}

// ============================================================================
// Request/Response Interceptors
// ============================================================================

type RequestInterceptor = (endpoint: string, options: RequestOptions) => RequestOptions;
type ResponseInterceptor = <T>(response: T, endpoint: string) => T;

const requestInterceptors: RequestInterceptor[] = [];
const responseInterceptors: ResponseInterceptor[] = [];

/**
 * Add a request interceptor
 */
export function addRequestInterceptor(interceptor: RequestInterceptor): void {
  requestInterceptors.push(interceptor);
}

/**
 * Add a response interceptor
 */
export function addResponseInterceptor(interceptor: ResponseInterceptor): void {
  responseInterceptors.push(interceptor);
}

/**
 * Clear all interceptors
 */
export function clearInterceptors(): void {
  requestInterceptors.length = 0;
  responseInterceptors.length = 0;
}

// ============================================================================
// Centralized Error Handler
// ============================================================================

/**
 * Process API errors with user-friendly messages
 */
function processApiError(error: unknown): APIError {
  if (error instanceof APIError) {
    return error;
  }

  if (error instanceof TypeError && error.message === 'Failed to fetch') {
    return new APIError(0, 'Network error. Please check your connection.', { type: ErrorType.NETWORK });
  }

  if (error instanceof DOMException && error.name === 'AbortError') {
    return new APIError(0, 'Request timeout. Please try again.', { type: ErrorType.TIMEOUT });
  }

  return new APIError(
    0,
    error instanceof Error ? error.message : 'Unknown error occurred',
    error
  );
}

async function apiCall<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  // Apply request interceptors
  let processedOptions = { ...options };
  for (const interceptor of requestInterceptors) {
    processedOptions = interceptor(endpoint, processedOptions);
  }

  const {
    method = 'GET',
    headers = {},
    body,
    timeout = 30000,
    skipAuth = false,
    skipDedup = false,
    retries = 0,
  } = processedOptions;

  // Request deduplication for GET requests
  const requestKey = getRequestKey(endpoint, processedOptions);
  if (method === 'GET' && !skipDedup) {
    const existingRequest = inFlightRequests.get(requestKey);
    if (existingRequest) {
      console.log(`[API] Deduplicating request: ${requestKey}`);
      return existingRequest;
    }
  }

  // Check for offline status before making request
  if (typeof navigator !== 'undefined' && !navigator.onLine) {
    throw new APIError(
      0,
      'Device is offline',
      { type: ErrorType.OFFLINE }
    );
  }

  // Inject auth header if available and not skipped
  const authHeaders: Record<string, string> = {};
  if (!skipAuth) {
    const authHeader = authService.getAuthHeader();
    if (authHeader) {
      authHeaders['Authorization'] = authHeader;
    }
  }

  const url = `${API_BASE_URL}${endpoint}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;

    const response = await fetch(url, {
      method,
      headers: {
        ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
        ...authHeaders,
        ...headers,
      },
      body: body ? (isFormData ? body : JSON.stringify(body)) : undefined,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle 401 Unauthorized - token expired or invalid
    if (response.status === 401 && !skipAuth) {
      // Try to refresh token
      const refreshed = await handleTokenRefresh();
      if (refreshed) {
        // Retry request with new token
        return apiCall<T>(endpoint, options);
      } else {
        // Refresh failed, clear auth
        authService.clearAuth();
        // Redirect to login if in browser (window defined)
        if (typeof window !== 'undefined') {
          window.location.hash = '#/login';
        }
        throw new APIError(401, 'Session expired. Please log in again.');
      }
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        response.status,
        errorData.error || `HTTP ${response.status}`,
        errorData
      );
    }

    let result = (await response.json()) as T;

    // Apply response interceptors
    for (const interceptor of responseInterceptors) {
      result = interceptor(result, endpoint);
    }

    // Clear from in-flight on success
    inFlightRequests.delete(requestKey);

    return result;
  } catch (error) {
    clearTimeout(timeoutId);

    // Clear from in-flight on error
    inFlightRequests.delete(requestKey);

    // Retry logic for transient errors
    if (retries > 0 && error instanceof APIError) {
      // Retry on network errors or 5xx errors
      if (error.status === 0 || error.status >= 500) {
        console.log(`[API] Retrying request (${retries} attempts left): ${endpoint}`);
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
        return apiCall<T>(endpoint, { ...processedOptions, retries: retries - 1 });
      }
    }

    throw processApiError(error);
  }
}

// ============================================================================
// Token Refresh Helper
// ============================================================================

/**
 * Attempt to refresh the access token using the refresh token
 * Returns true if successful, false otherwise
 */
async function handleTokenRefresh(): Promise<boolean> {
  try {
    const refreshToken = authService.getRefreshToken();
    if (!refreshToken) {
      console.log('[Auth] No refresh token available');
      return false;
    }

    console.log('[Auth] Attempting token refresh...');
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      console.log('[Auth] Token refresh failed:', response.status);
      return false;
    }

    const data = await response.json();

    // Store new tokens
    authService.setToken(data.token);
    if (data.refresh_token) {
      authService.setRefreshToken(data.refresh_token);
    }

    console.log('[Auth] Token refreshed successfully');
    return true;
  } catch (error) {
    console.error('[Auth] Token refresh error:', error);
    return false;
  }
}

// ============================================================================
// Retry Logic
// ============================================================================

const RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504];
const MAX_RETRIES = 3;
const INITIAL_DELAY_MS = 1000;

interface RetryOptions extends RequestOptions {
  maxRetries?: number;
  retryDelay?: number;
}

/**
 * API call with automatic retry for transient failures
 * Uses exponential backoff between retries
 */
async function apiCallWithRetry<T>(
  endpoint: string,
  options: RetryOptions = {}
): Promise<T> {
  const { maxRetries = MAX_RETRIES, retryDelay = INITIAL_DELAY_MS, ...requestOptions } = options;
  let lastError: APIError | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await apiCall<T>(endpoint, requestOptions);
    } catch (error) {
      if (!(error instanceof APIError)) {
        throw error;
      }

      lastError = error;

      // Don't retry client errors (4xx) except for specific ones
      const isRetryableStatus = RETRY_STATUS_CODES.includes(error.status);
      const isNetworkError = error.status === 0;

      if (!isRetryableStatus && !isNetworkError) {
        throw error;
      }

      // Don't retry on last attempt
      if (attempt === maxRetries) {
        break;
      }

      // Exponential backoff with jitter
      const delay = retryDelay * Math.pow(2, attempt) + Math.random() * 100;
      console.log(`[API] Retry ${attempt + 1}/${maxRetries} after ${Math.round(delay)}ms for ${endpoint}`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

// ============================================================================
// API ENDPOINTS
// ============================================================================

export const apiClient = {
  // ==========================================================================
  // HEART DISEASE API
  // ==========================================================================

  /**
   * Predict Heart Disease Risk
   */
  predictHeartDisease: async (data: HeartDiseasePredictionRequest) => {
    return apiCall<HeartDiseasePredictionResponse>('/heart/predict', {
      method: 'POST',
      body: data,
      timeout: 150000, // 120 seconds – model inference can be slow
    });
  },

  // ==========================================================================
  // AUTHENTICATION API
  // ==========================================================================

  /**
   * Login with email and password
   */
  login: async (email: string, password: string) => {
    try {
      return await apiCall<{
        user: {
          id: string;
          email: string;
          name: string;
          role: string;
        };
        token: string;
        refresh_token?: string;
      }>('/auth/login', {
        method: 'POST',
        body: { email, password },
        skipAuth: true,
      });
    } catch (error) {
      console.error('[Auth] Login failed:', error);
      throw error;
    }
  },

  /**
   * Register new user
   */
  register: async (data: {
    email: string;
    password: string;
    name: string;
  }) => {
    try {
      return apiCall<{
        user: {
          id: string;
          email: string;
          name: string;
        };
        token: string;
        refresh_token?: string;
      }>('/auth/register', {
        method: 'POST',
        body: data,
        skipAuth: true,
      });
    } catch (error) {
      console.error('[Auth] Registration failed:', error);
      throw error;
    }
  },

  /**
   * Logout current user
   */
  logout: async () => {
    return apiCall<{ message: string }>('/auth/logout', {
      method: 'POST',
    });
  },

  /**
   * Change user password
   */
  changePassword: async (data: { current_password: string; new_password: string }) => {
    return apiCall<{ message: string }>('/auth/change-password', {
      method: 'POST',
      body: data,
    });
  },

  /**
   * Get current authenticated user
   */
  me: async () => {
    try {
      return await apiCall<{
        id: string;
        email: string;
        name: string;
        role: string;
      }>('/auth/me');
    } catch (error) {
      console.error('[Auth] Failed to fetch user profile:', error);
      throw error;
    }
  },

  /**
   * Refresh authentication token
   */
  refreshToken: async (refreshToken: string) => {
    return apiCall<{
      token: string;
      refresh_token?: string;
    }>('/auth/refresh', {
      method: 'POST',
      body: { refresh_token: refreshToken },
      skipAuth: true,
    });
  },

  /**
   * Generate daily health insight
   */
  generateInsight: async (params: {
    user_name: string;
    vitals: {
      heart_rate?: number;
      blood_pressure?: string;
      blood_glucose?: number;
    };
    activities?: string[];
    medications?: string[];
  }) => {
    // Backend expects a "query" field - build one from the params
    const vitalsDesc = Object.entries(params.vitals || {})
      .filter(([, v]) => v && v !== 'N/A')
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');
    const query = `Provide a daily cardiovascular health insight for ${params.user_name}. ${vitalsDesc ? `Current vitals: ${vitalsDesc}.` : ''} Give brief, actionable advice.`;

    return apiCall<{
      insight: string;
      timestamp: string;
      disclaimer?: string;
      context_used?: string[];
      provider?: string;
    }>('/heart/insight', {
      method: 'POST',
      body: { query, ...params },
    });
  },

  /**
   * Perform comprehensive health assessment
   */
  healthAssessment: async (params: {
    user_name: string;
    age?: number;
    vitals: Record<string, any>;
    health_history?: string[];
    lifestyle?: Record<string, any>;
  }) => {
    return apiCall<{
      assessment: string;
      user: string;
      timestamp: string;
    }>('/heart/insight', {
      method: 'POST',
      body: params,
    });
  },

  /**
   * Get medication-related insights
   */
  medicationInsights: async (params: {
    medications: Array<{ name: string; dosage: string }>;
    supplements?: string[];
    recent_vitals?: Record<string, any>;
  }) => {
    return apiCall<{
      insights: string;
      medication_count: number;
      timestamp: string;
    }>('/heart/insight', {
      method: 'POST',
      body: params,
    });
  },

  /**
   * Health check
   */
  healthCheck: async () => {
    return apiCall<{
      status: string;
      service: string;
    }>('/health');
  },

  /**
   * Process text with NLP service
   */
  processNLP: async (params: {
    message: string;
    session_id?: string;
    user_id?: string;
    context?: any;
    model?: 'gemini' | 'ollama';
  }) => {
    return apiCall<{
      intent: string;
      sentiment: string;
      entities: any[];
      suggested_response?: string;
      requires_escalation?: boolean;
    }>('/nlp/process', {
      method: 'POST',
      body: params,
    });
  },

  /**
   * Transcribe audio using the backend
   */
  transcribeAudio: async (base64Audio: string) => {
    return apiCall<AudioTranscriptionResponse>('/speech/transcribe', {
      method: 'POST',
      body: { audio: base64Audio },
    });
  },

  /**
   * Synthesize speech using the backend
   */
  synthesizeSpeech: async (text: string) => {
    return apiCall<TextToSpeechResponse>('/speech/synthesize', {
      method: 'POST',
      body: { text },
    });
  },

  /**
   * Stream response from Ollama (Server-Sent Events)
   * Returns an async generator that yields tokens as they arrive
   */
  streamOllamaResponse: async function* (params: {
    message: string;
    model?: string;
    conversation_history?: Array<{ role: string; content: string }>;
    temperature?: number;
    signal?: AbortSignal;
    web_search?: boolean;
    deep_search?: boolean;
    thinking?: boolean;
  }): AsyncGenerator<{ type: 'token' | 'done' | 'error'; data: string | any }> {
    // Check for offline status before making request
    if (typeof navigator !== 'undefined' && !navigator.onLine) {
      yield { type: 'error', data: { error: 'You are currently offline. Please check your internet connection.' } };
      return;
    }

    const url = `${API_BASE_URL}/chat/message`;
    const { signal, ...rest } = params;

    // Pre-check: if token is expired, try to refresh before making the request
    if (authService.isTokenExpired()) {
      console.log('[StreamChat] Token expired, attempting refresh before request...');
      const refreshed = await handleTokenRefresh();
      if (!refreshed) {
        // No valid token - try to continue anyway (backend may allow anonymous)
        console.warn('[StreamChat] Token refresh failed, proceeding without auth');
      }
    }

    // Build auth headers (re-read after potential refresh)
    const authHeaders: Record<string, string> = {};
    const authHeader = authService.getAuthHeader();
    if (authHeader) {
      authHeaders['Authorization'] = authHeader;
    }

    // Map to the backend ChatRequest schema
    const userId = authService.getUserId?.() || localStorage.getItem('user_id') || 'anonymous';
    const isResearchMode = rest.web_search || rest.deep_search || rest.thinking;
    const body = {
      user_id: userId,
      message: rest.message,
      sync: true, // Use sync mode for direct response
      thinking: rest.thinking || false,
      web_search: rest.web_search || false,
      deep_search: rest.deep_search || false,
    };

    // Helper to make the fetch call (used for retry on 401)
    // For research modes (web/deep search), use a longer timeout via AbortController
    const fetchTimeout = isResearchMode ? 300000 : 180000; // 5 min for research, 3 min for normal
    const doFetch = async (headers: Record<string, string>) => {
      // Create a timeout-based abort that composes with the user's signal
      const timeoutController = new AbortController();
      const timeoutId = setTimeout(() => timeoutController.abort(), fetchTimeout);

      // If the caller's signal aborts, also abort our controller
      if (signal) {
        signal.addEventListener('abort', () => timeoutController.abort(), { once: true });
      }

      try {
        const resp = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...headers,
          },
          body: JSON.stringify(body),
          signal: timeoutController.signal,
        });
        return resp;
      } finally {
        clearTimeout(timeoutId);
      }
    };

    try {
      let response = await doFetch(authHeaders);

      // Handle 401 Unauthorized - try token refresh and retry once
      if (response.status === 401) {
        console.log('[StreamChat] Got 401, attempting token refresh...');
        const refreshed = await handleTokenRefresh();
        if (refreshed) {
          // Rebuild auth headers with new token
          const newAuthHeaders: Record<string, string> = {};
          const newAuthHeader = authService.getAuthHeader();
          if (newAuthHeader) {
            newAuthHeaders['Authorization'] = newAuthHeader;
          }
          response = await doFetch(newAuthHeaders);
        } else {
          // Refresh failed - clear auth and redirect to login
          authService.clearAuth();
          if (typeof window !== 'undefined') {
            window.location.hash = '#/login';
          }
          yield { type: 'error', data: { error: 'Session expired. Please log in again.' } };
          return;
        }
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      // Simulate streaming by yielding the full response
      // The backend /chat/message sync mode returns ChatResponse
      if (data.response) {
        // Yield response in chunks for smooth streaming effect
        const words = data.response.split(' ');
        let accumulated = '';
        for (let i = 0; i < words.length; i++) {
          accumulated += (i > 0 ? ' ' : '') + words[i];
          yield { type: 'token', data: words[i] + (i < words.length - 1 ? ' ' : '') };
          // Small delay for streaming effect
          await new Promise(r => setTimeout(r, 15));
        }
        yield {
          type: 'done',
          data: {
            sources: data.sources || [],
            metadata: data.metadata || {},
            session_id: data.session_id,
          }
        };
      } else if (data.error || !data.success) {
        yield { type: 'error', data: { error: data.error || 'No response received' } };
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      yield { type: 'error', data: { error: error instanceof Error ? error.message : 'Unknown error' } };
    }
  },

  // ============================================================================
  // STRUCTURED OUTPUT ENDPOINTS
  // These endpoints return LLM responses that match predefined JSON schemas
  // ============================================================================

  /**
   * Check if structured outputs feature is available
   */
  getStructuredOutputsStatus: async () => {
    return apiCall<StructuredOutputsStatus>('/structured-outputs/status');
  },

  /**
   * Get the JSON schema for a specific output type
   */
  getStructuredSchema: async (schemaName: StructuredSchemaName) => {
    return apiCall<{
      schema_name: string;
      json_schema: Record<string, any>;
      description: string;
    }>(`/structured-outputs/schema/${schemaName}`);
  },

  /**
   * Generate a structured health analysis from user message
   * Returns comprehensive health analysis with intent, sentiment, entities, recommendations
   */
  structuredHealthAnalysis: async (params: StructuredHealthAnalysisRequest) => {
    return apiCall<StructuredResponse<CardioHealthAnalysis>>(
      '/structured-outputs/health-analysis',
      {
        method: 'POST',
        body: params,
      }
    );
  },

  /**
   * Generate a quick intent analysis
   * Lightweight classification of user intent
   */
  structuredIntentAnalysis: async (params: { message: string }) => {
    return apiCall<StructuredResponse<SimpleIntentAnalysis>>(
      '/structured-outputs/intent',
      {
        method: 'POST',
        body: params,
      }
    );
  },

  /**
   * Generate a structured conversation response
   */
  structuredConversation: async (params: StructuredConversationRequest) => {
    return apiCall<StructuredResponse<ConversationResponse>>(
      '/structured-outputs/conversation',
      {
        method: 'POST',
        body: params,
      }
    );
  },

  // ==========================================================================
  // USER PREFERENCES API
  // ==========================================================================

  /**
   * Get all user preferences
   */
  getPreferences: async (userId: string): Promise<UserPreferences> => {
    return apiCall<UserPreferences>(`/memory/preferences/${userId}`);
  },

  /**
   * Get a specific user preference
   */
  getPreference: async (userId: string, key: string): Promise<{ key: string; value: unknown }> => {
    return apiCall<{ key: string; value: unknown }>(`/memory/preferences/${userId}/${key}`);
  },

  /**
   * Update user preferences
   */
  updatePreferences: async (userId: string, preferences: Partial<UserPreferences>): Promise<void> => {
    return apiCall<void>(`/memory/preferences/${userId}`, {
      method: 'PUT',
      body: preferences,
    });
  },

  /**
   * Bulk update user preferences
   */
  bulkUpdatePreferences: async (userId: string, preferences: Record<string, unknown>): Promise<void> => {
    return apiCall<void>(`/memory/preferences/${userId}/bulk`, {
      method: 'PUT',
      body: { preferences },
    });
  },

  /**
   * Delete a specific user preference
   */
  deletePreference: async (userId: string, key: string): Promise<void> => {
    return apiCall<void>(`/memory/preferences/${userId}/${key}`, {
      method: 'DELETE',
    });
  },

  // ==========================================================================
  // GDPR COMPLIANCE API
  // ==========================================================================

  /**
   * Export all user data for GDPR compliance
   * Returns all stored data associated with the user
   */
  exportUserData: async (userId: string): Promise<GDPRExportData> => {
    return apiCall<GDPRExportData>(`/memory/gdpr/export/${userId}`, {
      method: 'POST',
      timeout: 60000, // Longer timeout for data export
    });
  },

  /**
   * Delete all user data for GDPR compliance (Right to be Forgotten)
   * This is irreversible - use with caution
   */
  deleteUserData: async (userId: string): Promise<GDPRDeleteResponse> => {
    return apiCall<GDPRDeleteResponse>(`/memory/gdpr/delete/${userId}`, {
      method: 'DELETE',
      timeout: 60000, // Longer timeout for data deletion
    });
  },

  /**
   * Get audit log for user data access
   */
  getAuditLog: async (userId: string, limit?: number): Promise<AuditLogEntry[]> => {
    const params = limit ? `?limit=${limit}` : '';
    return apiCall<AuditLogEntry[]>(`/memory/audit/${userId}${params}`);
  },

  // ==========================================================================
  // MODEL MANAGEMENT API
  // ==========================================================================

  /**
   * Get all model versions
   */
  getModelVersions: async (): Promise<ModelVersionsResponse> => {
    return apiCall<ModelVersionsResponse>('/models/versions');
  },

  /**
   * Get version history for a specific model
   */
  getModelHistory: async (modelName: string): Promise<ModelHistoryResponse> => {
    return apiCall<ModelHistoryResponse>(`/models/history/${modelName}`);
  },

  /**
   * List all available models
   */
  listModels: async (): Promise<ModelsListResponse> => {
    return apiCall<ModelsListResponse>('/models/list');
  },

  // ==========================================================================
  // DOCUMENT API
  // ==========================================================================

  uploadDocument: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiCall<DocumentUploadResponse>('/documents/upload', {
      method: 'POST',
      body: formData,
    });
  },

  processDocument: async (documentId: string) => {
    return apiCall<DocumentProcessingResult>(`/documents/process/${documentId}`, {
      method: 'POST',
    });
  },

  getDocument: async (documentId: string) => {
    return apiCall<DocumentResponse>(`/documents/${documentId}`);
  },

  getDocuments: async () => {
    return apiCall<DocumentDetails[]>('/documents');
  },

  // ==========================================================================
  // VISION API
  // ==========================================================================

  analyzeECG: async (file: File, context?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (context) formData.append('patient_context', context);
    return apiCall<ECGAnalysisResponse>('/vision/ecg/analyze', {
      method: 'POST',
      body: formData,
    });
  },

  // ==========================================================================
  // CALENDAR API
  // ==========================================================================

  storeCalendarCredentials: async (userId: string, credentials: any) => {
    return apiCall<any>(`/calendar/${userId}/credentials`, {
      method: 'POST',
      body: credentials,
    });
  },

  syncCalendar: async (userId: string, options: any) => {
    return apiCall<SyncResponse>(`/calendar/${userId}/sync`, {
      method: 'POST',
      body: options,
    });
  },

  getCalendarEvents: async (userId: string, start?: string, end?: string) => {
    const params = new URLSearchParams();
    if (start) params.append('start_date', start);
    if (end) params.append('end_date', end);
    return apiCall<CalendarEventResponse[]>(`/calendar/${userId}/events?${params.toString()}`);
  },

  scheduleReminder: async (userId: string, reminder: any) => {
    return apiCall<ReminderResponse>(`/calendar/${userId}/reminder`, {
      method: 'POST',
      body: reminder,
    });
  },

  // ==========================================================================
  // NOTIFICATIONS API
  // ==========================================================================

  sendWhatsApp: async (request: any) => {
    return apiCall<any>('/notifications/whatsapp', {
      method: 'POST',
      body: request,
    });
  },

  sendEmail: async (request: any) => {
    return apiCall<any>('/notifications/email', {
      method: 'POST',
      body: request,
    });
  },

  registerDevice: async (userId: string, token: string, platform: string) => {
    return apiCall<any>('/notifications/register-device', {
      method: 'POST',
      body: { user_id: userId, device_token: token, platform },
    });
  },

  sendPushNotification: async (request: {
    user_id: string;
    title: string;
    body: string;
    data?: any;
  }) => {
    return apiCall<any>('/notifications/push', {
      method: 'POST',
      body: request,
    });
  },

  // ==========================================================================
  // SMARTWATCH API
  // ==========================================================================

  registerSmartwatch: async (device: any) => {
    return apiCall<any>('/smartwatch/register', {
      method: 'POST',
      body: device,
    });
  },

  ingestVitals: async (payload: any) => {
    return apiCall<any>('/smartwatch/vitals/ingest', {
      method: 'POST',
      body: payload,
    });
  },

  getAggregatedVitals: async (deviceId: string, metric: string, interval: string) => {
    return apiCall<any>(`/smartwatch/vitals/${deviceId}/aggregated?metric_type=${metric}&interval=${interval}`);
  },

  analyzeHealth: async (data: any) => {
    return apiCall<any>('/smartwatch/analyze', {
      method: 'POST',
      body: data,
    });
  },

  // ==========================================================================
  // MEDICATIONS API
  // ==========================================================================

  /**
   * Get all medications for a user
   */
  getMedications: async (userId: string) => {
    return apiCall<Array<{
      id: string;
      name: string;
      dosage: string;
      schedule: string[];
      frequency: string;
      startDate?: string;
      endDate?: string;
      notes?: string;
      quantity?: number;
      instructions?: string;
      times: string[];
      takenToday: boolean[];
    }>>(`/users/${userId}/medications`);
  },

  /**
   * Add a new medication for a user
   */
  addMedication: async (userId: string, medication: {
    name: string;
    dosage: string;
    schedule: string[];
    frequency: string;
    startDate?: string;
    endDate?: string;
    notes?: string;
    quantity?: number;
    instructions?: string;
    times?: string[];
    takenToday?: boolean[];
  }) => {
    return apiCall<{
      id: string;
      name: string;
      dosage: string;
      schedule: string[];
      frequency: string;
      startDate?: string;
      endDate?: string;
      notes?: string;
      quantity?: number;
      instructions?: string;
      times: string[];
      takenToday: boolean[];
    }>(`/users/${userId}/medications`, {
      method: 'POST',
      body: medication,
    });
  },

  /**
   * Update an existing medication
   */
  updateMedication: async (userId: string, medicationId: string, medication: Partial<{
    name: string;
    dosage: string;
    schedule: string[];
    frequency: string;
    startDate?: string;
    endDate?: string;
    notes?: string;
    quantity?: number;
    instructions?: string;
    times?: string[];
    takenToday?: boolean[];
  }>) => {
    return apiCall<{
      id: string;
      name: string;
      dosage: string;
      schedule: string[];
      frequency: string;
      startDate?: string;
      endDate?: string;
      notes?: string;
      quantity?: number;
      instructions?: string;
      times: string[];
      takenToday: boolean[];
    }>(`/users/${userId}/medications/${medicationId}`, {
      method: 'PUT',
      body: medication,
    });
  },

  /**
   * Delete a medication
   */
  deleteMedication: async (userId: string, medicationId: string) => {
    return apiCall<{ message: string }>(`/users/${userId}/medications/${medicationId}`, {
      method: 'DELETE',
    });
  },

  // ==========================================================================
  // PROFILE API
  // ==========================================================================

  /**
   * Get user profile
   */
  getProfile: async (userId: string) => {
    return apiCall<{
      id: string;
      name: string;
      email: string;
      phone: string;
      dob: string;
      gender: string;
      conditions: string[];
      allergies: string[];
      medications: string[];
      emergencyContact: { name: string; relation: string; phone: string };
      avatar: string;
    }>(`/profile/${userId}`);
  },

  /**
   * Update user profile
   */
  updateProfile: async (userId: string, profile: Partial<{
    name: string;
    email: string;
    phone: string;
    dob: string;
    gender: string;
  }>) => {
    return apiCall<any>(`/profile/${userId}`, {
      method: 'PUT',
      body: profile,
    });
  },

  /**
   * Update user avatar
   */
  updateAvatar: async (userId: string, avatar: string) => {
    return apiCall<{ message: string; avatar: string }>(`/profile/${userId}/avatar`, {
      method: 'PUT',
      body: { avatar },
    });
  },

  /**
   * Update emergency contact
   */
  updateEmergencyContact: async (userId: string, contact: { name: string; relation: string; phone: string }) => {
    return apiCall<{ name: string; relation: string; phone: string }>(`/profile/${userId}/emergency-contact`, {
      method: 'PUT',
      body: contact,
    });
  },

  /**
   * Add a medical condition
   */
  addCondition: async (userId: string, value: string) => {
    return apiCall<{ conditions: string[] }>(`/profile/${userId}/conditions`, {
      method: 'POST',
      body: { value },
    });
  },

  /**
   * Remove a medical condition
   */
  removeCondition: async (userId: string, condition: string) => {
    return apiCall<{ conditions: string[] }>(`/profile/${userId}/conditions/${encodeURIComponent(condition)}`, {
      method: 'DELETE',
    });
  },

  /**
   * Add an allergy
   */
  addAllergy: async (userId: string, value: string) => {
    return apiCall<{ allergies: string[] }>(`/profile/${userId}/allergies`, {
      method: 'POST',
      body: { value },
    });
  },

  /**
   * Remove an allergy
   */
  removeAllergy: async (userId: string, allergy: string) => {
    return apiCall<{ allergies: string[] }>(`/profile/${userId}/allergies/${encodeURIComponent(allergy)}`, {
      method: 'DELETE',
    });
  },

  /**
   * Get family members
   */
  getFamilyMembers: async (userId: string) => {
    return apiCall<Array<{
      id: string;
      name: string;
      relation: string;
      avatar: string;
      accessLevel: string;
      status: string;
      lastActive: string;
    }>>(`/profile/${userId}/family`);
  },

  /**
   * Add a family member
   */
  addFamilyMember: async (userId: string, member: { name: string; relation: string; avatar?: string; accessLevel?: string; status?: string }) => {
    return apiCall<any>(`/profile/${userId}/family`, {
      method: 'POST',
      body: member,
    });
  },

  /**
   * Remove a family member
   */
  removeFamilyMember: async (userId: string, memberId: string) => {
    return apiCall<{ message: string }>(`/profile/${userId}/family/${memberId}`, {
      method: 'DELETE',
    });
  },

  // ==========================================================================
  // APP SETTINGS API
  // ==========================================================================

  /**
   * Get app settings for a user
   */
  getAppSettings: async (userId: string) => {
    return apiCall<{
      user_id: string;
      notifications: { all: boolean; meds: boolean; insights: boolean };
      preferences: { units: string; language: string; theme: string };
    }>(`/settings/${userId}`);
  },

  /**
   * Update app settings
   */
  updateAppSettings: async (userId: string, settings: {
    notifications?: { all: boolean; meds: boolean; insights: boolean };
    preferences?: { units: string; language: string; theme: string };
  }) => {
    return apiCall<any>(`/settings/${userId}`, {
      method: 'PUT',
      body: settings,
    });
  },

  /**
   * Get connected devices
   */
  getDevices: async (userId: string) => {
    return apiCall<Array<{
      id: string;
      name: string;
      type: string;
      lastSync: string;
      status: string;
      battery: number;
    }>>(`/settings/${userId}/devices`);
  },

  /**
   * Add a connected device
   */
  addDevice: async (userId: string, device: { id: string; name: string; type: string; status?: string; battery?: number }) => {
    return apiCall<any>(`/settings/${userId}/devices`, {
      method: 'POST',
      body: device,
    });
  },

  /**
   * Remove a connected device
   */
  removeDevice: async (userId: string, deviceId: string) => {
    return apiCall<{ message: string }>(`/settings/${userId}/devices/${deviceId}`, {
      method: 'DELETE',
    });
  },

  // ==========================================================================
  // INTEGRATIONS API
  // ==========================================================================

  getWeeklySummary: async (userId: string) => {
    return apiCall<any>(`/integrations/weekly-summary/${userId}`);
  },

  predictFromDocument: async (documentId: string, userId: string, patientProfile: any = {}) => {
    return apiCall<any>('/integrations/predict-from-document', {
      method: 'POST',
      body: { document_id: documentId, user_id: userId, patient_profile: patientProfile },
    });
  },

  triggerWeeklySummary: async (userId: string) => {
    return apiCall<any>('/weekly-summary/trigger', {
      method: 'POST',
      body: { user_id: userId },
    });
  },

  // ==========================================================================
  // MEDICAL AI API
  // ==========================================================================

  extractMedicalEntities: async (text: string) => {
    return apiCall<any>('/medical-ai/extract-entities', {
      method: 'POST',
      body: { text },
    });
  },

  getPatientSummary: async (userId: string) => {
    return apiCall<any>('/medical-ai/patient-summary', {
      method: 'POST',
      body: { user_id: userId },
    });
  },

  expandTerminology: async (term: string) => {
    return apiCall<any>('/medical-ai/terminology', {
      method: 'POST',
      body: { term },
    });
  },

  // ==========================================================================
  // TOOLS API
  // ==========================================================================

  recordBloodPressure: async (systolic: number, diastolic: number, userId: string) => {
    return apiCall<any>('/tools/blood-pressure', {
      method: 'POST',
      body: { systolic, diastolic, user_id: userId, timestamp: new Date().toISOString() },
    });
  },

  recordHeartRate: async (bpm: number, userId: string) => {
    return apiCall<any>('/tools/heart-rate', {
      method: 'POST',
      body: { bpm, user_id: userId, timestamp: new Date().toISOString() },
    });
  },

  checkDrugInteractions: async (medications: string[]) => {
    return apiCall<any>('/tools/drug-interactions', {
      method: 'POST',
      body: { medications },
    });
  },

  symptomTriage: async (symptoms: string[], userId: string) => {
    return apiCall<any>('/tools/symptom-triage', {
      method: 'POST',
      body: { symptoms, user_id: userId },
    });
  },

  // ==========================================================================
  // COMPLIANCE API
  // ==========================================================================

  getDisclaimer: async (type: string) => {
    return apiCall<any>(`/compliance/disclaimer/${type}`);
  },

  encryptPHI: async (data: any) => {
    return apiCall<any>('/compliance/encrypt-phi', {
      method: 'POST',
      body: { data },
    });
  },

  getVerificationQueue: async () => {
    return apiCall<any>('/compliance/verification/pending');
  },

  submitVerification: async (itemId: string, verified: boolean, notes?: string) => {
    return apiCall<any>('/compliance/verification/submit', {
      method: 'POST',
      body: { item_id: itemId, verified, notes },
    });
  },

  // ==========================================================================
  // CONSENT MANAGEMENT API
  // ==========================================================================

  getConsent: async (userId: string) => {
    return apiCall<any>(`/consent/${userId}`);
  },

  updateConsent: async (userId: string, consents: any) => {
    return apiCall<any>(`/consent/${userId}`, {
      method: 'PUT',
      body: consents,
    });
  },

  revokeConsent: async (userId: string, consentType: string) => {
    return apiCall<any>(`/consent/${userId}/${consentType}`, {
      method: 'DELETE',
    });
  },

  // ==========================================================================
  // FEEDBACK API
  // ==========================================================================

  /**
   * Submit user feedback
   */
  submitFeedback: async (data: { type: string; message: string; rating?: number; userId: string }) => {
    return apiCall<{ message: string }>('/feedback', {
      method: 'POST',
      body: data,
    });
  },

  // ==========================================================================
  // SMARTWATCH ADDITIONAL APIs
  // ==========================================================================

  // Removed duplicate analyzeHealth function to fix TS1117 error

  // ==========================================================================
  // EVALUATION APIs (Admin/Developer)
  // ==========================================================================

  evaluateRAG: async (queries: string[], groundTruth?: any[]) => {
    return apiCall<any>('/evaluation/rag', {
      method: 'POST',
      body: { queries, ground_truth: groundTruth },
    });
  },

  getWebSocketUrl: (endpoint: string) => {
    const wsBase = API_BASE_URL.replace('http', 'ws');
    return `${wsBase}${endpoint}`;
  },

  // ==========================================================================
  // APPOINTMENTS API
  // ==========================================================================

  /**
   * Get providers with optional filters
   */
  getProviders: async (params?: {
    specialty?: string;
    search?: string;
    telehealth?: boolean;
    limit?: number;
    offset?: number;
  }) => {
    const query = new URLSearchParams();
    if (params?.specialty && params.specialty !== 'All') query.set('specialty', params.specialty);
    if (params?.search) query.set('search', params.search);
    if (params?.telehealth !== undefined) query.set('telehealth', String(params.telehealth));
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));
    const qs = query.toString();
    const data = await apiCall<any[]>(`/appointments/providers${qs ? `?${qs}` : ''}`, { method: 'GET' });
    // Map provider_id to id for frontend type compatibility
    return (data || []).map((p: any) => ({ ...p, id: p.provider_id || p.id }));
  },

  /**
   * Get provider specialties list
   */
  getSpecialties: async () => {
    return apiCall<{ specialties: string[] }>('/appointments/providers/specialties', { method: 'GET' });
  },

  /**
   * Get a single provider by ID
   */
  getProvider: async (providerId: string) => {
    const data = await apiCall<any>(`/appointments/providers/${providerId}`, { method: 'GET' });
    if (data) {
      data.id = data.provider_id || data.id;
    }
    return data;
  },

  /**
   * Get available time slots for a provider on a date
   */
  getProviderAvailability: async (providerId: string, date: string) => {
    return apiCall<{ provider_id: string; date: string; slots: string[] }>(
      `/appointments/providers/${providerId}/availability?date=${date}`,
      { method: 'GET' }
    );
  },

  /**
   * Get all appointments for a user
   */
  getUserAppointments: async (userId: string, params?: {
    status?: string;
    upcoming?: boolean;
    limit?: number;
    offset?: number;
  }) => {
    const query = new URLSearchParams();
    if (params?.status) query.set('status', params.status);
    if (params?.upcoming) query.set('upcoming', 'true');
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));
    const qs = query.toString();
    return apiCall<any[]>(`/appointments/${userId}${qs ? `?${qs}` : ''}`, { method: 'GET' });
  },

  /**
   * Get a single appointment
   */
  getAppointment: async (userId: string, appointmentId: string) => {
    return apiCall<any>(`/appointments/${userId}/${appointmentId}`, { method: 'GET' });
  },

  /**
   * Book a new appointment
   */
  createAppointment: async (userId: string, data: {
    provider_id: string;
    date: string;
    time: string;
    appointment_type: string;
    reason?: string;
    intake_summary?: string;
    shared_chart_data?: Record<string, any>;
    insurance_provider?: string;
    insurance_member_id?: string;
    insurance_group_id?: string;
    duration_minutes?: number;
    estimated_cost?: number;
  }) => {
    return apiCall<any>(`/appointments/${userId}`, {
      method: 'POST',
      body: data,
    });
  },

  /**
   * Update an appointment
   */
  updateAppointment: async (userId: string, appointmentId: string, data: {
    appointment_type?: string;
    reason?: string;
    intake_summary?: string;
    consultation_summary?: string;
    insurance_provider?: string;
    insurance_member_id?: string;
    insurance_group_id?: string;
    status?: string;
    actual_cost?: number;
    virtual_link?: string;
    location?: string;
  }) => {
    return apiCall<any>(`/appointments/${userId}/${appointmentId}`, {
      method: 'PUT',
      body: data,
    });
  },

  /**
   * Cancel an appointment
   */
  cancelAppointment: async (userId: string, appointmentId: string, reason?: string) => {
    return apiCall<any>(`/appointments/${userId}/${appointmentId}/cancel`, {
      method: 'POST',
      body: { reason },
    });
  },

  /**
   * Mark an appointment as completed
   */
  completeAppointment: async (userId: string, appointmentId: string, summary?: string) => {
    return apiCall<any>(`/appointments/${userId}/${appointmentId}/complete`, {
      method: 'POST',
      body: { consultation_summary: summary },
    });
  },

  /**
   * Get user insurance info
   */
  getUserInsurance: async (userId: string) => {
    return apiCall<any[]>(`/appointments/${userId}/insurance`, { method: 'GET' });
  },

  /**
   * Save insurance info
   */
  saveInsurance: async (userId: string, data: {
    insurance_provider: string;
    member_id: string;
    group_id?: string;
    plan_type?: string;
  }) => {
    return apiCall<any>(`/appointments/${userId}/insurance`, {
      method: 'POST',
      body: data,
    });
  },

  /**
   * AI intake / symptom triage
   */
  analyzeIntake: async (symptoms: string, userName?: string) => {
    return apiCall<{
      urgency: 'emergency' | 'urgent' | 'routine';
      reason: string;
      summary: string;
      recommendation: string;
    }>('/appointments/intake/analyze', {
      method: 'POST',
      body: { symptoms, user_name: userName || 'Patient' },
    });
  },
};

// ============================================================================
// STRUCTURED OUTPUT TYPE DEFINITIONS
// These types match the Pydantic schemas in the backend
// ============================================================================

/** Confidence levels for LLM responses */
export type ResponseConfidence = 'high' | 'medium' | 'low' | 'uncertain';

/** Healthcare-specific intents */
export type HealthIntent =
  | 'symptom_report'
  | 'medication_question'
  | 'lifestyle_advice'
  | 'emergency'
  | 'appointment'
  | 'general_health'
  | 'vital_signs'
  | 'mental_health'
  | 'unknown';

/** Urgency classification for health queries */
export type UrgencyLevel =
  | 'critical'   // Requires immediate attention
  | 'high'       // Should see doctor soon
  | 'moderate'   // Can wait for regular appointment
  | 'low'        // General information/advice
  | 'informational';  // Just seeking knowledge

/** Available schema names */
export type StructuredSchemaName =
  | 'CardioHealthAnalysis'
  | 'SimpleIntentAnalysis'
  | 'ConversationResponse'
  | 'VitalSignsAnalysis'
  | 'MedicationInfo';

/** Entity extracted from user input */
export interface ExtractedEntity {
  entity_type: string;
  value: string;
  confidence: number;
  context?: string;
}

/** Suggested follow-up question */
export interface FollowUpQuestion {
  question: string;
  priority: number;
  reason?: string;
}

/** Health recommendation */
export interface HealthRecommendation {
  recommendation: string;
  category: string;
  urgency: UrgencyLevel;
  evidence_based: boolean;
}

/**
 * Main structured output for cardiovascular health analysis
 * This is the primary schema for health-related queries
 */
export interface CardioHealthAnalysis {
  intent: HealthIntent;
  intent_confidence: number;
  sentiment: string;
  urgency: UrgencyLevel;
  entities: ExtractedEntity[];
  response: string;
  explanation?: string;
  recommendations: HealthRecommendation[];
  follow_up_questions: FollowUpQuestion[];
  requires_professional: boolean;
  disclaimer?: string;
  confidence: ResponseConfidence;
}

/** Lightweight intent analysis */
export interface SimpleIntentAnalysis {
  intent: string;
  confidence: number;
  keywords: string[];
  summary: string;
}

/** Structured conversation response */
export interface ConversationResponse {
  response: string;
  tone: string;
  topics: string[];
  action_items: string[];
  needs_clarification: boolean;
}

/** Vital signs interpretation */
export interface VitalSignsAnalysis {
  metric_type: string;
  value: number;
  unit: string;
  status: string;
  interpretation: string;
  recommendations: string[];
  reference_range?: string;
}

/** Medication information */
export interface MedicationInfo {
  medication_name: string;
  purpose: string;
  common_side_effects: string[];
  interactions_warning?: string;
  dosage_reminder?: string;
  important_notes: string[];
  consult_doctor: boolean;
}

/** Status of structured outputs feature */
export interface StructuredOutputsStatus {
  enabled: boolean;
  message: string;
  available_schemas: string[];
  endpoints?: string[];
}

/** Request for structured health analysis */
export interface StructuredHealthAnalysisRequest {
  message: string;
  session_id?: string;
  patient_context?: Record<string, any>;
  model?: string;
}

/** Request for structured conversation */
export interface StructuredConversationRequest {
  message: string;
  conversation_history?: Array<{ role: string; content: string }>;
  session_id?: string;
}

/** Generic structured response wrapper */
export interface StructuredResponse<T> {
  success: boolean;
  data: T;
  metadata: {
    generation_time_ms: number;
    model?: string;
    schema: string;
  };
}

// ============================================================================
// USER PREFERENCES TYPES
// ============================================================================

/** User preferences structure */
export interface UserPreferences {
  user_id: string;
  theme?: 'light' | 'dark' | 'system';
  language?: string;
  notifications_enabled?: boolean;
  email_notifications?: boolean;
  reminder_time?: string;
  health_goals?: {
    steps?: number;
    water_intake?: number;
  };
  privacy_settings?: {
    share_with_doctors?: boolean;
    share_with_family?: boolean;
    anonymous_analytics?: boolean;
  };
  accessibility?: {
    font_size?: 'small' | 'medium' | 'large';
    high_contrast?: boolean;
    reduce_motion?: boolean;
  };
  agent_settings?: {
    model?: string;
    persona?: string;
    response_length?: string;
    temperature?: number;
  };
  created_at?: string;
  updated_at?: string;
}

// ============================================================================
// GDPR TYPES
// ============================================================================

/** GDPR data export response */
export interface GDPRExportData {
  user_id: string;
  export_date: string;
  data: {
    profile: Record<string, unknown>;
    preferences: UserPreferences;
    conversations: Array<{
      session_id: string;
      messages: Array<{ role: string; content: string; timestamp: string }>;
    }>;
    health_data: Array<{
      type: string;
      data: Record<string, unknown>;
      timestamp: string;
    }>;
    audit_log: AuditLogEntry[];
  };
  format_version: string;
}

/** GDPR delete response */
export interface GDPRDeleteResponse {
  success: boolean;
  user_id: string;
  deleted_items: {
    profile: boolean;
    preferences: boolean;
    conversations: number;
    health_data: number;
    audit_entries: number;
  };
  deletion_date: string;
  confirmation_id: string;
}

/** Audit log entry for data access tracking */
export interface AuditLogEntry {
  id: string;
  user_id: string;
  action: 'read' | 'write' | 'delete' | 'export';
  resource_type: string;
  resource_id?: string;
  timestamp: string;
  ip_address?: string;
  user_agent?: string;
  details?: Record<string, unknown>;
}

// ============================================================================
// MODEL MANAGEMENT TYPES
// ============================================================================

/** Model versions response */
export interface ModelVersionsResponse {
  models: Array<{
    name: string;
    current_version: string;
    versions: string[];
    last_updated: string;
  }>;
}

/** Model history response */
export interface ModelHistoryResponse {
  model_name: string;
  history: Array<{
    version: string;
    deployed_at: string;
    metrics?: {
      accuracy?: number;
      latency_ms?: number;
    };
    notes?: string;
  }>;
}

/** Models list response */
export interface ModelsListResponse {
  models: Array<{
    name: string;
    type: string;
    status: 'active' | 'deprecated' | 'testing';
    description?: string;
  }>;
}

// ============================================================================
// DOCUMENT TYPES
// ============================================================================

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
}

export interface DocumentProcessingResult {
  text: string;
  metadata: Record<string, any>;
  entities: any[];
}

export interface DocumentResponse {
  id: string;
  content: string;
  processed_at: string;
}

// ============================================================================
// VISION TYPES
// ============================================================================

export interface ECGAnalysisResponse {
  rhythm: string;
  heart_rate_bpm?: number;
  abnormalities: string[];
  recommendations: string[];
  confidence: number;
}

// ============================================================================
// CALENDAR TYPES
// ============================================================================

export interface SyncResponse {
  events_synced: number;
  reminders_created: number;
  sync_completed_at: string;
}

export interface CalendarEventResponse {
  id: string;
  title: string;
  start_time: string;
  end_time: string;
  location?: string;
  description?: string;
}

export interface ReminderResponse {
  id: string;
  appointment_id: string;
  scheduled_for: string;
  status: string;
}

export { APIError };
