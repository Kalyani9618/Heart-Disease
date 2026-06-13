/**
 * Enhanced Memory Service for Healthcare AI
 *
 * Integrates with backend memory management API for:
 * - User preferences persistence
 * - Chat session management
 * - Context retrieval preview
 * - GDPR compliance (export, delete)
 *
 * Implements chat.md architecture principles on the frontend.
 *
 * @version 2.0.0
 */

import { authService } from './authService';

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface UserPreference {
  key: string;
  value: any;
  category?: string;
  isSensitive?: boolean;
  updatedAt?: string;
}

export interface ContextItem {
  type: string;
  source: string;
  relevanceScore: number;
  tokenEstimate: number;
  dataPreview: Record<string, any>;
}

export interface ChatSession {
  sessionId: string;
  userId?: string;
  createdAt?: string;
  lastActivity?: string;
  messageCount: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  metadata?: Record<string, any>;
}

export interface AIQueryResponse {
  response: string;
  sessionId: string;
  success: boolean;
  contextUsed: Array<{
    type: string;
    source: string;
    relevance: string;
  }>;
  metadata: {
    userId: string;
    timestamp: string;
    processingTimeMs: number;
    tokensEstimated: number;
    contextItemsCount: number;
    aiProvider: string;
    isEmergency: boolean;
  };
  audit: {
    action: string;
    userId: string;
    sessionId: string;
    timestamp: string;
    contextTypesAccessed: string[];
    phiAccessed: boolean;
  };
  error?: string;
}

export interface GDPRExportData {
  userId: string;
  exportTimestamp: string;
  preferences: Array<{
    key: string;
    value: any;
    dataType: string;
    category: string;
    isSensitive: boolean;
    createdAt?: string;
    updatedAt?: string;
  }>;
  totalCount: number;
}

export interface MemoryHealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    preferences: any;
    contextRetriever: any;
    chatHistory: any;
  };
  timestamp: string;
}

// ============================================================================
// Memory Chunk (Legacy Support)
// ============================================================================

export interface MemoryChunk {
  id: string;
  source: 'medication' | 'assessment' | 'general';
  text: string;
  embedding?: number[];
  timestamp: number;
}

// ============================================================================
// Configuration
// ============================================================================

interface MemoryServiceConfig {
  baseUrl: string;
  timeout: number;
  enableLocalCache: boolean;
  cacheExpiryMs: number;
}

// Build base URL for memory service - must match backend route prefix
const MEMORY_BASE_URL = (() => {
  const nlpUrl = import.meta.env?.VITE_NLP_SERVICE_URL;
  const base = nlpUrl && nlpUrl !== '/' ? nlpUrl.replace(/\/+$/, '') : 'http://localhost:5001';
  return `${base}/memory`;
})();

const DEFAULT_CONFIG: MemoryServiceConfig = {
  baseUrl: MEMORY_BASE_URL,
  timeout: 180000, // 3 minutes - enough for web search / deep research operations
  enableLocalCache: true,
  cacheExpiryMs: 5 * 60 * 1000, // 5 minutes
};

// ============================================================================
// Cache Types
// ============================================================================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

// ============================================================================
// Memory Service Class
// ============================================================================

class EnhancedMemoryService {
  private config: MemoryServiceConfig;
  private preferencesCache: Map<string, CacheEntry<Record<string, any>>> = new Map();
  private sessionsCache: Map<string, CacheEntry<ChatSession[]>> = new Map();

  // Legacy support
  private localStore: MemoryChunk[] = [];
  private isInitialized = false;
  private readonly MEMORY_STORE_KEY = 'cardio_ai_vector_store';

  constructor(config: Partial<MemoryServiceConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.loadLocalStore();
  }

  // ==========================================================================
  // Configuration
  // ==========================================================================

  /**
   * Update service configuration.
   */
  configure(config: Partial<MemoryServiceConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Initialize the memory service.
   */
  init(apiKey?: string): void {
    this.isInitialized = true;
    console.log('EnhancedMemoryService initialized');
  }

  // ==========================================================================
  // Preferences API
  // ==========================================================================

  /**
   * Get all preferences for a user.
   */
  async getPreferences(
    userId: string,
    options: { includeSensitive?: boolean; category?: string } = {}
  ): Promise<Record<string, any>> {
    // Check cache
    if (this.config.enableLocalCache) {
      const cached = this.preferencesCache.get(userId);
      if (cached && Date.now() - cached.timestamp < this.config.cacheExpiryMs) {
        return cached.data;
      }
    }

    const params = new URLSearchParams();
    if (options.includeSensitive) params.append('include_sensitive', 'true');
    if (options.category) params.append('category', options.category);

    const response = await this.fetch(
      `${this.config.baseUrl}/preferences/${userId}?${params}`
    );
    const data = await response.json();

    // Cache the result
    if (this.config.enableLocalCache) {
      this.preferencesCache.set(userId, {
        data: data.preferences,
        timestamp: Date.now(),
      });
    }

    return data.preferences;
  }

  /**
   * Get a single preference value.
   */
  async getPreference(
    userId: string,
    key: string,
    defaultValue?: any
  ): Promise<any> {
    const response = await this.fetch(
      `${this.config.baseUrl}/preferences/${userId}/${key}?default=${defaultValue ?? ''}`
    );
    const data = await response.json();
    return data.value;
  }

  /**
   * Set a user preference.
   */
  async setPreference(
    userId: string,
    key: string,
    value: any,
    options: { isSensitive?: boolean; category?: string } = {}
  ): Promise<void> {
    const body: UserPreference = {
      key,
      value,
      isSensitive: options.isSensitive ?? false,
      category: options.category ?? 'general',
    };

    await this.fetch(`${this.config.baseUrl}/preferences/${userId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        key: body.key,
        value: body.value,
        is_sensitive: body.isSensitive,
        category: body.category,
      }),
    });

    // Invalidate cache
    this.preferencesCache.delete(userId);
  }

  /**
   * Set multiple preferences at once.
   */
  async setPreferences(
    userId: string,
    preferences: Record<string, any>,
    category: string = 'general'
  ): Promise<number> {
    const response = await this.fetch(
      `${this.config.baseUrl}/preferences/${userId}/bulk`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preferences, category }),
      }
    );

    const data = await response.json();

    // Invalidate cache
    this.preferencesCache.delete(userId);

    return data.preferences_set;
  }

  /**
   * Delete a preference.
   */
  async deletePreference(userId: string, key: string): Promise<boolean> {
    const response = await this.fetch(
      `${this.config.baseUrl}/preferences/${userId}/${key}`,
      { method: 'DELETE' }
    );

    const data = await response.json();

    // Invalidate cache
    this.preferencesCache.delete(userId);

    return data.deleted;
  }

  // ==========================================================================
  // Sessions API
  // ==========================================================================

  /**
   * Get all sessions for a user.
   */
  async getSessions(
    userId: string,
    options: { limit?: number; includeExpired?: boolean } = {}
  ): Promise<ChatSession[]> {
    const params = new URLSearchParams();
    if (options.limit) params.append('limit', String(options.limit));
    if (options.includeExpired) params.append('include_expired', 'true');

    const response = await this.fetch(
      `${this.config.baseUrl}/sessions/${userId}?${params}`
    );
    const data = await response.json();
    return data.sessions;
  }

  /**
   * Get history for a specific session.
   */
  async getSessionHistory(
    sessionId: string,
    limit: number = 100
  ): Promise<{ messages: ChatMessage[]; sessionInfo: any }> {
    const response = await this.fetch(
      `${this.config.baseUrl}/sessions/${sessionId}/history?limit=${limit}`
    );
    return response.json();
  }

  /**
   * Delete a session.
   */
  async deleteSession(sessionId: string): Promise<boolean> {
    const response = await this.fetch(
      `${this.config.baseUrl}/sessions/${sessionId}`,
      { method: 'DELETE' }
    );
    const data = await response.json();
    return data.deleted;
  }

  /**
   * Update a session (e.g. rename).
   */
  async updateSession(sessionId: string, updates: { title?: string }): Promise<ChatSession> {
    const response = await this.fetch(
      `${this.config.baseUrl}/sessions/${sessionId}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      }
    );
    const data = await response.json();
    return data.session;
  }

  // ==========================================================================
  // Context API
  // ==========================================================================

  /**
   * Preview what context will be retrieved for a query.
   */
  async previewContext(
    userId: string,
    sessionId: string,
    query: string,
    contextTypes?: string[]
  ): Promise<ContextItem[]> {
    const params = new URLSearchParams();
    params.append('user_id', userId);
    params.append('session_id', sessionId);

    const response = await this.fetch(
      `${this.config.baseUrl}/context/retrieve?${params}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          context_types: contextTypes,
        }),
      }
    );

    const data = await response.json();
    return data.contexts.map((ctx: any) => ({
      type: ctx.type,
      source: ctx.source,
      relevanceScore: ctx.relevance_score,
      tokenEstimate: ctx.token_estimate,
      dataPreview: ctx.data_preview,
    }));
  }

  /**
   * Get available context types.
   */
  async getContextTypes(): Promise<Array<{ name: string; description: string }>> {
    const response = await this.fetch(`${this.config.baseUrl}/context/types`);
    const data = await response.json();
    return data.context_types;
  }

  // ==========================================================================
  // AI Query API
  // ==========================================================================

  /**
   * Send a query through the integrated AI service.
   */
  async aiQuery(
    userId: string,
    sessionId: string,
    query: string,
    options: {
      patientName?: string;
      patientAge?: number;
      isEmergency?: boolean;
      aiProvider?: 'gemini' | 'ollama';
      searchMode?: 'web_search' | 'deep_search' | 'memory';
      signal?: AbortSignal;
    } = {}
  ): Promise<AIQueryResponse> {
    const params = new URLSearchParams();
    params.append('user_id', userId);
    params.append('session_id', sessionId);

    const fetchOptions: RequestInit = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        patient_name: options.patientName,
        patient_age: options.patientAge,
        is_emergency: options.isEmergency ?? false,
        ai_provider: options.aiProvider,
        search_mode: options.searchMode,
      }),
    };
    if (options.signal) {
      fetchOptions.signal = options.signal;
    }

    const response = await this.fetch(
      `${this.config.baseUrl}/ai/query?${params}`,
      fetchOptions
    );

    const data = await response.json();
    return {
      response: data.response,
      sessionId: data.session_id,
      success: data.success,
      contextUsed: data.context_used,
      metadata: {
        userId: data.metadata.user_id,
        timestamp: data.metadata.timestamp,
        processingTimeMs: data.metadata.processing_time_ms,
        tokensEstimated: data.metadata.tokens_estimated,
        contextItemsCount: data.metadata.context_items_count,
        aiProvider: data.metadata.ai_provider,
        isEmergency: data.metadata.is_emergency,
      },
      audit: {
        action: data.audit.action,
        userId: data.audit.user_id,
        sessionId: data.audit.session_id,
        timestamp: data.audit.timestamp,
        contextTypesAccessed: data.audit.context_types_accessed,
        phiAccessed: data.audit.phi_accessed,
      },
      error: data.error,
    };
  }

  // ==========================================================================
  // GDPR Compliance
  // ==========================================================================

  /**
   * Export all user data (GDPR compliance).
   */
  async exportAllData(userId: string): Promise<GDPRExportData> {
    const response = await this.fetch(
      `${this.config.baseUrl}/gdpr/export/${userId}`,
      { method: 'POST' }
    );

    const data = await response.json();
    return {
      userId: data.user_id,
      exportTimestamp: data.export_timestamp,
      preferences: data.preferences,
      totalCount: data.total_count,
    };
  }

  /**
   * Download exported data as a file.
   */
  async downloadExport(userId: string): Promise<Blob> {
    const data = await this.exportAllData(userId);
    const jsonString = JSON.stringify(data, null, 2);
    return new Blob([jsonString], { type: 'application/json' });
  }

  /**
   * Delete all user data (GDPR right to erasure).
   */
  async deleteAllData(userId: string): Promise<{
    preferencesDeleted: number;
    sessionsDeleted: number;
    timestamp: string;
  }> {
    const response = await this.fetch(
      `${this.config.baseUrl}/gdpr/delete/${userId}?confirm=true`,
      { method: 'DELETE' }
    );

    const data = await response.json();

    // Clear all caches
    this.preferencesCache.delete(userId);
    this.sessionsCache.delete(userId);

    return {
      preferencesDeleted: data.preferences_deleted,
      sessionsDeleted: data.sessions_deleted,
      timestamp: data.timestamp,
    };
  }

  // ==========================================================================
  // Audit
  // ==========================================================================

  /**
   * Get audit log for a user.
   */
  async getAuditLog(
    userId: string,
    options: { limit?: number; action?: string } = {}
  ): Promise<Array<{
    id: number;
    preferenceKey: string;
    action: string;
    timestamp: string;
    ipAddress?: string;
  }>> {
    const params = new URLSearchParams();
    if (options.limit) params.append('limit', String(options.limit));
    if (options.action) params.append('action', options.action);

    const response = await this.fetch(
      `${this.config.baseUrl}/audit/${userId}?${params}`
    );

    const data = await response.json();
    return data.audit_logs.map((log: any) => ({
      id: log.id,
      preferenceKey: log.preference_key,
      action: log.action,
      timestamp: log.timestamp,
      ipAddress: log.ip_address,
    }));
  }

  // ==========================================================================
  // Health Check
  // ==========================================================================

  /**
   * Check health of memory management system.
   */
  async healthCheck(): Promise<MemoryHealthStatus> {
    const response = await this.fetch(`${this.config.baseUrl}/health`);
    return response.json();
  }

  // ==========================================================================
  // Legacy Support: Local Memory Store
  // ==========================================================================

  /**
   * Load local memory store from localStorage.
   */
  private loadLocalStore(): void {
    try {
      const saved = localStorage.getItem(this.MEMORY_STORE_KEY);
      if (saved) {
        this.localStore = JSON.parse(saved);
      }
    } catch (e) {
      console.warn('Failed to load local memory store:', e);
      this.localStore = [];
    }
  }

  /**
   * Save local memory store to localStorage.
   */
  private saveLocalStore(): void {
    try {
      localStorage.setItem(this.MEMORY_STORE_KEY, JSON.stringify(this.localStore));
    } catch (e) {
      console.warn('Failed to save local memory store:', e);
    }
  }

  /**
   * Sync local app data into memory chunks (legacy support).
   */
  async syncContext(): Promise<void> {
    if (!this.isInitialized) return;

    const newChunks: MemoryChunk[] = [];

    // 1. Medications
    const medsRaw = localStorage.getItem('user_medications');
    if (medsRaw) {
      try {
        const meds = JSON.parse(medsRaw);
        meds.forEach((m: any) => {
          newChunks.push({
            id: `med_${m.id}`,
            source: 'medication',
            text: `Medication: ${m.name} ${m.dosage}, taken ${m.frequency} at ${m.times?.join(',') ?? 'unspecified'}.`,
            timestamp: Date.now(),
          });
        });
      } catch (e) {
        console.warn('Failed to parse medications:', e);
      }
    }

    // 2. Assessment
    const assessRaw = localStorage.getItem('last_assessment');
    if (assessRaw) {
      try {
        const a = JSON.parse(assessRaw);
        newChunks.push({
          id: `assess_${new Date(a.date).getTime()}`,
          source: 'assessment',
          text: `Health Assessment from ${new Date(a.date).toLocaleDateString()}: Risk ${a.risk}, Score ${a.score}, BP ${a.vitals?.systolic ?? 'N/A'}, Cholesterol ${a.vitals?.cholesterol ?? 'N/A'}. Details: ${a.details ?? 'None'}`,
          timestamp: new Date(a.date).getTime(),
        });
      } catch (e) {
        console.warn('Failed to parse assessment:', e);
      }
    }

    // Process Chunks: Store locally for context retrieval
    let updated = false;
    for (const chunk of newChunks) {
      const existing = this.localStore.find(c => c.id === chunk.id);
      if (!existing) {
        this.localStore.push(chunk);
        updated = true;
      }
    }

    if (updated) {
      this.saveLocalStore();
    }
  }

  /**
   * Search local memory store (legacy support).
   */
  async search(query: string, topK: number = 5): Promise<string> {
    if (!this.isInitialized || this.localStore.length === 0) return '';

    try {
      // Simple text-based search (no embeddings)
      const queryWords = query.toLowerCase().split(/\s+/);
      const scored = this.localStore.map(chunk => {
        const chunkText = chunk.text.toLowerCase();
        const score = queryWords.filter(word => chunkText.includes(word)).length;
        return { ...chunk, score };
      });

      // Sort & Slice
      scored.sort((a, b) => b.score - a.score);
      const topResults = scored.slice(0, topK).filter(r => r.score > 0);

      // Format Context
      if (topResults.length > 0) {
        return `RELEVANT CONTEXT FOUND:\n${topResults.map(r => `- ${r.text}`).join('\n')}\n`;
      }
      return '';
    } catch (e) {
      console.error('Search failed', e);
      return '';
    }
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Fetch wrapper with timeout and error handling.
   */
  private async fetch(url: string, options: RequestInit = {}): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    // Combine caller's signal with our timeout signal so both can trigger abort
    const callerSignal = options.signal;
    if (callerSignal) {
      callerSignal.addEventListener('abort', () => controller.abort(), { once: true });
    }

    try {
      const authHeader = authService.getAuthHeader();
      const response = await fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          ...(authHeader && { Authorization: authHeader }),
        },
        signal: controller.signal,
      });

      // Handle 401 Unauthorized - token expired or invalid
      if (response.status === 401) {
        authService.clearAuth();
        if (typeof window !== 'undefined') {
          window.location.hash = '#/login';
        }
        throw new Error('Session expired. Please log in again.');
      }

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`HTTP ${response.status}: ${error}`);
      }

      return response;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Clear all caches.
   */
  clearCache(): void {
    this.preferencesCache.clear();
    this.sessionsCache.clear();
  }

  /**
   * Get cache statistics.
   */
  getCacheStats(): { preferences: number; sessions: number } {
    return {
      preferences: this.preferencesCache.size,
      sessions: this.sessionsCache.size,
    };
  }
}

// ============================================================================
// Export Singleton Instance
// ============================================================================

export const memoryService = new EnhancedMemoryService();

// Export class for custom instantiation
export { EnhancedMemoryService };

// Export default for backward compatibility
export default memoryService;
