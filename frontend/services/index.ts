/**
 * Services Index
 *
 * Central export for all service modules
 */

// API Services
export * from './apiClient';

// Memory / State Persistence
// Note: Using selective exports to avoid naming conflicts with apiClient.ts
export {
  EnhancedMemoryService,
  memoryService,
} from './memoryService';

export type {
  UserPreference,
  ContextItem,
  ChatSession,
  ChatMessage,
  AIQueryResponse,
  MemoryHealthStatus,
  MemoryChunk,
} from './memoryService';

// PDF Export
export {
  pdfExportService,
  PDFExportService,
} from './pdfExport';

export type {
  HealthReport,
  BiometricEntry,
  ChatExport,
} from './pdfExport';

// Service Worker / PWA
export {
  serviceWorker,
  registerServiceWorker,
  unregisterServiceWorker,
  clearCaches,
  skipWaiting,
  cacheHealthData,
  queueOfflineRequest,
  triggerSync,
  getOnlineStatus,
  getRegistration,
  isServiceWorkerActive,
  isServiceWorkerSupported,
} from './serviceWorker';
// Document Processing
export * from './documentService';

// Vision Analysis
export * from './visionService';

// Calendar Integration
export * from './calendarService';

// Notification Services
export * from './notificationService';

// Note: api.types exports are intentionally NOT re-exported here
// to avoid naming conflicts with apiClient.ts.
// Import directly from './api.types' when needed.
