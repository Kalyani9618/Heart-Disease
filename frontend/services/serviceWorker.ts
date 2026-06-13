/**
 * Service Worker Registration
 *
 * Manages the registration and lifecycle of the service worker
 * for offline support and PWA functionality.
 */

// ============================================================================
// Types
// ============================================================================

interface ServiceWorkerConfig {
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  onOffline?: () => void;
  onOnline?: () => void;
}

// ============================================================================
// Registration State
// ============================================================================

let swRegistration: ServiceWorkerRegistration | null = null;
let isOnline = navigator.onLine;

// ============================================================================
// Registration Functions
// ============================================================================

/**
 * Check if service workers are supported
 */
export function isServiceWorkerSupported(): boolean {
  return 'serviceWorker' in navigator;
}

/**
 * Register the service worker
 */
export async function registerServiceWorker(
  config?: ServiceWorkerConfig
): Promise<ServiceWorkerRegistration | null> {
  if (!isServiceWorkerSupported()) {
    console.log('[SW Registration] Service workers not supported');
    return null;
  }

  // Only register in production or when explicitly enabled
  const isDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  if (isDev) {
    console.log('[SW Registration] Skipping SW registration in development');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    });

    swRegistration = registration;
    console.log('[SW Registration] Service Worker registered:', registration.scope);

    // Handle registration state
    handleRegistration(registration, config);

    // Set up online/offline listeners
    setupNetworkListeners(config);

    return registration;
  } catch (error) {
    console.error('[SW Registration] Failed to register:', error);
    return null;
  }
}

/**
 * Handle registration lifecycle
 */
function handleRegistration(
  registration: ServiceWorkerRegistration,
  config?: ServiceWorkerConfig
): void {
  // Check if there's an update available
  registration.onupdatefound = () => {
    const installingWorker = registration.installing;

    if (!installingWorker) return;

    installingWorker.onstatechange = () => {
      if (installingWorker.state === 'installed') {
        if (navigator.serviceWorker.controller) {
          // New content is available, notify user
          console.log('[SW Registration] New content available');
          config?.onUpdate?.(registration);
        } else {
          // Content is cached for offline use
          console.log('[SW Registration] Content cached for offline use');
          config?.onSuccess?.(registration);
        }
      }
    };
  };
}

/**
 * Set up network status listeners
 */
function setupNetworkListeners(config?: ServiceWorkerConfig): void {
  window.addEventListener('online', () => {
    isOnline = true;
    console.log('[SW Registration] Back online');
    config?.onOnline?.();
  });

  window.addEventListener('offline', () => {
    isOnline = false;
    console.log('[SW Registration] Gone offline');
    config?.onOffline?.();
  });
}

/**
 * Unregister all service workers
 */
export async function unregisterServiceWorker(): Promise<boolean> {
  if (!isServiceWorkerSupported()) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    const success = await registration.unregister();

    if (success) {
      console.log('[SW Registration] Service Worker unregistered');
      swRegistration = null;
    }

    return success;
  } catch (error) {
    console.error('[SW Registration] Failed to unregister:', error);
    return false;
  }
}

// ============================================================================
// Cache Management
// ============================================================================

/**
 * Clear all service worker caches
 */
export async function clearCaches(): Promise<void> {
  if (swRegistration?.active) {
    swRegistration.active.postMessage({ type: 'CLEAR_CACHE' });
    console.log('[SW Registration] Cache clear requested');
  }
}

/**
 * Force service worker to skip waiting and activate
 */
export function skipWaiting(): void {
  if (swRegistration?.waiting) {
    swRegistration.waiting.postMessage({ type: 'SKIP_WAITING' });
    console.log('[SW Registration] Skip waiting requested');
  }
}

/**
 * Cache health data for offline access
 */
export function cacheHealthData(data: any): void {
  if (swRegistration?.active) {
    swRegistration.active.postMessage({
      type: 'CACHE_HEALTH_DATA',
      payload: data,
    });
    console.log('[SW Registration] Health data cached');
  }
}

// ============================================================================
// Offline Queue
// ============================================================================

const offlineQueue: Array<{
  url: string;
  method: string;
  headers: Record<string, string>;
  body: any;
}> = [];

/**
 * Queue an API request for when back online
 */
export function queueOfflineRequest(
  url: string,
  method: string,
  body: any,
  headers: Record<string, string> = {}
): void {
  offlineQueue.push({ url, method, headers, body });

  // Store in localStorage as backup
  localStorage.setItem('offline_queue', JSON.stringify(offlineQueue));

  console.log('[SW Registration] Request queued for offline sync');
}

/**
 * Trigger background sync when online
 */
export async function triggerSync(tag: string = 'sync-health-data'): Promise<void> {
  if (!swRegistration) return;

  try {
    if ('sync' in swRegistration) {
      await (swRegistration as any).sync.register(tag);
      console.log('[SW Registration] Background sync registered:', tag);
    }
  } catch (error) {
    console.error('[SW Registration] Failed to register sync:', error);
  }
}

// ============================================================================
// Status Helpers
// ============================================================================

/**
 * Check if app is online
 */
export function getOnlineStatus(): boolean {
  return isOnline;
}

/**
 * Get current service worker registration
 */
export function getRegistration(): ServiceWorkerRegistration | null {
  return swRegistration;
}

/**
 * Check if service worker is active
 */
export function isServiceWorkerActive(): boolean {
  return swRegistration?.active !== null;
}

// ============================================================================
// Export
// ============================================================================

export const serviceWorker = {
  register: registerServiceWorker,
  unregister: unregisterServiceWorker,
  clearCaches,
  skipWaiting,
  cacheHealthData,
  queueOfflineRequest,
  triggerSync,
  isOnline: getOnlineStatus,
  getRegistration,
  isActive: isServiceWorkerActive,
  isSupported: isServiceWorkerSupported,
};

export default serviceWorker;
