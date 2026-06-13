/**
 * useOfflineStatus Hook
 *
 * React hook for tracking online/offline status
 * with UI feedback components
 */

import { useState, useEffect, useCallback } from 'react';
import { getOnlineStatus, serviceWorker } from '../services/serviceWorker';

// ============================================================================
// Hook
// ============================================================================

interface UseOfflineStatusReturn {
  isOnline: boolean;
  wasOffline: boolean;
  showReconnected: boolean;
  dismissReconnected: () => void;
}

export function useOfflineStatus(): UseOfflineStatusReturn {
  const [isOnline, setIsOnline] = useState(getOnlineStatus());
  const [wasOffline, setWasOffline] = useState(false);
  const [showReconnected, setShowReconnected] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      if (wasOffline) {
        setShowReconnected(true);
        // Auto-hide after 3 seconds
        setTimeout(() => setShowReconnected(false), 3000);
      }
    };

    const handleOffline = () => {
      setIsOnline(false);
      setWasOffline(true);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [wasOffline]);

  const dismissReconnected = useCallback(() => {
    setShowReconnected(false);
  }, []);

  return {
    isOnline,
    wasOffline,
    showReconnected,
    dismissReconnected,
  };
}

// ============================================================================
// Offline Banner Component
// ============================================================================

interface OfflineBannerProps {
  className?: string;
}

export function OfflineBanner({ className = '' }: OfflineBannerProps) {
  const { isOnline, showReconnected, dismissReconnected } = useOfflineStatus();

  if (isOnline && !showReconnected) {
    return null;
  }

  return (
    <div
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${className}`}
    >
      {!isOnline ? (
        <div className="bg-amber-500 text-amber-900 px-4 py-2 flex items-center justify-center gap-2 text-sm font-medium shadow-lg">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414" />
          </svg>
          <span>You're offline. Some features may be unavailable.</span>
        </div>
      ) : showReconnected ? (
        <div className="bg-green-500 text-white px-4 py-2 flex items-center justify-center gap-2 text-sm font-medium shadow-lg">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          <span>Back online!</span>
          <button
            onClick={dismissReconnected}
            className="ml-2 hover:bg-green-600 rounded p-1 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ) : null}
    </div>
  );
}

// ============================================================================
// Offline Fallback Component
// ============================================================================

interface OfflineFallbackProps {
  message?: string;
  showRetry?: boolean;
  onRetry?: () => void;
  children?: React.ReactNode;
}

export function OfflineFallback({
  message = 'This feature requires an internet connection.',
  showRetry = true,
  onRetry,
  children,
}: OfflineFallbackProps) {
  const { isOnline } = useOfflineStatus();

  if (isOnline) {
    return <>{children}</>;
  }

  return (
    <div className="flex flex-col items-center justify-center p-8 text-center">
      <div className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mb-4">
        <svg
          className="w-8 h-8 text-slate-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414"
          />
        </svg>
      </div>
      <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
        You're Offline
      </h3>
      <p className="text-slate-500 dark:text-slate-400 mb-4 max-w-sm">
        {message}
      </p>
      {showRetry && (
        <button
          onClick={onRetry || (() => window.location.reload())}
          className="px-4 py-2 bg-primary text-white rounded-lg font-medium hover:bg-primary-dark transition-colors"
        >
          Try Again
        </button>
      )}
    </div>
  );
}

// ============================================================================
// Default Export
// ============================================================================

export default useOfflineStatus;
