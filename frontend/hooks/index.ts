/**
 * Hooks Index
 *
 * Central export for all custom React hooks
 */

// Domain Hooks
export { useDailyInsight } from './useDailyInsight';
export { useWaterTracking } from './useWaterTracking';

// Offline/PWA Hooks
export {
  useOfflineStatus,
  OfflineBanner,
  OfflineFallback,
} from './useOfflineStatus';

// Bluetooth Hook
export { useBluetooth, type UseBluetoothReturn } from './useBluetooth';
