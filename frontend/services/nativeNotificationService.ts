/**
 * Native Notification Service
 *
 * Bridges Capacitor LocalNotifications & PushNotifications with
 * SMS, WhatsApp, and Email delivery channels for the Settings screens.
 *
 * Handles:
 * - Push notification permission requests on Android 13+
 * - Local notification scheduling (medication reminders, weekly summaries)
 * - SMS sending via native intent
 * - WhatsApp deep-link sharing
 * - Email intent launching
 * - Notification channel management
 */

import { Capacitor } from '@capacitor/core';

// ============================================================================
// Types
// ============================================================================

export interface NotificationScheduleOptions {
  id: number;
  title: string;
  body: string;
  scheduledAt: Date;
  repeats?: boolean;
  repeatInterval?: 'day' | 'week' | 'month';
  channelId?: string;
  extra?: Record<string, unknown>;
}

export interface DeliveryResult {
  channel: 'push' | 'email' | 'whatsapp' | 'sms';
  success: boolean;
  error?: string;
}

// ============================================================================
// Platform Detection
// ============================================================================

function isNativePlatform(): boolean {
  return Capacitor.isNativePlatform();
}

function getPlatform(): 'android' | 'ios' | 'web' {
  return Capacitor.getPlatform() as 'android' | 'ios' | 'web';
}

// ============================================================================
// Dynamic imports for Capacitor plugins (they may not be installed)
// ============================================================================

async function getLocalNotifications() {
  try {
    const { LocalNotifications } = await import('@capacitor/local-notifications');
    return LocalNotifications;
  } catch {
    console.warn('[NativeNotif] @capacitor/local-notifications not available');
    return null;
  }
}

async function getPushNotifications() {
  try {
    const { PushNotifications } = await import('@capacitor/push-notifications');
    return PushNotifications;
  } catch {
    console.warn('[NativeNotif] @capacitor/push-notifications not available');
    return null;
  }
}

// ============================================================================
// Permission Management
// ============================================================================

/**
 * Request notification permissions.
 * On Android 13+, this triggers the POST_NOTIFICATIONS runtime permission dialog.
 * Falls back to Web Notification API on web platform.
 */
export async function requestNotificationPermission(): Promise<boolean> {
  if (isNativePlatform()) {
    // Try local notifications first
    const LocalNotifications = await getLocalNotifications();
    if (LocalNotifications) {
      const result = await LocalNotifications.requestPermissions();
      if (result.display === 'granted') {
        return true;
      }
    }

    // Also register for push notifications
    const PushNotif = await getPushNotifications();
    if (PushNotif) {
      const pushResult = await PushNotif.requestPermissions();
      if (pushResult.receive === 'granted') {
        await PushNotif.register();
        return true;
      }
    }

    return false;
  }

  // Web fallback
  if ('Notification' in window) {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }

  return false;
}

/**
 * Check current notification permission status
 */
export async function checkNotificationPermission(): Promise<'granted' | 'denied' | 'prompt'> {
  if (isNativePlatform()) {
    const LocalNotifications = await getLocalNotifications();
    if (LocalNotifications) {
      const result = await LocalNotifications.checkPermissions();
      return result.display as 'granted' | 'denied' | 'prompt';
    }
    return 'prompt';
  }

  if ('Notification' in window) {
    return Notification.permission === 'default' ? 'prompt' : Notification.permission as 'granted' | 'denied';
  }
  return 'denied';
}

// ============================================================================
// Local Notification Scheduling
// ============================================================================

/**
 * Schedule a local notification (medication reminder, weekly summary, etc.)
 */
export async function scheduleLocalNotification(options: NotificationScheduleOptions): Promise<boolean> {
  if (isNativePlatform()) {
    const LocalNotifications = await getLocalNotifications();
    if (LocalNotifications) {
      try {
        await LocalNotifications.schedule({
          notifications: [
            {
              id: options.id,
              title: options.title,
              body: options.body,
              schedule: {
                at: options.scheduledAt,
                repeats: options.repeats || false,
                every: options.repeatInterval as any,
                allowWhileIdle: true,
              },
              channelId: options.channelId || 'cardio-reminders',
              extra: options.extra,
              smallIcon: 'ic_stat_heart',
              iconColor: '#D32F2F',
            },
          ],
        });
        console.log(`[NativeNotif] Scheduled notification #${options.id}: ${options.title}`);
        return true;
      } catch (err) {
        console.error('[NativeNotif] Failed to schedule:', err);
        return false;
      }
    }
  }

  // Web fallback: use setTimeout for demo / simple cases
  const delay = options.scheduledAt.getTime() - Date.now();
  if (delay > 0 && 'Notification' in window && Notification.permission === 'granted') {
    setTimeout(() => {
      new Notification(options.title, { body: options.body, icon: '/icons/icon-192.png' });
    }, delay);
    return true;
  }

  return false;
}

/**
 * Cancel a scheduled notification by ID
 */
export async function cancelLocalNotification(id: number): Promise<void> {
  if (isNativePlatform()) {
    const LocalNotifications = await getLocalNotifications();
    if (LocalNotifications) {
      await LocalNotifications.cancel({ notifications: [{ id }] });
    }
  }
}

/**
 * Cancel all pending notifications
 */
export async function cancelAllNotifications(): Promise<void> {
  if (isNativePlatform()) {
    const LocalNotifications = await getLocalNotifications();
    if (LocalNotifications) {
      const pending = await LocalNotifications.getPending();
      if (pending.notifications.length > 0) {
        await LocalNotifications.cancel({
          notifications: pending.notifications.map(n => ({ id: n.id })),
        });
      }
    }
  }
}

// ============================================================================
// Notification Channels (Android)
// ============================================================================

/**
 * Create notification channels for Android (medication reminders, health alerts, etc.)
 */
export async function createNotificationChannels(): Promise<void> {
  if (getPlatform() !== 'android') return;

  const LocalNotifications = await getLocalNotifications();
  if (!LocalNotifications) return;

  try {
    await LocalNotifications.createChannel({
      id: 'cardio-reminders',
      name: 'Health Reminders',
      description: 'Medication and appointment reminders',
      importance: 5, // Max importance
      visibility: 1,
      vibration: true,
      sound: 'notification.wav',
    });

    await LocalNotifications.createChannel({
      id: 'cardio-insights',
      name: 'AI Health Insights',
      description: 'Daily AI-powered health insights',
      importance: 3,
      visibility: 1,
      vibration: false,
    });

    await LocalNotifications.createChannel({
      id: 'cardio-weekly',
      name: 'Weekly Summary',
      description: 'Weekly health recap notifications',
      importance: 3,
      visibility: 1,
      vibration: true,
    });

    console.log('[NativeNotif] Notification channels created');
  } catch (err) {
    console.error('[NativeNotif] Failed to create channels:', err);
  }
}

// ============================================================================
// External Delivery Channels (SMS, WhatsApp, Email)
// ============================================================================

/**
 * Send a message via SMS using native intent
 */
export async function sendSMS(phoneNumber: string, message: string): Promise<DeliveryResult> {
  try {
    if (isNativePlatform()) {
      // Open native SMS app with pre-filled message
      const smsUrl = `sms:${phoneNumber}?body=${encodeURIComponent(message)}`;
      window.open(smsUrl, '_system');
      return { channel: 'sms', success: true };
    }

    // Web fallback: open sms: URI
    window.open(`sms:${phoneNumber}?body=${encodeURIComponent(message)}`);
    return { channel: 'sms', success: true };
  } catch (err) {
    console.error('[NativeNotif] SMS failed:', err);
    return { channel: 'sms', success: false, error: String(err) };
  }
}

/**
 * Send a message via WhatsApp deep link
 */
export async function sendWhatsApp(phoneNumber: string, message: string): Promise<DeliveryResult> {
  try {
    // WhatsApp deep link with phone number (international format, no +)
    const cleanPhone = phoneNumber.replace(/[^0-9]/g, '');
    const whatsappUrl = `https://wa.me/${cleanPhone}?text=${encodeURIComponent(message)}`;

    if (isNativePlatform()) {
      // Use Capacitor Browser to open WhatsApp
      try {
        const { Browser } = await import('@capacitor/browser');
        await Browser.open({ url: whatsappUrl });
      } catch {
        window.open(whatsappUrl, '_system');
      }
    } else {
      window.open(whatsappUrl, '_blank');
    }

    return { channel: 'whatsapp', success: true };
  } catch (err) {
    console.error('[NativeNotif] WhatsApp failed:', err);
    return { channel: 'whatsapp', success: false, error: String(err) };
  }
}

/**
 * Open email client with pre-filled email
 */
export async function sendEmail(
  to: string,
  subject: string,
  body: string
): Promise<DeliveryResult> {
  try {
    const mailtoUrl = `mailto:${to}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

    if (isNativePlatform()) {
      try {
        const { Browser } = await import('@capacitor/browser');
        await Browser.open({ url: mailtoUrl });
      } catch {
        window.open(mailtoUrl, '_system');
      }
    } else {
      window.open(mailtoUrl);
    }

    return { channel: 'email', success: true };
  } catch (err) {
    console.error('[NativeNotif] Email failed:', err);
    return { channel: 'email', success: false, error: String(err) };
  }
}

/**
 * Deliver a message through a specific channel
 */
export async function deliverMessage(
  channel: 'push' | 'email' | 'whatsapp' | 'sms',
  options: {
    title: string;
    body: string;
    destination?: string; // phone or email
  }
): Promise<DeliveryResult> {
  switch (channel) {
    case 'push':
      const success = await scheduleLocalNotification({
        id: Date.now(),
        title: options.title,
        body: options.body,
        scheduledAt: new Date(),
      });
      return { channel: 'push', success };

    case 'sms':
      if (!options.destination) return { channel: 'sms', success: false, error: 'No phone number provided' };
      return sendSMS(options.destination, `${options.title}\n\n${options.body}`);

    case 'whatsapp':
      if (!options.destination) return { channel: 'whatsapp', success: false, error: 'No phone number provided' };
      return sendWhatsApp(options.destination, `*${options.title}*\n\n${options.body}`);

    case 'email':
      if (!options.destination) return { channel: 'email', success: false, error: 'No email address provided' };
      return sendEmail(options.destination, options.title, options.body);

    default:
      return { channel, success: false, error: 'Unknown channel' };
  }
}

// ============================================================================
// Exported Service Object
// ============================================================================

export const nativeNotificationService = {
  // Permissions
  requestNotificationPermission,
  checkNotificationPermission,

  // Local Notifications
  scheduleLocalNotification,
  cancelLocalNotification,
  cancelAllNotifications,
  createNotificationChannels,

  // External Channels
  sendSMS,
  sendWhatsApp,
  sendEmail,
  deliverMessage,

  // Utility
  isNativePlatform,
  getPlatform,
};

export default nativeNotificationService;
