
import React, { useState, useEffect, lazy, Suspense, useRef } from 'react';
import { HashRouter, Routes, Route, useLocation, Navigate, useNavigate } from 'react-router-dom';

// ============================================================================
// Capacitor Platform Initialization
// ============================================================================
const initCapacitorPlugins = async () => {
  const { Capacitor } = await import('@capacitor/core');
  const isNative = Capacitor.getPlatform() !== 'web';

  try {
    // Hide splash screen after app loads
    const { SplashScreen } = await import('@capacitor/splash-screen');
    await SplashScreen.hide({ fadeOutDuration: 300 });
  } catch { /* not on native */ }

  try {
    // Configure status bar (initial setup, will be updated by theme toggle)
    const { StatusBar, Style } = await import('@capacitor/status-bar');
    await StatusBar.setStyle({ style: Style.Dark });
    await StatusBar.setBackgroundColor({ color: '#111111' });
  } catch { /* not on native */ }

  try {
    // Keyboard plugin is native-only; skip on web to avoid unhandled plugin errors.
    if (isNative && Capacitor.isPluginAvailable('Keyboard')) {
      const { Keyboard } = await import('@capacitor/keyboard');
      await Keyboard.addListener('keyboardWillShow', () => {
        document.body.classList.add('keyboard-visible');
      });
      await Keyboard.addListener('keyboardWillHide', () => {
        document.body.classList.remove('keyboard-visible');
      });
    }
  } catch { /* not on native */ }

  try {
    // Create notification channels on startup so they exist before any notification is sent
    const { createNotificationChannels } = await import('./services/nativeNotificationService');
    await createNotificationChannels();
  } catch { /* not on native or service unavailable */ }
};

// Eagerly loaded screens (critical path - login/signup)
import LoginScreen from './screens/LoginScreen';
import SignUpScreen from './screens/SignUpScreen';

// Lazily loaded screens for code splitting
const DashboardScreen = lazy(() => import('./screens/DashboardScreen'));
const AssessmentScreen = lazy(() => import('./screens/AssessmentScreen'));
const AppointmentScreen = lazy(() => import('./screens/AppointmentScreen'));
const ChatScreen = lazy(() => import('./screens/ChatScreen'));
const SettingsScreen = lazy(() => import('./screens/SettingsScreen'));
const ProfileScreen = lazy(() => import('./screens/ProfileScreen'));
const MedicationScreen = lazy(() => import('./screens/MedicationScreen'));
const DocumentScanner = lazy(() => import('./screens/DocumentScanner'));
const VisionAnalysis = lazy(() => import('./screens/VisionAnalysis'));
const NotificationScreen = lazy(() => import('./screens/NotificationScreen'));
const SmartWatchScreen = lazy(() => import('./screens/SmartWatchScreen'));
const CalendarScreen = lazy(() => import('./screens/CalendarScreen'));
const ComplianceScreen = lazy(() => import('./screens/ComplianceScreen'));
const WeeklySummaryScreen = lazy(() => import('./screens/WeeklySummaryScreen'));
const PatientSummaryScreen = lazy(() => import('./screens/PatientSummaryScreen'));
const ConsentScreen = lazy(() => import('./screens/ConsentScreen'));
const DocumentScreen = lazy(() => import('./screens/DocumentScreen'));
const AgentSettingsScreen = lazy(() => import('./screens/AgentSettingsScreen'));
const SavedResultsScreen = lazy(() => import('./screens/SavedResultsScreen'));

import BottomNav from './components/BottomNav';
import ErrorBoundary from './components/ErrorBoundary';
import { LanguageProvider } from './contexts/LanguageContext';
import { ProviderProvider } from './contexts/ProviderContext';
import { PageSkeleton, ChatListSkeleton } from './components/Skeleton';
import { ConfirmDialogProvider } from './components/ConfirmDialog';
import { ToastProvider } from './components/Toast';
import { AuthProvider } from './hooks/useAuth';

// Loading fallback components for different screen types
const DashboardFallback = () => <PageSkeleton type="dashboard" className="h-screen bg-background-light dark:bg-background-dark" />;
const ChatFallback = () => <PageSkeleton type="chat" className="h-screen bg-background-light dark:bg-background-dark" />;
const ListFallback = () => <PageSkeleton type="list" className="h-screen bg-background-light dark:bg-background-dark" />;
const DetailFallback = () => <PageSkeleton type="detail" className="h-screen bg-background-light dark:bg-background-dark" />;

const AppContent: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Swipe Back Logic
  const touchStartRef = useRef(0);
  const touchEndRef = useRef(0);

  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartRef.current = e.targetTouches[0].clientX;
    touchEndRef.current = 0;
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    touchEndRef.current = e.targetTouches[0].clientX;
  };

  const handleTouchEnd = () => {
    if (!touchStartRef.current || !touchEndRef.current) return;
    const distance = touchEndRef.current - touchStartRef.current;
    const isLeftEdge = touchStartRef.current < 50;

    if (isLeftEdge && distance > 100) {
      navigate(-1);
    }
    touchStartRef.current = 0;
    touchEndRef.current = 0;
  };

  // Routes where the bottom navigation should be visible
  const showBottomNav = [
    '/dashboard',
    '/profile',
    '/settings',
    '/appointment',
    '/assessment',
    '/medications',
    '/documents',
    '/notifications',
    '/smartwatch',
    '/calendar',
    '/weekly-summary',
    '/saved-results',
  ].includes(location.pathname);

  // Update HTML class for dark mode and native status bar
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }

    // Update native status bar to match theme
    (async () => {
      try {
        const { StatusBar, Style } = await import('@capacitor/status-bar');
        if (isDarkMode) {
          await StatusBar.setStyle({ style: Style.Dark });
          await StatusBar.setBackgroundColor({ color: '#111111' });
        } else {
          await StatusBar.setStyle({ style: Style.Light });
          await StatusBar.setBackgroundColor({ color: '#f8fafc' });
        }
      } catch { /* not on native */ }
    })();

    console.log('AppContent mounted, isDarkMode:', isDarkMode, 'location:', location.pathname);
  }, [isDarkMode, location.pathname]);

  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  return (
    <div
      className={`flex flex-col max-w-md mx-auto relative bg-background-light dark:bg-background-dark shadow-2xl overflow-hidden ${location.pathname.startsWith('/chat') ? 'h-[100dvh]' : 'min-h-screen'}`}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      <ErrorBoundary>
        <div className={`flex-1 min-h-0 ${location.pathname.startsWith('/chat') || ['/login', '/signup'].includes(location.pathname) ? 'overflow-hidden' : 'overflow-y-auto no-scrollbar pb-20'}`}>
          <Routes>
            <Route path="/" element={<Navigate to="/login" replace />} />
            <Route path="/login" element={<LoginScreen />} />
            <Route path="/signup" element={<SignUpScreen />} />

            <Route path="/dashboard" element={
              <Suspense fallback={<DashboardFallback />}>
                <DashboardScreen />
              </Suspense>
            } />
            <Route path="/assessment" element={
              <Suspense fallback={<ListFallback />}>
                <AssessmentScreen />
              </Suspense>
            } />
            <Route path="/appointment" element={
              <Suspense fallback={<ListFallback />}>
                <AppointmentScreen />
              </Suspense>
            } />
            <Route path="/medications" element={
              <Suspense fallback={<ListFallback />}>
                <MedicationScreen />
              </Suspense>
            } />
            <Route path="/chat" element={
              <Suspense fallback={<ChatFallback />}>
                <ChatScreen />
              </Suspense>
            } />
            <Route path="/settings" element={
              <Suspense fallback={<ListFallback />}>
                <SettingsScreen isDark={isDarkMode} toggleTheme={toggleTheme} />
              </Suspense>
            } />
            <Route path="/profile" element={
              <Suspense fallback={<DetailFallback />}>
                <ProfileScreen />
              </Suspense>
            } />
            <Route path="/scan-document" element={
              <Suspense fallback={<ListFallback />}>
                <DocumentScanner />
              </Suspense>
            } />
            <Route path="/vision" element={
              <Suspense fallback={<ListFallback />}>
                <VisionAnalysis />
              </Suspense>
            } />
            <Route path="/notifications" element={
              <Suspense fallback={<ListFallback />}>
                <NotificationScreen />
              </Suspense>
            } />
            <Route path="/smartwatch" element={
              <Suspense fallback={<ListFallback />}>
                <SmartWatchScreen />
              </Suspense>
            } />
            <Route path="/calendar" element={
              <Suspense fallback={<ListFallback />}>
                <CalendarScreen />
              </Suspense>
            } />
            <Route path="/compliance" element={
              <Suspense fallback={<ListFallback />}>
                <ComplianceScreen />
              </Suspense>
            } />
            <Route path="/weekly-summary" element={
              <Suspense fallback={<ListFallback />}>
                <WeeklySummaryScreen />
              </Suspense>
            } />
            <Route path="/patient-summary" element={
              <Suspense fallback={<ListFallback />}>
                <PatientSummaryScreen />
              </Suspense>
            } />
            <Route path="/consent" element={
              <Suspense fallback={<ListFallback />}>
                <ConsentScreen />
              </Suspense>
            } />
            <Route path="/documents" element={
              <Suspense fallback={<ListFallback />}>
                <DocumentScreen />
              </Suspense>
            } />
            <Route path="/agent-settings" element={
              <Suspense fallback={<ListFallback />}>
                <AgentSettingsScreen />
              </Suspense>
            } />
            <Route path="/saved-results" element={
              <Suspense fallback={<ListFallback />}>
                <SavedResultsScreen />
              </Suspense>
            } />
          </Routes>
        </div>

        {showBottomNav && <BottomNav />}
      </ErrorBoundary>
    </div>
  );
};



const App: React.FC = () => {
  useEffect(() => {
    initCapacitorPlugins();
  }, []);

  return (
    <AuthProvider>
      <LanguageProvider>
        <ProviderProvider>
          <ConfirmDialogProvider>
            <ToastProvider>
              <HashRouter>
                <AppContent />
              </HashRouter>
            </ToastProvider>
          </ConfirmDialogProvider>
        </ProviderProvider>
      </LanguageProvider>
    </AuthProvider>
  );
};

export default App;
