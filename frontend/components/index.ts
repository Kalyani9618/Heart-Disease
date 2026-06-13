/**
 * Components Index
 *
 * Central export for all reusable UI components
 */

// Loading & State Components
export {
  Skeleton,
  TextSkeleton,
  AvatarSkeleton,
  CardSkeleton,
  ListItemSkeleton,
  ListSkeleton,
  ChatMessageSkeleton,
  ChatListSkeleton,
  StatsCardSkeleton,
  ChartSkeleton,
  VitalCardSkeleton,
  AppointmentCardSkeleton,
  MedicationCardSkeleton,
  FormSkeleton,
  PageSkeleton,
  ShimmerEffect,
  DashboardSkeleton,
  ChatScreenSkeleton,
  MedicationScreenSkeleton,
  ProfileScreenSkeleton,
} from './Skeleton';

// Navigation
export { default as BottomNav } from './BottomNav';

// Error Handling
export { default as ErrorBoundary } from './ErrorBoundary';

// Markdown Rendering
export {
  MarkdownRenderer,
  ChatMessageMarkdown,
  HealthAlertMarkdown,
} from './MarkdownRenderer';

// Medical Media Rendering
export {
  MedicalMediaSection,
  MedicalImageGallery,
  MedicalVideoGrid,
  VideoCard,
  ResearchPaperCard,
  MedicalNewsCard,
  parseMedicalContent,
} from './MedicalMediaRenderer';

// Confirmation Dialog
export {
  ConfirmDialog,
  ConfirmDialogProvider,
  useConfirmDialog,
  useConfirm,
} from './ConfirmDialog';

// Toast Notifications
export { ToastProvider } from './Toast';
