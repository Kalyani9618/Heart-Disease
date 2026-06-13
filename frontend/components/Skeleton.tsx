/**
 * Skeleton Components - Loading placeholders for improved UX
 *
 * Provides shimmer/skeleton effects for various content types:
 * - Text
 * - Cards
 * - Lists
 * - Charts
 * - Chat messages
 * - Forms
 *
 * Usage:
 * ```tsx
 * import { Skeleton, CardSkeleton, ChatMessageSkeleton } from './Skeleton';
 *
 * // While loading
 * if (loading) return <CardSkeleton />;
 * ```
 */

import React from 'react';

// ============================================================================
// Base Skeleton with Shimmer Effect
// ============================================================================

interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
  animate?: boolean;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  width,
  height,
  rounded = 'md',
  animate = true,
}) => {
  const roundedClasses = {
    none: 'rounded-none',
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    full: 'rounded-full',
  };

  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  return (
    <div
      className={`
        bg-slate-200 dark:bg-slate-700
        ${roundedClasses[rounded]}
        ${animate ? 'animate-pulse' : ''}
        ${className}
      `}
      style={style}
    />
  );
};

// ============================================================================
// Text Skeleton
// ============================================================================

interface TextSkeletonProps {
  lines?: number;
  className?: string;
}

export const TextSkeleton: React.FC<TextSkeletonProps> = ({
  lines = 3,
  className = '',
}) => {
  const widths = ['100%', '90%', '80%', '95%', '75%', '85%'];

  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          height={16}
          width={widths[i % widths.length]}
          rounded="sm"
        />
      ))}
    </div>
  );
};

// ============================================================================
// Avatar Skeleton
// ============================================================================

interface AvatarSkeletonProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export const AvatarSkeleton: React.FC<AvatarSkeletonProps> = ({
  size = 'md',
}) => {
  const sizes = {
    sm: 32,
    md: 40,
    lg: 48,
    xl: 64,
  };

  return (
    <Skeleton
      width={sizes[size]}
      height={sizes[size]}
      rounded="full"
    />
  );
};

// ============================================================================
// Card Skeleton
// ============================================================================

interface CardSkeletonProps {
  hasImage?: boolean;
  imageHeight?: number;
  lines?: number;
  className?: string;
}

export const CardSkeleton: React.FC<CardSkeletonProps> = ({
  hasImage = false,
  imageHeight = 160,
  lines = 3,
  className = '',
}) => {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl shadow-sm overflow-hidden ${className}`}>
      {hasImage && (
        <Skeleton height={imageHeight} rounded="none" className="w-full" />
      )}
      <div className="p-4 space-y-3">
        <Skeleton height={20} width="60%" rounded="sm" />
        <TextSkeleton lines={lines} />
      </div>
    </div>
  );
};

// ============================================================================
// List Item Skeleton
// ============================================================================

interface ListItemSkeletonProps {
  hasAvatar?: boolean;
  hasAction?: boolean;
  className?: string;
}

export const ListItemSkeleton: React.FC<ListItemSkeletonProps> = ({
  hasAvatar = true,
  hasAction = false,
  className = '',
}) => {
  return (
    <div className={`flex items-center gap-3 p-3 ${className}`}>
      {hasAvatar && <AvatarSkeleton size="md" />}
      <div className="flex-1 space-y-2">
        <Skeleton height={16} width="70%" rounded="sm" />
        <Skeleton height={12} width="50%" rounded="sm" />
      </div>
      {hasAction && <Skeleton height={32} width={32} rounded="full" />}
    </div>
  );
};

// ============================================================================
// List Skeleton
// ============================================================================

interface ListSkeletonProps {
  items?: number;
  hasAvatar?: boolean;
  hasAction?: boolean;
  className?: string;
}

export const ListSkeleton: React.FC<ListSkeletonProps> = ({
  items = 5,
  hasAvatar = true,
  hasAction = false,
  className = '',
}) => {
  return (
    <div className={`divide-y divide-slate-200 dark:divide-slate-700 ${className}`}>
      {Array.from({ length: items }).map((_, i) => (
        <ListItemSkeleton key={i} hasAvatar={hasAvatar} hasAction={hasAction} />
      ))}
    </div>
  );
};

// ============================================================================
// Chat Message Skeleton
// ============================================================================

interface ChatMessageSkeletonProps {
  isUser?: boolean;
  className?: string;
}

export const ChatMessageSkeleton: React.FC<ChatMessageSkeletonProps> = ({
  isUser = false,
  className = '',
}) => {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} ${className}`}>
      <div className={`flex gap-2 max-w-[80%] ${isUser ? 'flex-row-reverse' : ''}`}>
        {!isUser && <AvatarSkeleton size="sm" />}
        <div
          className={`
            p-3 rounded-2xl space-y-2
            ${isUser
              ? 'bg-blue-100 dark:bg-blue-900/30'
              : 'bg-slate-100 dark:bg-slate-800'}
          `}
          style={{ minWidth: '200px' }}
        >
          <Skeleton height={14} width="100%" rounded="sm" />
          <Skeleton height={14} width="85%" rounded="sm" />
          <Skeleton height={14} width="60%" rounded="sm" />
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Chat List Skeleton
// ============================================================================

interface ChatListSkeletonProps {
  messages?: number;
  className?: string;
}

export const ChatListSkeleton: React.FC<ChatListSkeletonProps> = ({
  messages = 5,
  className = '',
}) => {
  return (
    <div className={`space-y-4 p-4 ${className}`}>
      {Array.from({ length: messages }).map((_, i) => (
        <ChatMessageSkeleton key={i} isUser={i % 2 === 1} />
      ))}
    </div>
  );
};

// ============================================================================
// Stats Card Skeleton
// ============================================================================

interface StatsCardSkeletonProps {
  className?: string;
}

export const StatsCardSkeleton: React.FC<StatsCardSkeletonProps> = ({
  className = '',
}) => {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <Skeleton height={14} width={80} rounded="sm" />
        <Skeleton height={24} width={24} rounded="full" />
      </div>
      <Skeleton height={32} width={100} rounded="sm" className="mb-2" />
      <Skeleton height={12} width={60} rounded="sm" />
    </div>
  );
};

// ============================================================================
// Chart Skeleton
// ============================================================================

interface ChartSkeletonProps {
  type?: 'line' | 'bar' | 'pie';
  height?: number;
  className?: string;
}

export const ChartSkeleton: React.FC<ChartSkeletonProps> = ({
  type = 'line',
  height = 200,
  className = '',
}) => {
  if (type === 'pie') {
    return (
      <div className={`flex items-center justify-center ${className}`} style={{ height }}>
        <Skeleton width={height * 0.8} height={height * 0.8} rounded="full" />
      </div>
    );
  }

  // Line or bar chart
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl p-4 ${className}`}>
      {/* Title */}
      <Skeleton height={18} width={120} rounded="sm" className="mb-4" />

      {/* Chart area */}
      <div className="flex items-end gap-2" style={{ height }}>
        {type === 'bar' ? (
          // Bar chart
          Array.from({ length: 7 }).map((_, i) => (
            <div key={i} className="flex-1 flex flex-col justify-end">
              <Skeleton
                height={`${30 + Math.random() * 70}%`}
                rounded="sm"
                className="w-full"
              />
            </div>
          ))
        ) : (
          // Line chart placeholder
          <div className="flex-1 relative">
            <svg
              className="w-full h-full text-slate-200 dark:text-slate-700"
              viewBox="0 0 100 50"
              preserveAspectRatio="none"
            >
              <path
                d="M0,40 L15,35 L30,38 L45,25 L60,30 L75,20 L90,22 L100,15"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="animate-pulse"
              />
            </svg>
          </div>
        )}
      </div>

      {/* X-axis labels */}
      <div className="flex justify-between mt-2">
        {Array.from({ length: 7 }).map((_, i) => (
          <Skeleton key={i} height={10} width={30} rounded="sm" />
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Vital Card Skeleton
// ============================================================================

export const VitalCardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm ${className}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Skeleton width={40} height={40} rounded="lg" />
          <div className="space-y-1">
            <Skeleton height={14} width={80} rounded="sm" />
            <Skeleton height={10} width={60} rounded="sm" />
          </div>
        </div>
        <Skeleton height={24} width={50} rounded="full" />
      </div>
      <div className="flex items-baseline gap-1">
        <Skeleton height={36} width={80} rounded="sm" />
        <Skeleton height={14} width={40} rounded="sm" />
      </div>
    </div>
  );
};

// ============================================================================
// Appointment Card Skeleton
// ============================================================================

export const AppointmentCardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm ${className}`}>
      <div className="flex gap-4">
        {/* Date column */}
        <div className="flex flex-col items-center p-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
          <Skeleton height={12} width={30} rounded="sm" />
          <Skeleton height={24} width={24} rounded="sm" className="my-1" />
          <Skeleton height={10} width={40} rounded="sm" />
        </div>

        {/* Details */}
        <div className="flex-1 space-y-2">
          <Skeleton height={18} width="70%" rounded="sm" />
          <div className="flex items-center gap-2">
            <Skeleton height={14} width={14} rounded="full" />
            <Skeleton height={14} width={100} rounded="sm" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton height={14} width={14} rounded="full" />
            <Skeleton height={14} width={80} rounded="sm" />
          </div>
        </div>

        {/* Actions */}
        <Skeleton height={32} width={32} rounded="full" />
      </div>
    </div>
  );
};

// ============================================================================
// Medication Card Skeleton
// ============================================================================

export const MedicationCardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm ${className}`}>
      <div className="flex items-start gap-3">
        <Skeleton width={48} height={48} rounded="lg" />
        <div className="flex-1 space-y-2">
          <Skeleton height={18} width="60%" rounded="sm" />
          <Skeleton height={14} width="40%" rounded="sm" />
          <div className="flex gap-2 mt-2">
            <Skeleton height={24} width={60} rounded="full" />
            <Skeleton height={24} width={80} rounded="full" />
          </div>
        </div>
        <Skeleton height={24} width={24} rounded="sm" />
      </div>
    </div>
  );
};

// ============================================================================
// Form Skeleton
// ============================================================================

interface FormSkeletonProps {
  fields?: number;
  hasSubmit?: boolean;
  className?: string;
}

export const FormSkeleton: React.FC<FormSkeletonProps> = ({
  fields = 4,
  hasSubmit = true,
  className = '',
}) => {
  return (
    <div className={`space-y-4 ${className}`}>
      {Array.from({ length: fields }).map((_, i) => (
        <div key={i} className="space-y-1">
          <Skeleton height={14} width={100} rounded="sm" />
          <Skeleton height={40} width="100%" rounded="md" />
        </div>
      ))}
      {hasSubmit && (
        <Skeleton height={44} width="100%" rounded="lg" className="mt-6" />
      )}
    </div>
  );
};

// ============================================================================
// Page Skeleton (Full page loading state)
// ============================================================================

interface PageSkeletonProps {
  type?: 'dashboard' | 'list' | 'detail' | 'chat';
  className?: string;
}

export const PageSkeleton: React.FC<PageSkeletonProps> = ({
  type = 'dashboard',
  className = '',
}) => {
  if (type === 'chat') {
    return (
      <div className={`h-full flex flex-col ${className}`}>
        {/* Header */}
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <AvatarSkeleton size="md" />
            <div className="space-y-1">
              <Skeleton height={18} width={120} rounded="sm" />
              <Skeleton height={12} width={80} rounded="sm" />
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-hidden">
          <ChatListSkeleton messages={6} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-700">
          <Skeleton height={48} width="100%" rounded="full" />
        </div>
      </div>
    );
  }

  if (type === 'list') {
    return (
      <div className={`p-4 space-y-4 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <Skeleton height={28} width={150} rounded="sm" />
          <Skeleton height={36} width={100} rounded="lg" />
        </div>

        {/* Search/Filter */}
        <Skeleton height={44} width="100%" rounded="lg" />

        {/* List */}
        <ListSkeleton items={8} hasAction />
      </div>
    );
  }

  if (type === 'detail') {
    return (
      <div className={`p-4 space-y-6 ${className}`}>
        {/* Header */}
        <div className="flex items-center gap-4">
          <Skeleton width={80} height={80} rounded="lg" />
          <div className="flex-1 space-y-2">
            <Skeleton height={24} width="60%" rounded="sm" />
            <Skeleton height={16} width="40%" rounded="sm" />
          </div>
        </div>

        {/* Content */}
        <CardSkeleton lines={5} />

        {/* Additional sections */}
        <div className="grid grid-cols-2 gap-4">
          <StatsCardSkeleton />
          <StatsCardSkeleton />
        </div>
      </div>
    );
  }

  // Dashboard (default)
  return (
    <div className={`p-4 space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <Skeleton height={28} width={200} rounded="sm" />
          <Skeleton height={14} width={150} rounded="sm" />
        </div>
        <AvatarSkeleton size="lg" />
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatsCardSkeleton />
        <StatsCardSkeleton />
        <StatsCardSkeleton />
        <StatsCardSkeleton />
      </div>

      {/* Chart */}
      <ChartSkeleton type="line" height={200} />

      {/* Recent items */}
      <div>
        <Skeleton height={20} width={150} rounded="sm" className="mb-3" />
        <ListSkeleton items={3} />
      </div>
    </div>
  );
};

// ============================================================================
// Shimmer Effect (alternative animation)
// ============================================================================

export const ShimmerEffect: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div
      className={`
        relative overflow-hidden
        before:absolute before:inset-0
        before:-translate-x-full
        before:animate-[shimmer_2s_infinite]
        before:bg-gradient-to-r
        before:from-transparent before:via-white/20 before:to-transparent
        ${className}
      `}
    />
  );
};

// Add shimmer animation to tailwind config:
// animation: { shimmer: 'shimmer 2s infinite' }
// keyframes: { shimmer: { '100%': { transform: 'translateX(100%)' } } }

// ============================================================================
// Dashboard Skeleton - Matches exact Dashboard layout
// ============================================================================

export const DashboardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`p-4 space-y-6 pb-24 ${className}`}>
      {/* Header - Greeting + Profile */}
      <div className="flex justify-between items-center py-2">
        <div className="space-y-1">
          <Skeleton height={14} width={100} rounded="sm" />
          <Skeleton height={28} width={180} rounded="sm" />
        </div>
        <div className="flex items-center gap-3">
          <Skeleton height={40} width={40} rounded="full" />
          <Skeleton height={40} width={40} rounded="full" />
        </div>
      </div>

      {/* AI Insight Card */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-5 text-white shadow-lg">
        <div className="flex items-start gap-3">
          <Skeleton height={48} width={48} rounded="full" className="bg-white/20" />
          <div className="flex-1 space-y-2">
            <Skeleton height={18} width={120} rounded="sm" className="bg-white/20" />
            <Skeleton height={14} width="90%" rounded="sm" className="bg-white/20" />
            <Skeleton height={14} width="70%" rounded="sm" className="bg-white/20" />
          </div>
        </div>
      </div>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Heart Rate Card */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <Skeleton height={32} width={32} rounded="lg" />
            <Skeleton height={20} width={60} rounded="full" />
          </div>
          <Skeleton height={36} width={60} rounded="sm" className="mb-1" />
          <Skeleton height={12} width={80} rounded="sm" />
        </div>

        {/* Steps Card */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <Skeleton height={32} width={32} rounded="lg" />
            <Skeleton height={20} width={50} rounded="full" />
          </div>
          <Skeleton height={36} width={80} rounded="sm" className="mb-1" />
          <Skeleton height={12} width={100} rounded="sm" />
        </div>
      </div>

      {/* Heart Rate Chart */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <Skeleton height={18} width={140} rounded="sm" />
          <Skeleton height={28} width={80} rounded="lg" />
        </div>
        <ChartSkeleton type="line" height={180} className="bg-transparent shadow-none p-0" />
      </div>

      {/* Next Medication */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <Skeleton height={18} width={140} rounded="sm" />
          <Skeleton height={24} width={60} rounded="lg" />
        </div>
        <div className="flex items-center gap-3">
          <Skeleton height={48} width={48} rounded="lg" />
          <div className="flex-1 space-y-2">
            <Skeleton height={16} width={120} rounded="sm" />
            <Skeleton height={12} width={80} rounded="sm" />
          </div>
          <Skeleton height={40} width={40} rounded="full" />
        </div>
      </div>

      {/* Next Appointment */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <Skeleton height={18} width={160} rounded="sm" />
          <Skeleton height={24} width={80} rounded="lg" />
        </div>
        <div className="flex items-center gap-4">
          <div className="flex flex-col items-center p-3 bg-slate-100 dark:bg-slate-700 rounded-xl">
            <Skeleton height={10} width={30} rounded="sm" />
            <Skeleton height={28} width={28} rounded="sm" className="my-1" />
          </div>
          <div className="flex-1 space-y-2">
            <Skeleton height={16} width={140} rounded="sm" />
            <Skeleton height={12} width={100} rounded="sm" />
            <Skeleton height={12} width={80} rounded="sm" />
          </div>
          <Skeleton height={44} width={44} rounded="full" />
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Chat Screen Skeleton
// ============================================================================

export const ChatScreenSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Skeleton height={36} width={36} rounded="full" />
            <div className="space-y-1">
              <Skeleton height={16} width={120} rounded="sm" />
              <Skeleton height={10} width={80} rounded="sm" />
            </div>
          </div>
          <div className="flex gap-2">
            <Skeleton height={32} width={32} rounded="full" />
            <Skeleton height={32} width={32} rounded="full" />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-hidden p-4 space-y-4">
        {/* AI Welcome Message */}
        <div className="flex gap-2">
          <Skeleton height={32} width={32} rounded="full" />
          <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl rounded-tl-sm p-4 max-w-[80%] space-y-2">
            <Skeleton height={14} width="100%" rounded="sm" />
            <Skeleton height={14} width="90%" rounded="sm" />
            <Skeleton height={14} width="60%" rounded="sm" />
          </div>
        </div>

        {/* User Message */}
        <div className="flex gap-2 justify-end">
          <div className="bg-blue-500 rounded-2xl rounded-tr-sm p-4 max-w-[70%] space-y-2">
            <Skeleton height={14} width={150} rounded="sm" className="bg-blue-400" />
          </div>
        </div>

        {/* AI Response */}
        <div className="flex gap-2">
          <Skeleton height={32} width={32} rounded="full" />
          <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl rounded-tl-sm p-4 max-w-[80%] space-y-2">
            <Skeleton height={14} width="100%" rounded="sm" />
            <Skeleton height={14} width="85%" rounded="sm" />
            <Skeleton height={14} width="75%" rounded="sm" />
            <Skeleton height={14} width="50%" rounded="sm" />
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
        <div className="flex items-center gap-2">
          <Skeleton height={44} width={44} rounded="full" />
          <Skeleton height={44} className="flex-1" rounded="full" />
          <Skeleton height={44} width={44} rounded="full" />
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Medication Screen Skeleton
// ============================================================================

export const MedicationScreenSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`pb-24 ${className}`}>
      {/* Header */}
      <div className="p-4 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center">
          <Skeleton height={40} width={40} rounded="full" />
          <Skeleton height={20} width={150} rounded="sm" className="mx-auto" />
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Safety Check Card */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-5">
          <div className="flex items-start gap-4">
            <Skeleton height={48} width={48} rounded="full" className="bg-white/20" />
            <div className="flex-1 space-y-2">
              <Skeleton height={18} width={100} rounded="sm" className="bg-white/20" />
              <Skeleton height={14} width="80%" rounded="sm" className="bg-white/20" />
              <Skeleton height={36} width={140} rounded="lg" className="bg-white/20 mt-3" />
            </div>
          </div>
        </div>

        {/* Medication Cards */}
        <div className="space-y-3">
          <MedicationCardSkeleton />
          <MedicationCardSkeleton />
          <MedicationCardSkeleton />
        </div>
      </div>

      {/* FAB */}
      <div className="fixed bottom-24 right-4">
        <Skeleton height={56} width={56} rounded="full" />
      </div>
    </div>
  );
};

// ============================================================================
// Profile Screen Skeleton
// ============================================================================

export const ProfileScreenSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`pb-24 ${className}`}>
      {/* Header with Avatar */}
      <div className="bg-gradient-to-b from-blue-600 to-blue-700 pt-8 pb-16 px-4 text-center">
        <Skeleton height={96} width={96} rounded="full" className="mx-auto mb-3 bg-white/20" />
        <Skeleton height={24} width={150} rounded="sm" className="mx-auto mb-1 bg-white/20" />
        <Skeleton height={14} width={180} rounded="sm" className="mx-auto bg-white/20" />
      </div>

      {/* Stats */}
      <div className="px-4 -mt-8">
        <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-lg grid grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="text-center space-y-1">
              <Skeleton height={28} width={50} rounded="sm" className="mx-auto" />
              <Skeleton height={12} width={60} rounded="sm" className="mx-auto" />
            </div>
          ))}
        </div>
      </div>

      {/* Menu Items */}
      <div className="p-4 space-y-3">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="bg-white dark:bg-slate-800 rounded-xl p-4 flex items-center gap-3">
            <Skeleton height={40} width={40} rounded="lg" />
            <div className="flex-1 space-y-1">
              <Skeleton height={16} width={120} rounded="sm" />
              <Skeleton height={12} width={80} rounded="sm" />
            </div>
            <Skeleton height={20} width={20} rounded="sm" />
          </div>
        ))}
      </div>
    </div>
  );
};

export default Skeleton;
