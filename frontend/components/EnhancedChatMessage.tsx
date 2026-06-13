/**
 * Enhanced Chat Message Component
 * 
 * Provides rich formatting for AI assistant responses in the chat,
 * with special handling for health-related content sections,
 * structured data display, and visual hierarchy.
 */

import React, { memo, useMemo, useState } from 'react';
import { ChatMessageMarkdown } from './MarkdownRenderer';
import { MedicalMediaSection, parseMedicalContent } from './MedicalMediaRenderer';

// ============================================================================
// Types
// ============================================================================

interface Source {
  title: string;
  category?: string;
  relevance?: number;
}

interface EnhancedChatMessageProps {
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
  isError?: boolean;
  metadata?: {
    model?: string;
    processingTime?: number;
    tokens?: number;
    memoryContext?: string[];
  };
}

// ============================================================================
// Section Detection Patterns
// ============================================================================

interface DetectedSection {
  type: 'warning' | 'important' | 'recommendation' | 'medication' | 'emergency' | 'note' | 'result' | 'normal';
  icon: string;
  iconColor: string;
  bgColor: string;
  borderColor: string;
  title?: string;
  content: string;
}

const SECTION_PATTERNS: Array<{
  regex: RegExp;
  type: DetectedSection['type'];
  icon: string;
  iconColor: string;
  bgColor: string;
  borderColor: string;
}> = [
  {
    regex: /^(?:⚠️\s*|🚨\s*)?(?:\*{0,2})(?:Emergency|EMERGENCY|Urgent|URGENT|Critical|CRITICAL|Danger|DANGER)(?:\*{0,2})[:\s]/i,
    type: 'emergency',
    icon: 'emergency',
    iconColor: 'text-red-500',
    bgColor: 'bg-red-50 dark:bg-red-950/30',
    borderColor: 'border-red-200 dark:border-red-800/50',
  },
  {
    regex: /^(?:⚠️\s*|⚡\s*)?(?:\*{0,2})(?:Warning|WARNING|Caution|CAUTION|Alert|ALERT)(?:\*{0,2})[:\s]/i,
    type: 'warning',
    icon: 'warning',
    iconColor: 'text-amber-500',
    bgColor: 'bg-amber-50 dark:bg-amber-950/30',
    borderColor: 'border-amber-200 dark:border-amber-800/50',
  },
  {
    regex: /^(?:💊\s*|💉\s*)?(?:\*{0,2})(?:Medication|MEDICATION|Drug|Prescription|Dosage|Treatment Plan)(?:\*{0,2})[:\s]/i,
    type: 'medication',
    icon: 'medication',
    iconColor: 'text-purple-500',
    bgColor: 'bg-purple-50 dark:bg-purple-950/30',
    borderColor: 'border-purple-200 dark:border-purple-800/50',
  },
  {
    regex: /^(?:✅\s*|💡\s*|👉\s*)?(?:\*{0,2})(?:Recommendation|RECOMMENDATION|Advice|Suggested|Tips|Lifestyle|Prevention|Action Items)(?:\*{0,2})[:\s]/i,
    type: 'recommendation',
    icon: 'lightbulb',
    iconColor: 'text-emerald-500',
    bgColor: 'bg-emerald-50 dark:bg-emerald-950/30',
    borderColor: 'border-emerald-200 dark:border-emerald-800/50',
  },
  {
    regex: /^(?:📊\s*|🔬\s*|📋\s*)?(?:\*{0,2})(?:Result|Results|Findings|Analysis|Assessment|Diagnosis|Test Results|Lab Results|Risk Score|Prediction)(?:\*{0,2})[:\s]/i,
    type: 'result',
    icon: 'analytics',
    iconColor: 'text-blue-500',
    bgColor: 'bg-blue-50 dark:bg-blue-950/30',
    borderColor: 'border-blue-200 dark:border-blue-800/50',
  },
  {
    regex: /^(?:ℹ️\s*|📝\s*)?(?:\*{0,2})(?:Note|NOTE|Important|IMPORTANT|Please note|Key Point|Remember|Disclaimer)(?:\*{0,2})[:\s]/i,
    type: 'important',
    icon: 'info',
    iconColor: 'text-sky-500',
    bgColor: 'bg-sky-50 dark:bg-sky-950/30',
    borderColor: 'border-sky-200 dark:border-sky-800/50',
  },
  {
    regex: /^(?:📄\s*|🔬\s*)?(?:\*{0,2})(?:Research Papers?|Medical Research|PubMed|Clinical Studies?|Literature Review)(?:\*{0,2})[:\s]/i,
    type: 'result',
    icon: 'science',
    iconColor: 'text-indigo-500',
    bgColor: 'bg-indigo-50 dark:bg-indigo-950/30',
    borderColor: 'border-indigo-200 dark:border-indigo-800/50',
  },
  {
    regex: /^(?:📰\s*|📢\s*)?(?:\*{0,2})(?:Medical News|Health News|Latest News|Breaking|News Update)(?:\*{0,2})[:\s]/i,
    type: 'note',
    icon: 'newspaper',
    iconColor: 'text-emerald-500',
    bgColor: 'bg-emerald-50 dark:bg-emerald-950/30',
    borderColor: 'border-emerald-200 dark:border-emerald-800/50',
  },
  {
    regex: /^(?:🖼️\s*)?(?:\*{0,2})(?:Medical Images?|Clinical Images?|Radiology|Imaging|Anatomical)(?:\*{0,2})[:\s]/i,
    type: 'normal',
    icon: 'image',
    iconColor: 'text-teal-500',
    bgColor: 'bg-teal-50 dark:bg-teal-950/30',
    borderColor: 'border-teal-200 dark:border-teal-800/50',
  },
  {
    regex: /^(?:🎬\s*)?(?:\*{0,2})(?:Medical Videos?|Educational Videos?|Related.*Videos?|Video Resources?)(?:\*{0,2})[:\s]/i,
    type: 'normal',
    icon: 'smart_display',
    iconColor: 'text-red-500',
    bgColor: 'bg-red-50 dark:bg-red-950/30',
    borderColor: 'border-red-200 dark:border-red-800/50',
  },
];

// ============================================================================
// Section Parser
// ============================================================================

/**
 * Parse the AI response content into structured sections
 * Detects health-related patterns and creates visual sections
 */
function parseContentSections(content: string): DetectedSection[] {
  if (!content || content.trim().length === 0) return [];

  // Split content by markdown headers (##, ###) or double newlines that precede known patterns
  const lines = content.split('\n');
  const sections: DetectedSection[] = [];
  let currentSection: DetectedSection | null = null;
  let currentLines: string[] = [];

  const flushSection = () => {
    if (currentLines.length > 0) {
      const sectionContent = currentLines.join('\n').trim();
      if (sectionContent) {
        if (currentSection) {
          currentSection.content = sectionContent;
          sections.push(currentSection);
        } else {
          sections.push({
            type: 'normal',
            icon: '',
            iconColor: '',
            bgColor: '',
            borderColor: '',
            content: sectionContent,
          });
        }
      }
    }
    currentLines = [];
    currentSection = null;
  };

  for (const line of lines) {
    const trimmedLine = line.trim();

    // Check if this line starts a new recognized section (via ## header or pattern match)
    const isHeader = /^#{1,3}\s+/.test(trimmedLine);
    const headerContent = trimmedLine.replace(/^#{1,3}\s+/, '');

    let matchedPattern = null;
    for (const pattern of SECTION_PATTERNS) {
      if (pattern.regex.test(isHeader ? headerContent : trimmedLine)) {
        matchedPattern = pattern;
        break;
      }
    }

    if (matchedPattern) {
      // Flush previous section
      flushSection();
      currentSection = {
        type: matchedPattern.type,
        icon: matchedPattern.icon,
        iconColor: matchedPattern.iconColor,
        bgColor: matchedPattern.bgColor,
        borderColor: matchedPattern.borderColor,
        content: '',
      };
      currentLines.push(line);
    } else if (isHeader && currentLines.length > 0) {
      // New header that's not a recognized pattern - flush and start new normal section
      flushSection();
      currentLines.push(line);
    } else {
      currentLines.push(line);
    }
  }

  // Flush the last section
  flushSection();

  return sections;
}

// ============================================================================
// Typing Indicator
// ============================================================================

const TypingIndicator: React.FC = memo(() => (
  <div className="flex items-center gap-2 py-2 px-1">
    <div className="flex items-center gap-2">
      <span className="material-symbols-outlined text-sm text-amber-500 thinking-sparkle">auto_awesome</span>
      <span className="text-[13px] text-slate-500 dark:text-slate-400 font-medium">Generating response</span>
    </div>
    <div className="flex gap-0.5 ml-1">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-1 h-1 rounded-full bg-red-400/70 dark:bg-red-500/60"
          style={{
            animation: `thinkingDotWave 1.2s infinite ease-in-out`,
            animationDelay: `${i * 0.15}s`,
          }}
        />
      ))}
    </div>
  </div>
));

TypingIndicator.displayName = 'TypingIndicator';

// ============================================================================
// Section Card Component
// ============================================================================

const SectionCard: React.FC<{ section: DetectedSection }> = memo(({ section }) => {
  if (section.type === 'normal') {
    return (
      <div className="py-0.5">
        <ChatMessageMarkdown content={section.content} showHealthAlerts={true} />
      </div>
    );
  }

  return (
    <div className={`rounded-xl border ${section.bgColor} ${section.borderColor} p-3.5 my-2 transition-all hover:shadow-sm`}>
      <div className="flex items-start gap-2.5">
        <div className={`shrink-0 mt-0.5 ${section.iconColor}`}>
          <span className="material-symbols-outlined text-lg">{section.icon}</span>
        </div>
        <div className="flex-1 min-w-0">
          <ChatMessageMarkdown content={section.content} showHealthAlerts={false} />
        </div>
      </div>
    </div>
  );
});

SectionCard.displayName = 'SectionCard';

// ============================================================================
// Error Message Component
// ============================================================================

const ErrorMessage: React.FC<{ content: string }> = memo(({ content }) => (
  <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50">
    <span className="material-symbols-outlined text-red-500 text-lg shrink-0 mt-0.5">error</span>
    <div className="flex-1 min-w-0">
      <p className="text-sm text-red-700 dark:text-red-300 font-medium">{content}</p>
      {content.includes('log in') && (
        <a
          href="#/login"
          className="inline-flex items-center gap-1 mt-2 text-xs text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-200 font-medium transition-colors"
        >
          <span className="material-symbols-outlined text-sm">login</span>
          Go to Login
        </a>
      )}
    </div>
  </div>
));

ErrorMessage.displayName = 'ErrorMessage';

// ============================================================================
// Metadata Footer
// ============================================================================

const MetadataFooter: React.FC<{ metadata: EnhancedChatMessageProps['metadata'] }> = memo(({ metadata }) => {
  const [expanded, setExpanded] = useState(false);

  if (!metadata) return null;

  const items = [
    metadata.model && { icon: 'smart_toy', label: String(metadata.model), color: 'text-slate-400' },
    metadata.processingTime && { icon: 'timer', label: `${metadata.processingTime}ms`, color: 'text-slate-400' },
    metadata.tokens && { icon: 'token', label: `~${metadata.tokens} tokens`, color: 'text-slate-400' },
  ].filter(Boolean) as Array<{ icon: string; label: string; color: string }>;

  if (items.length === 0 && (!metadata.memoryContext || metadata.memoryContext.length === 0)) return null;

  return (
    <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-800/50">
      <div className="flex items-center gap-3 flex-wrap">
        {items.map((item, i) => (
          <span key={i} className={`flex items-center gap-0.5 text-[10px] ${item.color}`}>
            <span className="material-symbols-outlined text-[11px]">{item.icon}</span>
            {item.label}
          </span>
        ))}
        {metadata.memoryContext && metadata.memoryContext.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-0.5 text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span className="material-symbols-outlined text-[11px]">memory</span>
            {metadata.memoryContext.length} context{metadata.memoryContext.length > 1 ? 's' : ''} used
            <span className={`material-symbols-outlined text-[10px] transition-transform ${expanded ? 'rotate-180' : ''}`}>
              expand_more
            </span>
          </button>
        )}
      </div>
      {expanded && metadata.memoryContext && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {metadata.memoryContext.map((ctx, i) => (
            <span key={i} className="text-[9px] bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 px-1.5 py-0.5 rounded-full border border-blue-100 dark:border-blue-500/20">
              {ctx}
            </span>
          ))}
        </div>
      )}
    </div>
  );
});

MetadataFooter.displayName = 'MetadataFooter';

// ============================================================================
// Main Enhanced Chat Message Component
// ============================================================================

export const EnhancedChatMessage: React.FC<EnhancedChatMessageProps> = memo(({
  content,
  sources,
  isStreaming = false,
  isError = false,
  metadata,
}) => {
  // ALL hooks must be called unconditionally before any early returns
  // to avoid "Rendered more hooks than during the previous render" error.

  // Parse content into structured sections
  const sections = useMemo(() => {
    if (!content || isStreaming || isError) return [];
    return parseContentSections(content);
  }, [content, isStreaming, isError]);

  // Check if content has rich medical media (images, videos, papers, news)
  const hasMedicalMedia = useMemo(() => {
    if (!content || isStreaming || isError) return false;
    const { images, videos, papers, news } = parseMedicalContent(content);
    return images.length > 0 || videos.length > 0 || papers.length > 0 || news.length > 0;
  }, [content, isStreaming, isError]);

  const hasSpecialSections = useMemo(() => sections.some(s => s.type !== 'normal'), [sections]);

  // If streaming, show content as-is with typing indicator
  if (isStreaming) {
    if (!content) {
      return <TypingIndicator />;
    }
    return (
      <div className="py-0.5">
        <ChatMessageMarkdown content={content} showHealthAlerts={false} />
      </div>
    );
  }

  // If error, show error state
  if (isError) {
    return <ErrorMessage content={content} />;
  }

  if (!hasSpecialSections && !hasMedicalMedia) {
    return (
      <div>
        <ChatMessageMarkdown content={content} sources={sources} showHealthAlerts={true} />
        <MetadataFooter metadata={metadata} />
      </div>
    );
  }

  // Render with structured sections + medical media
  return (
    <div className="space-y-0.5">
      {sections.map((section, index) => (
        <SectionCard key={index} section={section} />
      ))}

      {/* Rich Medical Media Display */}
      {hasMedicalMedia && (
        <MedicalMediaSection content={content} />
      )}

      {/* Sources */}
      {sources && sources.length > 0 && (
        <ChatMessageMarkdown content="" sources={sources} showHealthAlerts={false} />
      )}

      {/* Metadata */}
      <MetadataFooter metadata={metadata} />
    </div>
  );
});

EnhancedChatMessage.displayName = 'EnhancedChatMessage';

export default EnhancedChatMessage;
