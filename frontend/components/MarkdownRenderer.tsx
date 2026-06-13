/**
 * Markdown Renderer Component
 *
 * Renders markdown content with proper styling for chat messages
 * and health information display.
 *
 * Features:
 * - XSS protection via DOMPurify sanitization
 * - Syntax highlighting for code blocks
 * - Custom styling for health-related content
 * - Support for tables (medication info)
 * - Links with safety handling
 *
 * For enhanced syntax highlighting, install: npm install prism-react-renderer
 */

import React, { memo, useMemo } from 'react';
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import DOMPurify from 'dompurify';

// ============================================================================
// DOMPurify Configuration for XSS Protection
// ============================================================================

/**
 * Sanitize content before markdown parsing
 * This provides defense-in-depth against XSS attacks
 */
const sanitizeContent = (content: string): string => {
  if (!content) return '';

  // First pass: sanitize the raw input
  const sanitized = DOMPurify.sanitize(content, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's', 'del', 'ins', 'mark',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'code', 'pre', 'kbd', 'samp', 'var',
      'blockquote', 'q', 'cite', 'abbr',
      'a', 'img', 'iframe',
      'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption', 'colgroup', 'col',
      'div', 'span', 'hr', 'sup', 'sub',
      'details', 'summary',
    ],
    ALLOWED_ATTR: [
      'href', 'target', 'rel', 'title', 'alt', 'src', 'width', 'height',
      'class', 'id', 'lang', 'dir', 'colspan', 'rowspan', 'scope',
      'frameborder', 'allow', 'allowfullscreen', 'loading',
    ],
    ADD_ATTR: ['target', 'rel'],
    ALLOW_DATA_ATTR: false,
    FORBID_TAGS: ['script', 'style', 'object', 'embed', 'form', 'input', 'button'],
    FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onblur'],
    RETURN_DOM: false,
    RETURN_DOM_FRAGMENT: false,
  });

  return String(sanitized);
};

/**
 * Sanitize URL for safe linking
 * Prevents javascript: and data: protocol attacks
 */
const sanitizeUrl = (url: string | undefined): string | undefined => {
  if (!url) return undefined;

  // Allow only http, https, mailto, and tel protocols
  const allowedProtocols = ['http:', 'https:', 'mailto:', 'tel:'];

  try {
    const parsed = new URL(url, window.location.origin);
    if (allowedProtocols.includes(parsed.protocol)) {
      return url;
    }
  } catch {
    // Relative URLs are OK
    if (url.startsWith('/') || url.startsWith('#')) {
      return url;
    }
  }

  // Block potentially dangerous URLs
  console.warn('[MarkdownRenderer] Blocked potentially unsafe URL:', url);
  return undefined;
};

// ============================================================================
// Types
// ============================================================================

interface MarkdownRendererProps {
  content: string;
  className?: string;
  variant?: 'chat' | 'article' | 'compact';
}

// ============================================================================
// Simple Syntax Highlighting (no external dependency)
// ============================================================================

interface CodeBlockProps {
  language: string;
  code: string;
}

// Language-specific keyword patterns for basic highlighting
const LANGUAGE_PATTERNS: Record<string, { keywords: string[]; builtins: string[]; strings: RegExp; comments: RegExp; numbers: RegExp }> = {
  javascript: {
    keywords: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'extends', 'import', 'export', 'from', 'default', 'async', 'await', 'try', 'catch', 'throw', 'new', 'this', 'typeof', 'instanceof'],
    builtins: ['console', 'window', 'document', 'Array', 'Object', 'String', 'Number', 'Boolean', 'Promise', 'Map', 'Set', 'JSON', 'Math', 'Date', 'Error', 'RegExp'],
    strings: /(['"`])(?:(?!\1)[^\\]|\\.)*\1/g,
    comments: /\/\/.*$|\/\*[\s\S]*?\*\//gm,
    numbers: /\b\d+\.?\d*\b/g,
  },
  typescript: {
    keywords: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'extends', 'import', 'export', 'from', 'default', 'async', 'await', 'try', 'catch', 'throw', 'new', 'this', 'typeof', 'instanceof', 'interface', 'type', 'enum', 'namespace', 'implements', 'private', 'public', 'protected', 'readonly', 'abstract', 'as', 'is'],
    builtins: ['console', 'window', 'document', 'Array', 'Object', 'String', 'Number', 'Boolean', 'Promise', 'Map', 'Set', 'JSON', 'Math', 'Date', 'Error', 'RegExp', 'any', 'void', 'never', 'unknown'],
    strings: /(['"`])(?:(?!\1)[^\\]|\\.)*\1/g,
    comments: /\/\/.*$|\/\*[\s\S]*?\*\//gm,
    numbers: /\b\d+\.?\d*\b/g,
  },
  python: {
    keywords: ['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'lambda', 'pass', 'break', 'continue', 'raise', 'global', 'nonlocal', 'assert', 'async', 'await', 'and', 'or', 'not', 'in', 'is'],
    builtins: ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'bool', 'None', 'True', 'False', 'self', 'cls', 'super', 'open', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr'],
    strings: /(['"])(?:(?!\1)[^\\]|\\.)*\1|'''[\s\S]*?'''|"""[\s\S]*?"""/g,
    comments: /#.*$/gm,
    numbers: /\b\d+\.?\d*\b/g,
  },
  json: {
    keywords: [],
    builtins: ['true', 'false', 'null'],
    strings: /"(?:[^"\\]|\\.)*"/g,
    comments: /(?:)/g, // JSON doesn't have comments
    numbers: /\b-?\d+\.?\d*(?:[eE][+-]?\d+)?\b/g,
  },
  bash: {
    keywords: ['if', 'then', 'else', 'elif', 'fi', 'for', 'while', 'do', 'done', 'case', 'esac', 'function', 'return', 'exit', 'export', 'local', 'readonly'],
    builtins: ['echo', 'cd', 'pwd', 'ls', 'rm', 'cp', 'mv', 'mkdir', 'cat', 'grep', 'sed', 'awk', 'curl', 'wget', 'npm', 'pip', 'git', 'docker'],
    strings: /(['"])(?:(?!\1)[^\\]|\\.)*\1/g,
    comments: /#.*$/gm,
    numbers: /\b\d+\b/g,
  },
};

// Alias mappings
const LANGUAGE_ALIASES: Record<string, string> = {
  js: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  jsx: 'javascript',
  py: 'python',
  sh: 'bash',
  shell: 'bash',
  zsh: 'bash',
};

/**
 * Simple syntax highlighter using regex-based token matching
 */
const highlightCode = (code: string, language: string): React.ReactNode => {
  const normalizedLang = LANGUAGE_ALIASES[language.toLowerCase()] || language.toLowerCase();
  const patterns = LANGUAGE_PATTERNS[normalizedLang];

  if (!patterns) {
    // Return plain code for unsupported languages
    return <span className="text-slate-100">{code}</span>;
  }

  // Tokenize the code
  const tokens: Array<{ type: string; value: string; start: number; end: number }> = [];

  // Match comments first (highest priority)
  let match;
  const commentRegex = new RegExp(patterns.comments.source, patterns.comments.flags);
  while ((match = commentRegex.exec(code)) !== null) {
    tokens.push({ type: 'comment', value: match[0], start: match.index, end: match.index + match[0].length });
  }

  // Match strings
  const stringRegex = new RegExp(patterns.strings.source, patterns.strings.flags);
  while ((match = stringRegex.exec(code)) !== null) {
    tokens.push({ type: 'string', value: match[0], start: match.index, end: match.index + match[0].length });
  }

  // Match numbers
  const numberRegex = new RegExp(patterns.numbers.source, patterns.numbers.flags);
  while ((match = numberRegex.exec(code)) !== null) {
    tokens.push({ type: 'number', value: match[0], start: match.index, end: match.index + match[0].length });
  }

  // Match keywords
  patterns.keywords.forEach(keyword => {
    const keywordRegex = new RegExp(`\\b${keyword}\\b`, 'g');
    while ((match = keywordRegex.exec(code)) !== null) {
      tokens.push({ type: 'keyword', value: match[0], start: match.index, end: match.index + match[0].length });
    }
  });

  // Match builtins
  patterns.builtins.forEach(builtin => {
    const builtinRegex = new RegExp(`\\b${builtin}\\b`, 'g');
    while ((match = builtinRegex.exec(code)) !== null) {
      tokens.push({ type: 'builtin', value: match[0], start: match.index, end: match.index + match[0].length });
    }
  });

  // Sort tokens by position and remove overlaps (prefer earlier and longer tokens)
  tokens.sort((a, b) => a.start - b.start || b.end - a.end);

  const filteredTokens: typeof tokens = [];
  let lastEnd = 0;

  tokens.forEach(token => {
    if (token.start >= lastEnd) {
      filteredTokens.push(token);
      lastEnd = token.end;
    }
  });

  // Build highlighted output
  const result: React.ReactNode[] = [];
  let currentPos = 0;

  filteredTokens.forEach((token, index) => {
    // Add plain text before this token
    if (token.start > currentPos) {
      result.push(
        <span key={`plain-${index}`} className="text-slate-100">
          {code.slice(currentPos, token.start)}
        </span>
      );
    }

    // Add the highlighted token
    const tokenClasses: Record<string, string> = {
      keyword: 'text-purple-400 font-medium',
      builtin: 'text-cyan-400',
      string: 'text-green-400',
      number: 'text-orange-400',
      comment: 'text-slate-500 italic',
    };

    result.push(
      <span key={`token-${index}`} className={tokenClasses[token.type] || 'text-slate-100'}>
        {token.value}
      </span>
    );

    currentPos = token.end;
  });

  // Add remaining plain text
  if (currentPos < code.length) {
    result.push(
      <span key="plain-end" className="text-slate-100">
        {code.slice(currentPos)}
      </span>
    );
  }

  return <>{result}</>;
};

/**
 * Code block with syntax highlighting and copy button
 */
const CodeBlock: React.FC<CodeBlockProps> = memo(({ language, code }) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy code:', error);
    }
  };

  const highlightedCode = useMemo(() => highlightCode(code, language), [code, language]);

  return (
    <div className="relative group my-3">
      {/* Language badge */}
      {language && language !== 'text' && (
        <div className="absolute top-0 left-0 px-2 py-1 text-[10px] font-mono text-slate-400 bg-slate-800 rounded-tl-lg rounded-br-lg uppercase tracking-wider">
          {language}
        </div>
      )}

      {/* Copy button */}
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 p-1.5 rounded bg-slate-700/50 hover:bg-slate-700 text-slate-400 hover:text-white opacity-0 group-hover:opacity-100 transition-all"
        title={copied ? 'Copied!' : 'Copy code'}
      >
        {copied ? (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        )}
      </button>

      {/* Code content */}
      <pre className={`p-4 ${language ? 'pt-8' : ''} bg-slate-900 dark:bg-slate-950 rounded-lg overflow-x-auto text-sm font-mono`}>
        <code>{highlightedCode}</code>
      </pre>
    </div>
  );
});

CodeBlock.displayName = 'CodeBlock';

// ============================================================================
// Custom Components for Markdown
// ============================================================================

const createComponents = (variant: string): Components => ({
  // Headings
  h1: ({ children }) => (
    <h1 className={`text-xl font-bold text-slate-900 dark:text-white mt-4 mb-2 ${
      variant === 'compact' ? 'text-lg' : ''
    }`}>
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className={`text-lg font-semibold text-slate-800 dark:text-slate-100 mt-3 mb-2 ${
      variant === 'compact' ? 'text-base' : ''
    }`}>
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className={`text-base font-semibold text-slate-700 dark:text-slate-200 mt-2 mb-1 ${
      variant === 'compact' ? 'text-sm' : ''
    }`}>
      {children}
    </h3>
  ),

  // Paragraphs
  p: ({ children }) => (
    <p className={`text-slate-700 dark:text-slate-300 mb-2 leading-relaxed ${
      variant === 'compact' ? 'text-sm mb-1' : ''
    }`}>
      {children}
    </p>
  ),

  // Lists
  ul: ({ children }) => (
    <ul className={`list-disc list-inside mb-2 space-y-1 text-slate-700 dark:text-slate-300 ${
      variant === 'compact' ? 'text-sm' : ''
    }`}>
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className={`list-decimal list-inside mb-2 space-y-1 text-slate-700 dark:text-slate-300 ${
      variant === 'compact' ? 'text-sm' : ''
    }`}>
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="ml-2">{children}</li>
  ),

  // Strong/Bold - often used for warnings in health content
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-900 dark:text-white">
      {children}
    </strong>
  ),

  // Emphasis/Italic
  em: ({ children }) => (
    <em className="italic text-slate-600 dark:text-slate-400">
      {children}
    </em>
  ),

  // Code blocks with syntax highlighting
  code: ({ className, children, ...props }) => {
    const isInline = !className;
    const code = String(children).replace(/\n$/, '');

    if (isInline) {
      return (
        <code className="px-1.5 py-0.5 bg-slate-100 dark:bg-slate-800 text-rose-600 dark:text-rose-400 rounded text-sm font-mono">
          {children}
        </code>
      );
    }

    // Extract language from className (format: language-xxx)
    const language = className?.replace('language-', '') || 'text';

    return <CodeBlock language={language} code={code} />;
  },

  // Pre (code block container) - simplified since CodeBlock handles styling
  pre: ({ children }) => (
    <>{children}</>
  ),

  // Blockquotes - useful for important health notes
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-blue-500 dark:border-blue-400 pl-4 py-2 my-2 bg-blue-50 dark:bg-blue-900/20 rounded-r-lg italic text-slate-600 dark:text-slate-400">
      {children}
    </blockquote>
  ),

  // Tables - great for medication schedules, vital signs
  table: ({ children }) => (
    <div className="overflow-x-auto my-3">
      <table className="min-w-full border-collapse text-sm">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-slate-100 dark:bg-slate-800">{children}</thead>
  ),
  tbody: ({ children }) => (
    <tbody className="divide-y divide-slate-200 dark:divide-slate-700">{children}</tbody>
  ),
  tr: ({ children }) => (
    <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/50">{children}</tr>
  ),
  th: ({ children }) => (
    <th className="px-3 py-2 text-left font-semibold text-slate-700 dark:text-slate-300 border-b border-slate-200 dark:border-slate-700">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-3 py-2 text-slate-600 dark:text-slate-400">
      {children}
    </td>
  ),

  // Links - with external link handling, URL sanitization, and video detection
  a: ({ href, children }) => {
    const safeHref = sanitizeUrl(href);
    if (!safeHref) {
      // Return just the text if URL is unsafe
      return <span className="text-slate-600">{children}</span>;
    }

    const isExternal = safeHref.startsWith('http');

    // Detect YouTube links and render as video cards
    const youtubeMatch = safeHref.match(/(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
    if (youtubeMatch) {
      const videoId = youtubeMatch[1];
      const title = typeof children === 'string' ? children : 'Medical Video';
      return (
        <div className="my-2 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 hover:border-red-400 dark:hover:border-red-500 transition-all max-w-lg">
          <a href={safeHref} target="_blank" rel="noopener noreferrer" className="block group">
            <div className="relative aspect-video bg-slate-200 dark:bg-slate-800">
              <img
                src={`https://img.youtube.com/vi/${videoId}/hqdefault.jpg`}
                alt={String(title)}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/30 transition-colors">
                <div className="w-12 h-12 rounded-full bg-red-600 group-hover:bg-red-500 flex items-center justify-center shadow-lg group-hover:scale-110 transition-all">
                  <svg className="w-6 h-6 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                </div>
              </div>
            </div>
            <div className="p-2 bg-white dark:bg-slate-800/50">
              <p className="text-sm font-medium text-slate-800 dark:text-slate-200 line-clamp-2">{children}</p>
              <p className="text-xs text-red-500 mt-0.5">YouTube</p>
            </div>
          </a>
        </div>
      );
    }

    return (
      <a
        href={safeHref}
        target={isExternal ? '_blank' : undefined}
        rel={isExternal ? 'noopener noreferrer' : undefined}
        className="text-blue-600 dark:text-blue-400 hover:underline inline-flex items-center gap-1"
      >
        {children}
        {isExternal && (
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        )}
      </a>
    );
  },

  // Horizontal rule
  hr: () => (
    <hr className="my-4 border-slate-200 dark:border-slate-700" />
  ),

  // Images - enhanced for medical content with click-to-expand
  img: ({ src, alt }) => {
    const safeUrl = sanitizeUrl(src);
    if (!safeUrl) return null;

    return (
      <span className="block my-2">
        <a href={safeUrl} target="_blank" rel="noopener noreferrer" className="block group">
          <span className="block relative rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-500 transition-all max-w-md">
            <img
              src={safeUrl}
              alt={alt || 'Medical image'}
              className="max-w-full h-auto rounded-lg group-hover:scale-[1.02] transition-transform duration-300"
              loading="lazy"
              onError={(e) => {
                (e.target as HTMLImageElement).parentElement!.style.display = 'none';
              }}
            />
            <span className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/50 rounded-full p-1.5">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
              </svg>
            </span>
          </span>
          {alt && alt !== 'Medical image' && (
            <span className="block text-xs text-slate-500 dark:text-slate-400 mt-1 text-center italic">{alt}</span>
          )}
        </a>
      </span>
    );
  },
});

// ============================================================================
// Main Component
// ============================================================================

/**
 * Renders markdown content with health-focused styling
 * Content is sanitized using DOMPurify for XSS protection
 *
 * @example
 * ```tsx
 * <MarkdownRenderer
 *   content="**Important:** Take medication with food"
 *   variant="chat"
 * />
 * ```
 */
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = memo(({
  content,
  className = '',
  variant = 'chat',
}) => {
  // Memoize sanitized content to avoid re-sanitizing on every render
  const sanitizedContent = useMemo(() => {
    if (!content) return '';
    return sanitizeContent(content);
  }, [content]);

  if (!sanitizedContent) return null;

  const components = createComponents(variant);

  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={components}
      >
        {sanitizedContent}
      </ReactMarkdown>
    </div>
  );
});

MarkdownRenderer.displayName = 'MarkdownRenderer';

// ============================================================================
// Citation/Source Display Component
// ============================================================================

interface Source {
  title: string;
  category?: string;
  relevance?: number;
}

interface CitationDisplayProps {
  sources: Source[];
}

/**
 * Displays RAG citation sources in a collapsible format
 */
const CitationDisplay: React.FC<CitationDisplayProps> = memo(({ sources }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-slate-700/50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
      >
        <svg
          className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="font-medium">
          {sources.length} source{sources.length > 1 ? 's' : ''} referenced
        </span>
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-1.5 animate-in slide-in-from-top-2 duration-200">
          {sources.map((source, index) => (
            <div
              key={index}
              className="flex items-start gap-2 text-xs bg-slate-800/50 rounded px-2 py-1.5"
            >
              <span className="text-slate-500 font-mono shrink-0">[{index + 1}]</span>
              <div className="flex-1 min-w-0">
                <span className="text-slate-300">{source.title}</span>
                {source.category && (
                  <span className="ml-2 text-slate-500 text-[10px] uppercase tracking-wider">
                    {source.category}
                  </span>
                )}
              </div>
              {source.relevance !== undefined && (
                <span className="text-green-400 text-[10px] shrink-0">
                  {Math.round(source.relevance * 100)}%
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

CitationDisplay.displayName = 'CitationDisplay';

// ============================================================================
// Specialized Variants
// ============================================================================

interface ChatMessageMarkdownProps {
  content: string;
  isAssistant?: boolean;
  sources?: Source[];
  showHealthAlerts?: boolean;
}

/**
 * Markdown specifically styled for chat messages with optional citations
 */
export const ChatMessageMarkdown: React.FC<ChatMessageMarkdownProps> = memo(({
  content,
  isAssistant = true,
  sources,
  showHealthAlerts = false,
}) => (
  <div>
    <MarkdownRenderer
      content={content}
      variant="compact"
      className={isAssistant ? 'assistant-message' : 'user-message'}
    />
    {isAssistant && sources && sources.length > 0 && (
      <CitationDisplay sources={sources} />
    )}
  </div>
));

ChatMessageMarkdown.displayName = 'ChatMessageMarkdown';

// ============================================================================
// Health Alert Component
// ============================================================================

interface HealthAlertMarkdownProps {
  content: string;
  severity?: 'info' | 'warning' | 'urgent';
}

/**
 * Markdown for health alerts with severity styling
 */
export const HealthAlertMarkdown: React.FC<HealthAlertMarkdownProps> = memo(({
  content,
  severity = 'info',
}) => {
  const severityStyles = {
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    warning: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800',
    urgent: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
  };

  const iconColors = {
    info: 'text-blue-500',
    warning: 'text-amber-500',
    urgent: 'text-red-500',
  };

  return (
    <div className={`p-4 rounded-lg border ${severityStyles[severity]}`}>
      <div className="flex gap-3">
        <div className={`shrink-0 ${iconColors[severity]}`}>
          {severity === 'info' && (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
          {severity === 'warning' && (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )}
          {severity === 'urgent' && (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
        </div>
        <MarkdownRenderer content={content} variant="compact" />
      </div>
    </div>
  );
});

HealthAlertMarkdown.displayName = 'HealthAlertMarkdown';

// ============================================================================
// Export
// ============================================================================

export default MarkdownRenderer;
