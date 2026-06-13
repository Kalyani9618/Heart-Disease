/**
 * Medical Media Renderer Component
 *
 * Renders rich medical content including:
 * - Research paper cards (PubMed, WHO, etc.)
 * - Medical image galleries with lightbox
 * - Video cards with YouTube embeds and play overlays
 * - Medical news article cards
 *
 * Only renders verified medical-domain content.
 */

import React, { memo, useState, useCallback, useMemo } from 'react';

// ============================================================================
// Types
// ============================================================================

interface MedicalImage {
  src: string;
  alt: string;
  source?: string;
}

interface MedicalVideo {
  url: string;
  title: string;
  thumbnail?: string;
  duration?: string;
  source?: string;
}

interface ResearchPaper {
  title: string;
  url: string;
  authors?: string;
  journal?: string;
  year?: string;
  abstract?: string;
  pmid?: string;
}

interface MedicalNewsItem {
  title: string;
  url: string;
  source?: string;
  date?: string;
  snippet?: string;
}

// ============================================================================
// Utility: YouTube URL Detection
// ============================================================================

const YOUTUBE_REGEX = /(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;

function extractYouTubeId(url: string): string | null {
  const match = url.match(YOUTUBE_REGEX);
  return match ? match[1] : null;
}

function isYouTubeUrl(url: string): boolean {
  return YOUTUBE_REGEX.test(url);
}

// ============================================================================
// Medical Image Gallery
// ============================================================================

interface ImageLightboxProps {
  src: string;
  alt: string;
  onClose: () => void;
}

const ImageLightbox: React.FC<ImageLightboxProps> = memo(({ src, alt, onClose }) => (
  <div
    className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
    onClick={onClose}
  >
    <button
      onClick={onClose}
      className="absolute top-4 right-4 text-white/80 hover:text-white p-2 rounded-full bg-black/30 hover:bg-black/50 transition-colors"
      aria-label="Close"
    >
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
    <div className="max-w-4xl max-h-[90vh] relative" onClick={(e) => e.stopPropagation()}>
      <img
        src={src}
        alt={alt}
        className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl"
        loading="lazy"
      />
      {alt && (
        <p className="text-center text-white/70 text-sm mt-2 px-4">{alt}</p>
      )}
    </div>
  </div>
));

ImageLightbox.displayName = 'ImageLightbox';

interface MedicalImageGalleryProps {
  images: MedicalImage[];
}

export const MedicalImageGallery: React.FC<MedicalImageGalleryProps> = memo(({ images }) => {
  const [lightboxImage, setLightboxImage] = useState<MedicalImage | null>(null);

  if (!images || images.length === 0) return null;

  return (
    <>
      <div className="my-3">
        <div className={`grid gap-2 ${
          images.length === 1 ? 'grid-cols-1' :
          images.length === 2 ? 'grid-cols-2' :
          'grid-cols-2 sm:grid-cols-3'
        }`}>
          {images.map((image, index) => (
            <button
              key={index}
              onClick={() => setLightboxImage(image)}
              className="group relative rounded-lg overflow-hidden bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-500 transition-all hover:shadow-md aspect-video"
            >
              <img
                src={image.src}
                alt={image.alt}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                loading="lazy"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
              {/* Hover overlay */}
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                <div className="opacity-0 group-hover:opacity-100 transition-opacity bg-black/50 rounded-full p-2">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                  </svg>
                </div>
              </div>
              {/* Caption */}
              {image.alt && (
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-2">
                  <p className="text-white text-xs truncate">{image.alt}</p>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Lightbox */}
      {lightboxImage && (
        <ImageLightbox
          src={lightboxImage.src}
          alt={lightboxImage.alt}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </>
  );
});

MedicalImageGallery.displayName = 'MedicalImageGallery';

// ============================================================================
// Video Card Component
// ============================================================================

interface VideoCardProps {
  video: MedicalVideo;
}

export const VideoCard: React.FC<VideoCardProps> = memo(({ video }) => {
  const [showEmbed, setShowEmbed] = useState(false);
  const youtubeId = useMemo(() => extractYouTubeId(video.url), [video.url]);

  const handlePlay = useCallback(() => {
    if (youtubeId) {
      setShowEmbed(true);
    } else {
      window.open(video.url, '_blank', 'noopener,noreferrer');
    }
  }, [youtubeId, video.url]);

  if (showEmbed && youtubeId) {
    return (
      <div className="my-2 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 shadow-sm">
        <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>
          <iframe
            className="absolute inset-0 w-full h-full"
            src={`https://www.youtube-nocookie.com/embed/${youtubeId}?autoplay=1&rel=0`}
            title={video.title}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
        <div className="p-2 bg-slate-50 dark:bg-slate-800">
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate">{video.title}</p>
          {video.source && (
            <p className="text-xs text-slate-500 dark:text-slate-400">{video.source}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={handlePlay}
      className="group w-full my-2 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-500 transition-all hover:shadow-md text-left"
    >
      <div className="relative aspect-video bg-slate-200 dark:bg-slate-800">
        {video.thumbnail ? (
          <img
            src={video.thumbnail}
            alt={video.title}
            className="w-full h-full object-cover"
            loading="lazy"
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
            }}
          />
        ) : youtubeId ? (
          <img
            src={`https://img.youtube.com/vi/${youtubeId}/hqdefault.jpg`}
            alt={video.title}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <svg className="w-16 h-16 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
        )}

        {/* Play button overlay */}
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/30 transition-colors">
          <div className="w-14 h-14 rounded-full bg-red-600 group-hover:bg-red-500 flex items-center justify-center shadow-lg group-hover:scale-110 transition-all">
            <svg className="w-7 h-7 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
        </div>

        {/* Duration badge */}
        {video.duration && (
          <div className="absolute bottom-2 right-2 bg-black/80 text-white text-xs px-1.5 py-0.5 rounded">
            {video.duration}
          </div>
        )}
      </div>

      <div className="p-2.5 bg-white dark:bg-slate-800/50">
        <p className="text-sm font-medium text-slate-800 dark:text-slate-200 line-clamp-2">{video.title}</p>
        {video.source && (
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{video.source}</p>
        )}
      </div>
    </button>
  );
});

VideoCard.displayName = 'VideoCard';

interface MedicalVideoGridProps {
  videos: MedicalVideo[];
}

export const MedicalVideoGrid: React.FC<MedicalVideoGridProps> = memo(({ videos }) => {
  if (!videos || videos.length === 0) return null;

  return (
    <div className={`grid gap-3 my-3 ${
      videos.length === 1 ? 'grid-cols-1 max-w-lg' :
      'grid-cols-1 sm:grid-cols-2'
    }`}>
      {videos.map((video, index) => (
        <VideoCard key={index} video={video} />
      ))}
    </div>
  );
});

MedicalVideoGrid.displayName = 'MedicalVideoGrid';

// ============================================================================
// Research Paper Card
// ============================================================================

interface ResearchPaperCardProps {
  paper: ResearchPaper;
}

export const ResearchPaperCard: React.FC<ResearchPaperCardProps> = memo(({ paper }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="my-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 hover:border-blue-300 dark:hover:border-blue-600 transition-colors overflow-hidden">
      <div className="p-3">
        <div className="flex items-start gap-2.5">
          {/* Paper icon */}
          <div className="shrink-0 mt-0.5">
            <div className="w-8 h-8 rounded bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center">
              <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
          </div>

          <div className="flex-1 min-w-0">
            {/* Title */}
            <a
              href={paper.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline line-clamp-2"
            >
              {paper.title}
            </a>

            {/* Meta info */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1">
              {paper.authors && (
                <span className="text-xs text-slate-500 dark:text-slate-400 truncate max-w-[200px]">
                  {paper.authors}
                </span>
              )}
              {paper.journal && (
                <span className="text-xs text-emerald-600 dark:text-emerald-400 italic">
                  {paper.journal}
                </span>
              )}
              {paper.year && (
                <span className="text-xs text-slate-400">
                  {paper.year}
                </span>
              )}
              {paper.pmid && (
                <span className="text-[10px] bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 px-1.5 py-0.5 rounded-full font-mono">
                  PMID: {paper.pmid}
                </span>
              )}
            </div>

            {/* Abstract (expandable) */}
            {paper.abstract && (
              <div className="mt-1.5">
                <p className={`text-xs text-slate-600 dark:text-slate-400 leading-relaxed ${
                  !expanded ? 'line-clamp-2' : ''
                }`}>
                  {paper.abstract}
                </p>
                {paper.abstract.length > 150 && (
                  <button
                    onClick={() => setExpanded(!expanded)}
                    className="text-xs text-blue-500 hover:text-blue-400 mt-0.5"
                  >
                    {expanded ? 'Show less' : 'Read more'}
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

ResearchPaperCard.displayName = 'ResearchPaperCard';

// ============================================================================
// Medical News Card
// ============================================================================

interface MedicalNewsCardProps {
  news: MedicalNewsItem;
}

export const MedicalNewsCard: React.FC<MedicalNewsCardProps> = memo(({ news }) => (
  <a
    href={news.url}
    target="_blank"
    rel="noopener noreferrer"
    className="block my-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 hover:border-emerald-300 dark:hover:border-emerald-600 transition-all hover:shadow-sm p-3"
  >
    <div className="flex items-start gap-2.5">
      <div className="shrink-0 mt-0.5">
        <div className="w-8 h-8 rounded bg-emerald-50 dark:bg-emerald-900/30 flex items-center justify-center">
          <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
          </svg>
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-800 dark:text-slate-200 line-clamp-2">
          {news.title}
        </p>
        {news.snippet && (
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
            {news.snippet}
          </p>
        )}
        <div className="flex items-center gap-2 mt-1.5">
          {news.source && (
            <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">
              {news.source}
            </span>
          )}
          {news.date && (
            <span className="text-[10px] text-slate-400">
              {news.date}
            </span>
          )}
        </div>
      </div>
      <svg className="w-4 h-4 text-slate-400 shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
      </svg>
    </div>
  </a>
));

MedicalNewsCard.displayName = 'MedicalNewsCard';

// ============================================================================
// Content Parser - Extracts structured data from markdown sections
// ============================================================================

/**
 * Parse medical content sections from markdown text.
 * Detects patterns from the backend medical_search.py output format.
 */
export function parseMedicalContent(content: string): {
  images: MedicalImage[];
  videos: MedicalVideo[];
  papers: ResearchPaper[];
  news: MedicalNewsItem[];
} {
  const images: MedicalImage[] = [];
  const videos: MedicalVideo[] = [];
  const papers: ResearchPaper[] = [];
  const news: MedicalNewsItem[] = [];

  if (!content) return { images, videos, papers, news };

  // Parse images: ![alt](url)
  const imageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
  let match;
  while ((match = imageRegex.exec(content)) !== null) {
    const alt = match[1];
    const src = match[2];
    // Only add if it looks like an actual image URL
    if (src.match(/\.(jpg|jpeg|png|gif|webp|svg|bmp)/i) || src.includes('images') || src.includes('img') || src.includes('photo') || src.includes('media')) {
      images.push({ src, alt: alt || 'Medical image' });
    }
  }

  // Parse videos: lines with YouTube or medical video URLs
  const videoLineRegex = /(?:\d+\.\s+)?\*\*\[?([^\]*]+)\]?\*\*[^\n]*\n[^\n]*(?:URL|Link|Watch):\s*\[?([^\]\s]+)\]?/gi;
  const youtubeLineRegex = /\[([^\]]+)\]\((https?:\/\/(?:www\.)?(?:youtube\.com|youtu\.be)[^\)]+)\)/g;

  while ((match = youtubeLineRegex.exec(content)) !== null) {
    const title = match[1];
    const url = match[2];
    if (!videos.some(v => v.url === url)) {
      videos.push({ title, url });
    }
  }

  // Parse video sections with thumbnail pattern: [![title](thumb)](url)
  const thumbVideoRegex = /\[\!\[([^\]]*)\]\(([^)]+)\)\]\(([^)]+)\)/g;
  while ((match = thumbVideoRegex.exec(content)) !== null) {
    const title = match[1];
    const thumbnail = match[2];
    const url = match[3];
    if (!videos.some(v => v.url === url)) {
      videos.push({ title: title || 'Medical Video', url, thumbnail });
    }
  }

  // Parse research papers from sections marked with 📄 or "Research Papers"
  const paperSectionRegex = /(?:##\s*(?:📄|🔬)\s*(?:Research Papers|Medical Research)[^\n]*\n)([\s\S]*?)(?=\n##\s|$)/i;
  const paperSection = content.match(paperSectionRegex);
  if (paperSection) {
    const paperItems = paperSection[1].match(/\d+\.\s+\*\*([^*]+)\*\*[\s\S]*?(?=\n\d+\.\s+\*\*|\n##|\n---|\n$)/g);
    if (paperItems) {
      for (const item of paperItems) {
        const titleMatch = item.match(/\*\*([^*]+)\*\*/);
        const urlMatch = item.match(/\[(?:Read|View|Link|Full Text)[^\]]*\]\(([^)]+)\)/i) || item.match(/https?:\/\/[^\s\)]+/);
        const authorsMatch = item.match(/Authors?:\s*([^\n]+)/i);
        const journalMatch = item.match(/Journal:\s*([^\n]+)/i);
        const yearMatch = item.match(/\((\d{4})\)/);
        const abstractMatch = item.match(/Abstract:\s*([^\n]+)/i) || item.match(/>\s*([^\n]+)/);
        const pmidMatch = item.match(/PMID:\s*(\d+)/i);

        if (titleMatch) {
          papers.push({
            title: titleMatch[1].trim(),
            url: urlMatch ? (urlMatch[1] || urlMatch[0]) : '#',
            authors: authorsMatch?.[1]?.trim(),
            journal: journalMatch?.[1]?.trim(),
            year: yearMatch?.[1],
            abstract: abstractMatch?.[1]?.trim(),
            pmid: pmidMatch?.[1],
          });
        }
      }
    }
  }

  // Parse news from sections marked with 📰 or "Medical News"
  const newsSectionRegex = /(?:##\s*(?:📰|📢)\s*(?:Medical News|Latest News|Health News)[^\n]*\n)([\s\S]*?)(?=\n##\s|$)/i;
  const newsSection = content.match(newsSectionRegex);
  if (newsSection) {
    const newsItems = newsSection[1].match(/\d+\.\s+\*\*([^*]+)\*\*[\s\S]*?(?=\n\d+\.\s+\*\*|\n##|\n---|\n$)/g);
    if (newsItems) {
      for (const item of newsItems) {
        const titleMatch = item.match(/\*\*\[?([^\]*]+)\]?\*\*/);
        const urlMatch = item.match(/\[(?:Read|View|Link|More)[^\]]*\]\(([^)]+)\)/i) || item.match(/https?:\/\/[^\s\)]+/);
        const sourceMatch = item.match(/Source:\s*([^\n]+)/i);
        const dateMatch = item.match(/Date:\s*([^\n]+)/i) || item.match(/(\d{4}-\d{2}-\d{2})/);
        const snippetMatch = item.match(/(?:Summary|Snippet|Description):\s*([^\n]+)/i);

        if (titleMatch) {
          news.push({
            title: titleMatch[1].trim(),
            url: urlMatch ? (urlMatch[1] || urlMatch[0]) : '#',
            source: sourceMatch?.[1]?.trim(),
            date: dateMatch?.[1]?.trim(),
            snippet: snippetMatch?.[1]?.trim(),
          });
        }
      }
    }
  }

  return { images, videos, papers, news };
}

// ============================================================================
// Composite Medical Media Section
// ============================================================================

interface MedicalMediaSectionProps {
  content: string;
}

/**
 * Renders rich medical media content extracted from markdown.
 * Displays research papers, images, videos, and news as interactive cards.
 */
export const MedicalMediaSection: React.FC<MedicalMediaSectionProps> = memo(({ content }) => {
  const { images, videos, papers, news } = useMemo(() => parseMedicalContent(content), [content]);

  const hasMedia = images.length > 0 || videos.length > 0 || papers.length > 0 || news.length > 0;
  if (!hasMedia) return null;

  return (
    <div className="space-y-4 my-3">
      {/* Research Papers */}
      {papers.length > 0 && (
        <div>
          <h4 className="flex items-center gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
            <span className="text-base">📄</span> Research Papers
            <span className="text-xs font-normal text-slate-400">({papers.length})</span>
          </h4>
          {papers.map((paper, i) => (
            <ResearchPaperCard key={i} paper={paper} />
          ))}
        </div>
      )}

      {/* Medical Images */}
      {images.length > 0 && (
        <div>
          <h4 className="flex items-center gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
            <span className="text-base">🖼️</span> Medical Images
            <span className="text-xs font-normal text-slate-400">({images.length})</span>
          </h4>
          <MedicalImageGallery images={images} />
        </div>
      )}

      {/* Medical Videos */}
      {videos.length > 0 && (
        <div>
          <h4 className="flex items-center gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
            <span className="text-base">🎬</span> Medical Videos
            <span className="text-xs font-normal text-slate-400">({videos.length})</span>
          </h4>
          <MedicalVideoGrid videos={videos} />
        </div>
      )}

      {/* Medical News */}
      {news.length > 0 && (
        <div>
          <h4 className="flex items-center gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
            <span className="text-base">📰</span> Medical News
            <span className="text-xs font-normal text-slate-400">({news.length})</span>
          </h4>
          {news.map((item, i) => (
            <MedicalNewsCard key={i} news={item} />
          ))}
        </div>
      )}
    </div>
  );
});

MedicalMediaSection.displayName = 'MedicalMediaSection';

export default MedicalMediaSection;
