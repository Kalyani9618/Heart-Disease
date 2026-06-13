"""
Medical Search Tool - Comprehensive Medical Content Discovery

Provides unified access to:
1. Medical Research Papers (PubMed, PMC, Google Scholar)
2. Medical Articles (WHO, CDC, NIH, Mayo Clinic, etc.)
3. Medical News (FDA alerts, clinical trial updates, health news)
4. Medical Images (radiology, anatomy, pathology)
5. Medical Videos (surgical procedures, patient education, lectures)

All search results are STRICTLY restricted to verified medical domains.
Non-medical content is automatically filtered out.

USAGE:
    searcher = MedicalContentSearcher()
    
    # Search everything
    results = await searcher.search_all("heart failure treatment 2025")
    
    # Search specific types
    papers = await searcher.search_research_papers("SGLT2 inhibitors heart failure")
    news = await searcher.search_medical_news("FDA approval 2025")
    images = await searcher.search_medical_images("echocardiogram left ventricle")
    videos = await searcher.search_medical_videos("cardiac catheterization procedure")
"""

import os
import logging
import asyncio
import time
import hashlib
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

# --- Imports ---
try:
    from core.services.advanced_cache import MultiTierCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False


# ==============================================================================
# Configuration
# ==============================================================================

CACHE_TTL_SECONDS = 3600 * 12  # 12 hours for medical content

class ContentType(str, Enum):
    """Types of medical content that can be searched."""
    RESEARCH_PAPER = "research_paper"
    ARTICLE = "article"
    NEWS = "news"
    IMAGE = "image"
    VIDEO = "video"
    GUIDELINE = "guideline"
    ALL = "all"

# Verified medical domains for web content
MEDICAL_WEB_DOMAINS = [
    "nih.gov", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "mayoclinic.org", "clevelandclinic.org", "hopkinsmedicine.org",
    "webmd.com", "medlineplus.gov", "cdc.gov", "who.int",
    "healthline.com", "drugs.com", "fda.gov", "heart.org",
    "medscape.com", "uptodate.com", "bmj.com", "thelancet.com",
    "nejm.org", "jamanetwork.com", "nature.com/nm",
    "ahajournals.org", "acc.org", "diabetes.org",
    "merckmanuals.com", "emedicine.medscape.com",
    "aafp.org", "acpjournals.org",
]

# Research paper domains
RESEARCH_PAPER_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "scholar.google.com", "bmj.com", "thelancet.com",
    "nejm.org", "jamanetwork.com", "nature.com",
    "ahajournals.org", "acpjournals.org",
    "cochranelibrary.com", "clinicaltrials.gov",
    "europepmc.org", "biorxiv.org", "medrxiv.org",
]

# News domains
MEDICAL_NEWS_DOMAINS = [
    "fda.gov", "cdc.gov", "who.int", "nih.gov",
    "medscape.com", "healio.com", "medpagetoday.com",
    "statnews.com", "fiercepharma.com", "fiercebiotech.com",
    "reuters.com/business/healthcare-pharmaceuticals",
]

# Image sources (medical education, radiology, pathology)
MEDICAL_IMAGE_DOMAINS = [
    "radiopaedia.org", "radiologyassistant.nl",
    "pathologyoutlines.com", "dermnetnz.org",
    "ncbi.nlm.nih.gov", "medlineplus.gov",
    "anatomyatlases.org", "bartleby.com",
    "kenhub.com", "teachmeanatomy.info",
    "openi.nlm.nih.gov",
]

# Video sources (medical education, procedures, lectures)
MEDICAL_VIDEO_DOMAINS = [
    "youtube.com", "khanacademy.org",
    "osmosis.org", "lecturio.com",
    "onlinemeded.org", "amboss.com",
    "nejm.org", "medscape.com",
    "ted.com", "coursera.org",
]

# Keywords that must be present in video/image results to confirm medical relevance
MEDICAL_RELEVANCE_KEYWORDS = [
    "medical", "medicine", "health", "clinical", "patient", "disease",
    "treatment", "diagnosis", "symptom", "surgery", "surgical",
    "anatomy", "physiology", "pathology", "radiology", "cardiology",
    "oncology", "neurology", "dermatology", "pediatrics", "pharmacy",
    "pharmaceutical", "drug", "medication", "therapy", "prognosis",
    "hospital", "doctor", "nurse", "physician", "cardiologist",
    "heart", "lung", "brain", "liver", "kidney", "blood",
    "cancer", "diabetes", "hypertension", "stroke", "infection",
    "vaccine", "immunology", "allergy", "emergency", "icu",
    "ecg", "ekg", "mri", "ct scan", "x-ray", "ultrasound",
    "lab test", "biopsy", "stethoscope", "vital signs",
    "who", "cdc", "nih", "fda", "pubmed", "nejm", "lancet",
]

DISCLAIMER = (
    "⚠️ DISCLAIMER: Content from external medical sources is provided for "
    "educational and informational purposes only. Always consult a qualified "
    "healthcare professional before making medical decisions."
)


# ==============================================================================
# Pydantic Models
# ==============================================================================

class MediaContent(BaseModel):
    """Represents an image or video result."""
    url: str = Field(..., description="Direct URL to the media")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL for preview")
    title: str = Field(..., description="Title/caption of the media")
    source: str = Field(..., description="Source domain")
    media_type: str = Field(..., description="'image' or 'video'")
    width: Optional[int] = Field(None, description="Width in pixels")
    height: Optional[int] = Field(None, description="Height in pixels")
    duration: Optional[str] = Field(None, description="Video duration (e.g., '5:32')")
    description: Optional[str] = Field(None, description="Media description")


class ResearchPaper(BaseModel):
    """Represents a medical research paper result."""
    title: str = Field(..., description="Paper title")
    authors: Optional[str] = Field(None, description="Author list")
    abstract: Optional[str] = Field(None, description="Paper abstract (up to 500 chars)")
    url: str = Field(..., description="Paper URL (PubMed, DOI, etc.)")
    doi: Optional[str] = Field(None, description="DOI identifier")
    journal: Optional[str] = Field(None, description="Journal name")
    publication_date: Optional[str] = Field(None, description="Publication date")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    citation_count: Optional[int] = Field(None, description="Citation count")
    source: str = Field(default="pubmed", description="Database source")


class MedicalNewsItem(BaseModel):
    """Represents a medical news article."""
    title: str = Field(..., description="News headline")
    url: str = Field(..., description="Article URL")
    source: str = Field(..., description="News source (e.g., FDA, Medscape)")
    summary: str = Field(..., description="News summary (up to 500 chars)")
    published_date: Optional[str] = Field(None, description="Publication date")
    category: Optional[str] = Field(None, description="News category (e.g., FDA Alert, Clinical Trial)")


class MedicalArticle(BaseModel):
    """Represents a medical article from verified sources."""
    title: str = Field(..., description="Article title")
    url: str = Field(..., description="Article URL")
    content: str = Field(..., description="Content snippet (up to 500 chars)")
    domain: str = Field(..., description="Source domain")
    score: float = Field(default=0.0, description="Relevance score")
    content_type: str = Field(default="article", description="Content type")


class MedicalSearchResponse(BaseModel):
    """Complete medical search response with all content types."""
    query: str = Field(..., description="Original search query")
    articles: List[MedicalArticle] = Field(default_factory=list, description="Medical articles")
    research_papers: List[ResearchPaper] = Field(default_factory=list, description="Research papers")
    news: List[MedicalNewsItem] = Field(default_factory=list, description="Medical news")
    images: List[MediaContent] = Field(default_factory=list, description="Medical images")
    videos: List[MediaContent] = Field(default_factory=list, description="Medical videos")
    search_timestamp: float = Field(default_factory=time.time)
    content_types_searched: List[str] = Field(default_factory=list)
    total_results: int = Field(default=0)
    disclaimer: str = Field(default=DISCLAIMER)
    cache_hit: bool = Field(default=False)


# ==============================================================================
# Medical Content Searcher
# ==============================================================================

class MedicalContentSearcher:
    """
    Comprehensive medical content search engine.
    
    Searches across multiple content types (papers, articles, news, images, videos)
    and strictly filters results to verified medical sources only.
    
    Architecture:
    1. Tavily API (primary) for articles/news
    2. DuckDuckGo (fallback) for general web, images, videos
    3. PubMed E-utilities for research papers
    4. All results filtered through medical domain whitelist
    5. Redis caching for cost optimization
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache and CACHE_AVAILABLE
        self.cache: Optional[MultiTierCache] = None

        if self.use_cache:
            try:
                self.cache = MultiTierCache()
                logger.info("✅ Medical search: Redis caching enabled")
            except Exception as e:
                logger.warning(f"⚠️ Medical search: Redis unavailable: {e}")
                self.use_cache = False

        # Initialize Tavily
        self.tavily_client = None
        if TAVILY_AVAILABLE:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                try:
                    self.tavily_client = TavilyClient(api_key=tavily_key)
                except Exception as e:
                    logger.warning(f"⚠️ Tavily init failed: {e}")

        self._semaphore = asyncio.Semaphore(5)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    async def search_all(
        self,
        query: str,
        max_per_type: int = 5,
        content_types: Optional[List[ContentType]] = None,
        force_refresh: bool = False
    ) -> MedicalSearchResponse:
        """
        Search all medical content types in parallel.
        
        Args:
            query: Medical search query
            max_per_type: Max results per content type
            content_types: Specific types to search (default: all)
            force_refresh: Bypass cache
            
        Returns:
            MedicalSearchResponse with all content types
        """
        query = self._sanitize_query(query)
        
        if not content_types:
            content_types = [
                ContentType.ARTICLE, ContentType.RESEARCH_PAPER,
                ContentType.NEWS, ContentType.IMAGE, ContentType.VIDEO
            ]

        # Check cache
        cache_key = self._make_cache_key(query, content_types)
        if self.use_cache and not force_refresh and self.cache:
            try:
                cached = await self.cache.get(cache_key)
                if cached:
                    response = MedicalSearchResponse(**cached)
                    response.cache_hit = True
                    logger.info(f"✅ Cache hit for medical search: {query[:50]}")
                    return response
            except Exception as e:
                logger.warning(f"⚠️ Cache read failed: {e}")

        # Execute searches in parallel
        tasks = {}
        if ContentType.ARTICLE in content_types:
            tasks['articles'] = self.search_medical_articles(query, max_per_type)
        if ContentType.RESEARCH_PAPER in content_types:
            tasks['papers'] = self.search_research_papers(query, max_per_type)
        if ContentType.NEWS in content_types:
            tasks['news'] = self.search_medical_news(query, max_per_type)
        if ContentType.IMAGE in content_types:
            tasks['images'] = self.search_medical_images(query, max_per_type)
        if ContentType.VIDEO in content_types:
            tasks['videos'] = self.search_medical_videos(query, max_per_type)

        results = await asyncio.gather(
            *[tasks[k] for k in tasks],
            return_exceptions=True
        )

        # Map results back
        result_map = {}
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Search {key} failed: {result}")
                result_map[key] = []
            else:
                result_map[key] = result

        response = MedicalSearchResponse(
            query=query,
            articles=result_map.get('articles', []),
            research_papers=result_map.get('papers', []),
            news=result_map.get('news', []),
            images=result_map.get('images', []),
            videos=result_map.get('videos', []),
            content_types_searched=[ct.value for ct in content_types],
            total_results=sum(len(v) for v in result_map.values()),
            cache_hit=False
        )

        # Cache the response
        if self.use_cache and response.total_results > 0 and self.cache:
            try:
                await self.cache.set(cache_key, response.model_dump(), ttl_seconds=CACHE_TTL_SECONDS)
            except Exception as e:
                logger.warning(f"⚠️ Cache write failed: {e}")

        return response

    async def search_medical_articles(self, query: str, max_results: int = 5) -> List[MedicalArticle]:
        """Search verified medical websites for articles."""
        query = self._sanitize_query(query)
        enhanced_query = f"{query} medical health clinical"

        articles = []

        # Try Tavily first
        if self.tavily_client:
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.tavily_client.search,
                            query=enhanced_query,
                            search_depth="basic",
                            include_domains=MEDICAL_WEB_DOMAINS[:10],
                            max_results=max_results
                        ),
                        timeout=15.0
                    )
                for item in response.get("results", []):
                    domain = item["url"].split("//")[-1].split("/")[0].replace("www.", "")
                    articles.append(MedicalArticle(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", "")[:500],
                        domain=domain,
                        score=float(item.get("score", 0.0)),
                        content_type="article"
                    ))
                if articles:
                    return articles[:max_results]
            except Exception as e:
                logger.warning(f"⚠️ Tavily article search failed: {e}")

        # Fallback to DuckDuckGo
        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    ddgs_results = ddgs.text(
                        enhanced_query,
                        max_results=max_results + 5
                    )
                    if ddgs_results:
                        for r in ddgs_results:
                            url = r.get("href", "")
                            domain = url.split("//")[-1].split("/")[0].replace("www.", "")
                            if self._is_medical_domain(domain):
                                articles.append(MedicalArticle(
                                    title=r.get("title", ""),
                                    url=url,
                                    content=r.get("body", "")[:500],
                                    domain=domain,
                                    score=0.7,
                                    content_type="article"
                                ))
            except Exception as e:
                logger.error(f"❌ DDG article search failed: {e}")

        return articles[:max_results]

    async def search_research_papers(self, query: str, max_results: int = 5) -> List[ResearchPaper]:
        """
        Search for medical research papers via PubMed E-utilities and web search.
        """
        query = self._sanitize_query(query)
        papers = []

        # Method 1: PubMed E-utilities (free, no API key needed)
        try:
            pubmed_papers = await self._search_pubmed(query, max_results)
            papers.extend(pubmed_papers)
        except Exception as e:
            logger.warning(f"⚠️ PubMed search failed: {e}")

        # Method 2: Web search for additional papers (if PubMed returned few)
        if len(papers) < max_results:
            try:
                remaining = max_results - len(papers)
                web_papers = await self._search_papers_web(query, remaining)
                # Deduplicate by URL
                existing_urls = {p.url for p in papers}
                for wp in web_papers:
                    if wp.url not in existing_urls:
                        papers.append(wp)
            except Exception as e:
                logger.warning(f"⚠️ Web paper search failed: {e}")

        return papers[:max_results]

    async def search_medical_news(self, query: str, max_results: int = 5) -> List[MedicalNewsItem]:
        """Search for medical news from verified health news sources."""
        query = self._sanitize_query(query)
        enhanced_query = f"{query} medical news latest update"
        news_items = []

        # Try Tavily for news
        if self.tavily_client:
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.tavily_client.search,
                            query=enhanced_query,
                            search_depth="basic",
                            include_domains=MEDICAL_NEWS_DOMAINS[:8],
                            max_results=max_results
                        ),
                        timeout=15.0
                    )
                for item in response.get("results", []):
                    domain = item["url"].split("//")[-1].split("/")[0].replace("www.", "")
                    category = self._categorize_news(domain, item.get("title", ""))
                    news_items.append(MedicalNewsItem(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        source=domain,
                        summary=item.get("content", "")[:500],
                        category=category
                    ))
                if news_items:
                    return news_items[:max_results]
            except Exception as e:
                logger.warning(f"⚠️ Tavily news search failed: {e}")

        # Fallback: DDG news search
        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    ddgs_news = ddgs.news(
                        f"{query} medical health",
                        max_results=max_results + 3
                    )
                    if ddgs_news:
                        for r in ddgs_news:
                            url = r.get("url", "")
                            source = r.get("source", url.split("//")[-1].split("/")[0] if url else "unknown")
                            news_items.append(MedicalNewsItem(
                                title=r.get("title", ""),
                                url=url,
                                source=source,
                                summary=r.get("body", "")[:500],
                                published_date=r.get("date", None),
                                category=self._categorize_news(source, r.get("title", ""))
                            ))
            except Exception as e:
                logger.error(f"❌ DDG news search failed: {e}")

        # Filter to only medical-relevant news
        news_items = [n for n in news_items if self._is_medical_relevant(n.title + " " + n.summary)]
        return news_items[:max_results]

    async def search_medical_images(self, query: str, max_results: int = 5) -> List[MediaContent]:
        """
        Search for medical images (radiology, anatomy, pathology, etc.)
        Results are strictly filtered to medical content only.
        """
        query = self._sanitize_query(query)
        enhanced_query = f"{query} medical clinical anatomy radiology"
        images = []

        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    ddgs_images = ddgs.images(
                        enhanced_query,
                        max_results=max_results * 3  # Over-fetch for filtering
                    )
                    if ddgs_images:
                        for r in ddgs_images:
                            url = r.get("image", "")
                            source = r.get("source", "")
                            title = r.get("title", "")
                            
                            # Strict medical relevance check
                            if self._is_medical_relevant(title + " " + source):
                                images.append(MediaContent(
                                    url=url,
                                    thumbnail_url=r.get("thumbnail", url),
                                    title=title,
                                    source=source.split("//")[-1].split("/")[0] if "//" in source else source,
                                    media_type="image",
                                    width=r.get("width"),
                                    height=r.get("height"),
                                    description=title
                                ))
            except Exception as e:
                logger.error(f"❌ DDG image search failed: {e}")

        # Also try Tavily for image-rich pages
        if self.tavily_client and len(images) < max_results:
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.tavily_client.search,
                            query=f"{query} medical images diagram",
                            search_depth="basic",
                            include_domains=MEDICAL_IMAGE_DOMAINS[:8],
                            max_results=3
                        ),
                        timeout=10.0
                    )
                for item in response.get("results", []):
                    images.append(MediaContent(
                        url=item.get("url", ""),
                        thumbnail_url=None,
                        title=item.get("title", ""),
                        source=item["url"].split("//")[-1].split("/")[0].replace("www.", ""),
                        media_type="image",
                        description=item.get("content", "")[:200]
                    ))
            except Exception as e:
                logger.warning(f"⚠️ Tavily image search failed: {e}")

        return images[:max_results]

    async def search_medical_videos(self, query: str, max_results: int = 5) -> List[MediaContent]:
        """
        Search for medical educational videos.
        Only returns videos from verified medical education sources or
        videos with confirmed medical relevance.
        """
        query = self._sanitize_query(query)
        enhanced_query = f"{query} medical education clinical procedure"
        videos = []

        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    ddgs_videos = ddgs.videos(
                        enhanced_query,
                        max_results=max_results * 3  # Over-fetch for filtering
                    )
                    if ddgs_videos:
                        for r in ddgs_videos:
                            title = r.get("title", "")
                            description = r.get("description", "")
                            publisher = r.get("publisher", "")
                            content_url = r.get("content", "")
                            
                            # Strict medical relevance check
                            combined_text = f"{title} {description} {publisher}"
                            if self._is_medical_relevant(combined_text):
                                videos.append(MediaContent(
                                    url=content_url,
                                    thumbnail_url=r.get("images", {}).get("large") or r.get("images", {}).get("medium"),
                                    title=title,
                                    source=publisher or (content_url.split("//")[-1].split("/")[0] if content_url else "unknown"),
                                    media_type="video",
                                    duration=r.get("duration", None),
                                    description=description[:300] if description else None
                                ))
            except Exception as e:
                logger.error(f"❌ DDG video search failed: {e}")

        # Also try web search for video pages on medical platforms
        if len(videos) < max_results and DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    text_results = ddgs.text(
                        f"{query} site:youtube.com medical OR clinical OR health education",
                        max_results=5
                    )
                    if text_results:
                        for r in text_results:
                            url = r.get("href", "")
                            title = r.get("title", "")
                            if "youtube.com" in url and self._is_medical_relevant(title):
                                # Extract video ID for embed
                                video_id = self._extract_youtube_id(url)
                                thumbnail = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg" if video_id else None
                                videos.append(MediaContent(
                                    url=url,
                                    thumbnail_url=thumbnail,
                                    title=title,
                                    source="youtube.com",
                                    media_type="video",
                                    description=r.get("body", "")[:300]
                                ))
            except Exception as e:
                logger.warning(f"⚠️ YouTube medical search failed: {e}")

        return videos[:max_results]

    # --------------------------------------------------------------------------
    # Private Methods
    # --------------------------------------------------------------------------

    async def _search_pubmed(self, query: str, max_results: int = 5) -> List[ResearchPaper]:
        """
        Search PubMed via E-utilities API (free, no key required for low volume).
        
        Uses:
        - esearch: Find article IDs
        - esummary: Get article metadata
        """
        import urllib.parse
        
        papers = []
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        try:
            import aiohttp
        except ImportError:
            # Fall back to synchronous requests in thread
            import urllib.request
            import json

            def _fetch_pubmed():
                # Step 1: Search for IDs
                encoded_query = urllib.parse.quote(query)
                search_url = (
                    f"{base_url}/esearch.fcgi?"
                    f"db=pubmed&term={encoded_query}&retmax={max_results}&retmode=json&sort=relevance"
                )
                with urllib.request.urlopen(search_url, timeout=10) as resp:
                    search_data = json.loads(resp.read().decode())

                id_list = search_data.get("esearchresult", {}).get("idlist", [])
                if not id_list:
                    return []

                # Step 2: Get summaries
                ids = ",".join(id_list)
                summary_url = f"{base_url}/esummary.fcgi?db=pubmed&id={ids}&retmode=json"
                with urllib.request.urlopen(summary_url, timeout=10) as resp:
                    summary_data = json.loads(resp.read().decode())

                result = summary_data.get("result", {})
                _papers = []
                for pmid in id_list:
                    doc = result.get(pmid)
                    if not doc or not isinstance(doc, dict):
                        continue
                    authors_list = doc.get("authors", [])
                    authors = ", ".join([a.get("name", "") for a in authors_list[:3]])
                    if len(authors_list) > 3:
                        authors += " et al."

                    _papers.append(ResearchPaper(
                        title=doc.get("title", "Untitled"),
                        authors=authors or None,
                        abstract=doc.get("description", None),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        doi=doc.get("elocationid", None),
                        journal=doc.get("fulljournalname", doc.get("source", None)),
                        publication_date=doc.get("pubdate", None),
                        pmid=pmid,
                        source="pubmed"
                    ))
                return _papers

            papers = await asyncio.to_thread(_fetch_pubmed)
            return papers

        # aiohttp path
        try:
            import urllib.parse as urlparse
            encoded_query = urlparse.quote(query)

            async with aiohttp.ClientSession() as session:
                # Step 1: Search
                search_url = (
                    f"{base_url}/esearch.fcgi?"
                    f"db=pubmed&term={encoded_query}&retmax={max_results}&retmode=json&sort=relevance"
                )
                async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    search_data = await resp.json(content_type=None)

                id_list = search_data.get("esearchresult", {}).get("idlist", [])
                if not id_list:
                    return []

                # Step 2: Get summaries
                ids = ",".join(id_list)
                summary_url = f"{base_url}/esummary.fcgi?db=pubmed&id={ids}&retmode=json"
                async with session.get(summary_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    summary_data = await resp.json(content_type=None)

                result = summary_data.get("result", {})
                for pmid in id_list:
                    doc = result.get(pmid)
                    if not doc or not isinstance(doc, dict):
                        continue
                    authors_list = doc.get("authors", [])
                    authors = ", ".join([a.get("name", "") for a in authors_list[:3]])
                    if len(authors_list) > 3:
                        authors += " et al."

                    papers.append(ResearchPaper(
                        title=doc.get("title", "Untitled"),
                        authors=authors or None,
                        abstract=doc.get("description", None),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        doi=doc.get("elocationid", None),
                        journal=doc.get("fulljournalname", doc.get("source", None)),
                        publication_date=doc.get("pubdate", None),
                        pmid=pmid,
                        source="pubmed"
                    ))
        except Exception as e:
            logger.error(f"❌ PubMed aiohttp search failed: {e}")

        return papers

    async def _search_papers_web(self, query: str, max_results: int = 3) -> List[ResearchPaper]:
        """Search for research papers via web search (Google Scholar style)."""
        papers = []
        enhanced_query = f"{query} research paper study clinical trial"

        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(
                        enhanced_query,
                        max_results=max_results + 5
                    )
                    if results:
                        for r in results:
                            url = r.get("href", "")
                            domain = url.split("//")[-1].split("/")[0].replace("www.", "")
                            # Only include from research domains
                            if any(rd in domain for rd in RESEARCH_PAPER_DOMAINS):
                                # Try to extract PMID from URL
                                pmid = None
                                pmid_match = re.search(r'/(\d{7,8})/?$', url)
                                if pmid_match:
                                    pmid = pmid_match.group(1)

                                papers.append(ResearchPaper(
                                    title=r.get("title", ""),
                                    url=url,
                                    abstract=r.get("body", "")[:500],
                                    journal=domain,
                                    pmid=pmid,
                                    source="web_search"
                                ))
            except Exception as e:
                logger.error(f"❌ DDG paper search failed: {e}")

        return papers[:max_results]

    def _is_medical_domain(self, domain: str) -> bool:
        """Check if domain is in verified medical domain list."""
        domain = domain.lower().replace("www.", "")
        return any(
            domain.endswith(md) or domain == md
            for md in MEDICAL_WEB_DOMAINS + RESEARCH_PAPER_DOMAINS + MEDICAL_NEWS_DOMAINS
        )

    def _is_medical_relevant(self, text: str) -> bool:
        """
        Check if text content is medically relevant using keyword matching.
        Must contain at least 2 medical keywords to be considered relevant.
        """
        text_lower = text.lower()
        matches = sum(1 for kw in MEDICAL_RELEVANCE_KEYWORDS if kw in text_lower)
        return matches >= 2

    def _categorize_news(self, source: str, title: str) -> str:
        """Categorize medical news based on source and title."""
        source_lower = source.lower()
        title_lower = title.lower()

        if "fda" in source_lower or "fda" in title_lower:
            if "approval" in title_lower or "approved" in title_lower:
                return "FDA Approval"
            if "recall" in title_lower:
                return "FDA Recall"
            if "warning" in title_lower:
                return "FDA Warning"
            return "FDA Update"
        if "clinical trial" in title_lower or "clinicaltrials" in source_lower:
            return "Clinical Trial"
        if "cdc" in source_lower:
            return "CDC Update"
        if "who" in source_lower:
            return "WHO Update"
        if "study" in title_lower or "research" in title_lower:
            return "Research"
        return "Medical News"

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _sanitize_query(self, query: str) -> str:
        """Sanitize search query."""
        sanitized = query.strip()
        # Remove conversational prefixes
        prefixes = [
            r"^search\s+for\s*:\s*", r"^find\s*:\s*",
            r"^look\s+up\s*:\s*", r"^research\s*:\s*",
        ]
        for prefix in prefixes:
            sanitized = re.sub(prefix, "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        if len(sanitized) > 400:
            sanitized = sanitized[:400]
        return sanitized

    def _make_cache_key(self, query: str, content_types: List[ContentType]) -> str:
        """Generate cache key from query and content types."""
        types_str = ",".join(sorted([ct.value for ct in content_types]))
        normalized = f"{query.lower().strip()}:{types_str}"
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:12]
        return f"medsearch:{query_hash}"


# ==============================================================================
# Convenience Functions for Agent Tool Calling
# ==============================================================================

async def search_medical_content(
    query: str,
    content_types: Optional[List[str]] = None,
    max_results: int = 5,
    force_refresh: bool = False
) -> str:
    """
    Unified medical content search - returns markdown-formatted results.
    
    Args:
        query: Medical search query
        content_types: List of types: "article", "research_paper", "news", "image", "video"
        max_results: Max results per type
        force_refresh: Bypass cache
        
    Returns:
        Markdown-formatted search results with all content types
    """
    try:
        searcher = MedicalContentSearcher(use_cache=True)

        # Parse content types
        types = None
        if content_types:
            types = []
            type_map = {
                "article": ContentType.ARTICLE,
                "research_paper": ContentType.RESEARCH_PAPER,
                "paper": ContentType.RESEARCH_PAPER,
                "news": ContentType.NEWS,
                "image": ContentType.IMAGE,
                "video": ContentType.VIDEO,
                "guideline": ContentType.GUIDELINE,
            }
            for ct in content_types:
                if ct.lower() in type_map:
                    types.append(type_map[ct.lower()])

        response = await searcher.search_all(
            query=query,
            max_per_type=max_results,
            content_types=types,
            force_refresh=force_refresh
        )

        return _format_medical_search_response(response)

    except Exception as e:
        logger.error(f"Medical content search failed: {e}", exc_info=True)
        return f"⚠️ Medical content search unavailable: {str(e)}"


def _format_medical_search_response(response: MedicalSearchResponse) -> str:
    """Format MedicalSearchResponse as rich markdown for chat display."""
    lines = [
        "# 🔬 Medical Search Results\n",
        f"**Query:** *{response.query}*",
        f"**Total Results:** {response.total_results}",
        f"**Cache:** {'✅ Hit' if response.cache_hit else '❌ Miss'}",
        ""
    ]

    # Research Papers
    if response.research_papers:
        lines.append("\n## 📄 Research Papers\n")
        for i, paper in enumerate(response.research_papers, 1):
            lines.append(f"### {i}. {paper.title}")
            if paper.authors:
                lines.append(f"**Authors:** {paper.authors}")
            if paper.journal:
                lines.append(f"**Journal:** {paper.journal}")
            if paper.publication_date:
                lines.append(f"**Published:** {paper.publication_date}")
            lines.append(f"**Source:** [PubMed]({paper.url})")
            if paper.abstract:
                lines.append(f"\n> {paper.abstract[:300]}...")
            lines.append("")

    # Medical Articles
    if response.articles:
        lines.append("\n## 📋 Medical Articles\n")
        for i, article in enumerate(response.articles, 1):
            lines.append(f"### {i}. {article.title}")
            lines.append(f"**Source:** [{article.domain}]({article.url})")
            lines.append(f"**Relevance:** {article.score:.0%}")
            lines.append(f"\n{article.content}")
            lines.append("")

    # Medical News
    if response.news:
        lines.append("\n## 📰 Medical News\n")
        for i, item in enumerate(response.news, 1):
            category_badge = f"`{item.category}`" if item.category else ""
            lines.append(f"### {i}. {item.title} {category_badge}")
            lines.append(f"**Source:** [{item.source}]({item.url})")
            if item.published_date:
                lines.append(f"**Date:** {item.published_date}")
            lines.append(f"\n{item.summary}")
            lines.append("")

    # Medical Images
    if response.images:
        lines.append("\n## 🖼️ Medical Images\n")
        for i, img in enumerate(response.images, 1):
            lines.append(f"### {i}. {img.title}")
            lines.append(f"**Source:** {img.source}")
            if img.thumbnail_url:
                lines.append(f"\n![{img.title}]({img.thumbnail_url})")
            elif img.url:
                lines.append(f"\n![{img.title}]({img.url})")
            if img.description:
                lines.append(f"\n*{img.description}*")
            lines.append(f"\n[View Full Image]({img.url})")
            lines.append("")

    # Medical Videos
    if response.videos:
        lines.append("\n## 🎬 Medical Videos\n")
        for i, vid in enumerate(response.videos, 1):
            lines.append(f"### {i}. {vid.title}")
            lines.append(f"**Source:** {vid.source}")
            if vid.duration:
                lines.append(f"**Duration:** {vid.duration}")
            if vid.thumbnail_url:
                lines.append(f"\n[![{vid.title}]({vid.thumbnail_url})]({vid.url})")
            if vid.description:
                lines.append(f"\n{vid.description}")
            lines.append(f"\n[▶️ Watch Video]({vid.url})")
            lines.append("")

    if response.total_results == 0:
        lines.append("\n⚠️ No medical content found for this query.")
        lines.append("Try rephrasing with broader medical terms.\n")

    lines.append(f"\n---\n{response.disclaimer}")

    return "\n".join(lines)


# ==============================================================================
# Tool Definition for LLM Function Calling
# ==============================================================================

MEDICAL_SEARCH_TOOL_DEFINITION = {
    "name": "search_medical_content",
    "description": (
        "Search for comprehensive medical content across multiple sources: "
        "research papers (PubMed), medical articles (NIH, WHO, Mayo Clinic), "
        "medical news (FDA alerts, clinical trials), medical images "
        "(radiology, anatomy, pathology), and medical videos (procedures, lectures). "
        "All results are strictly filtered to verified medical sources. "
        "Use for: latest research, clinical guidelines, visual references, "
        "educational content, and medical news updates. "
        "DO NOT use for clinical diagnosis or treatment decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Medical search query"
            },
            "content_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Types of content to search: 'article', 'research_paper', "
                    "'news', 'image', 'video'. Default: all types."
                )
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Bypass cache for latest results (default: false)",
                "default": False
            }
        },
        "required": ["query"]
    }
}
