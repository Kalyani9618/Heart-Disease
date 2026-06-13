import logging
from typing import List, Dict, Any, Optional

# Import from shared tools directory
# Note: deep_research.py ensures project root is in sys.path
try:
    from tools.web_search import VerifiedWebSearchTool
    from tools.medical_search import (
        MedicalContentSearcher, ContentType, MedicalSearchResponse,
        search_medical_content
    )
except ImportError:
    # Fallback: try relative import if running as package
    try:
        from ...tools.web_search import VerifiedWebSearchTool
        from ...tools.medical_search import (
            MedicalContentSearcher, ContentType, MedicalSearchResponse,
            search_medical_content
        )
    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import search tools: {e}")
        raise


logger = logging.getLogger(__name__)


async def get_search_results(query: str, max_results: int = 5, search_pdfs: bool = False) -> List[str]:
    """
    Adapter function to make VerifiedWebSearchTool compatible with Deep Research Agent.
    Handles 'filetype:pdf' injection for paper search.
    """
    # 1. Modify query for PDFs if requested
    final_query = query
    if search_pdfs:
        # Google/DDG operator for PDFs
        final_query += " filetype:pdf"
    
    # 2. Initialize the tool (using cache if available)
    tool = VerifiedWebSearchTool(use_cache=True)
    
    # 3. Execute Search
    try:
        response = await tool.search(final_query, num_results=max_results)
        
        # 4. Extract URLs
        urls = [result.url for result in response.results]
        
        # 5. Filter for PDFs strictly if requested (double-check)
        if search_pdfs:
            urls = [u for u in urls if u.lower().endswith('.pdf')]
            
        logger.info(f"🔍 Found {len(urls)} URLs for query: '{final_query}'")
        return urls
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


async def get_comprehensive_medical_search(
    query: str,
    max_results: int = 5,
    content_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive medical search for deep research.
    
    Returns URLs, research papers, news, images, and videos from
    verified medical sources (WHO, PubMed, FDA, CDC, NIH, etc.).
    
    Args:
        query: Medical research query
        max_results: Max results per content type
        content_types: Types to search ('article', 'research_paper', 'news', 'image', 'video')
        
    Returns:
        Dict with keys: urls, papers, news, images, videos, formatted_report
    """
    try:
        searcher = MedicalContentSearcher(use_cache=True)
        
        # Map string types to ContentType enum
        types = None
        if content_types:
            type_map = {
                "article": ContentType.ARTICLE,
                "research_paper": ContentType.RESEARCH_PAPER,
                "paper": ContentType.RESEARCH_PAPER,
                "news": ContentType.NEWS,
                "image": ContentType.IMAGE,
                "video": ContentType.VIDEO,
            }
            types = [type_map[ct] for ct in content_types if ct in type_map]
        
        response = await searcher.search_all(
            query=query,
            max_per_type=max_results,
            content_types=types
        )
        
        # Extract URLs for deep crawling
        all_urls = []
        all_urls.extend([a.url for a in response.articles])
        all_urls.extend([p.url for p in response.research_papers])
        all_urls.extend([n.url for n in response.news])
        
        # Build structured result
        result = {
            "urls": list(set(all_urls)),
            "papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": p.abstract,
                    "url": p.url,
                    "journal": p.journal,
                    "date": p.publication_date,
                    "pmid": p.pmid,
                }
                for p in response.research_papers
            ],
            "news": [
                {
                    "title": n.title,
                    "url": n.url,
                    "source": n.source,
                    "summary": n.summary,
                    "category": n.category,
                    "date": n.published_date,
                }
                for n in response.news
            ],
            "images": [
                {
                    "url": img.url,
                    "thumbnail": img.thumbnail_url,
                    "title": img.title,
                    "source": img.source,
                    "description": img.description,
                }
                for img in response.images
            ],
            "videos": [
                {
                    "url": vid.url,
                    "thumbnail": vid.thumbnail_url,
                    "title": vid.title,
                    "source": vid.source,
                    "duration": vid.duration,
                    "description": vid.description,
                }
                for vid in response.videos
            ],
            "articles": [
                {
                    "title": a.title,
                    "url": a.url,
                    "content": a.content,
                    "domain": a.domain,
                }
                for a in response.articles
            ],
            "total_results": response.total_results,
        }
        
        logger.info(
            f"🔬 Comprehensive medical search: {response.total_results} results "
            f"({len(response.research_papers)} papers, {len(response.articles)} articles, "
            f"{len(response.news)} news, {len(response.images)} images, {len(response.videos)} videos)"
        )
        return result
        
    except Exception as e:
        logger.error(f"Comprehensive medical search failed: {e}")
        return {"urls": [], "papers": [], "news": [], "images": [], "videos": [], "articles": [], "total_results": 0}
