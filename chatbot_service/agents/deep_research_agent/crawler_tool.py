import asyncio
import os
import json
import base64
import logging
import hashlib
from typing import List, Dict, Optional, Union, Set
from datetime import datetime


# External Dependencies
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# Try importing advanced modules
try:
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DeepCrawlConfig
    DEEP_CRAWL_AVAILABLE = True
except ImportError:
    DEEP_CRAWL_AVAILABLE = False

try:
    from crawl4ai.content_filter_strategy import PruningContentFilter
except ImportError:
    PruningContentFilter = None

from .models import ResearchInsight

logger = logging.getLogger(__name__)

# Configuration
BROWSER_PROFILE_DIR = os.path.join(os.getcwd(), "browser_profile")
SCREENSHOTS_DIR = os.path.join("research_outputs", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(BROWSER_PROFILE_DIR, exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai/gpt-4o")
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

# **CONCURRENCY CONTROL**: Limit concurrent browser instances to prevent OOM
# Each browser instance uses ~300MB+ RAM. Default: 3 concurrent crawlers.
MAX_CONCURRENT_CRAWLERS = int(os.getenv("MAX_CONCURRENT_CRAWLERS", "3"))
_crawler_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRAWLERS)

# Retry Configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds (exponential backoff: 1s, 2s, 4s)
PER_URL_TIMEOUT = 30  # seconds per URL

# Content Quality Thresholds
MIN_CONTENT_LENGTH = 100  # Minimum chars for valid extracted content
MIN_WORD_COUNT = 30  # Minimum words for valid content
CONTENT_QUALITY_KEYWORDS = {
    "study", "research", "findings", "treatment", "clinical", "evidence",
    "patient", "results", "analysis", "data", "methodology", "conclusion",
    "significant", "outcome", "trial", "diagnosis", "therapy", "mechanism",
    "efficacy", "safety", "adverse", "guideline", "protocol", "review",
}

# JavaScript injection for expanding hidden content
EXPAND_CONTENT_SCRIPT = """
async () => {
    // 1. Scroll to bottom to trigger lazy loading
    window.scrollTo(0, document.body.scrollHeight);
    await new Promise(r => setTimeout(r, 1000));
    
    // 2. Click common "Read More" buttons
    const buttons = document.querySelectorAll('button, a');
    for (const btn of buttons) {
        if (btn.innerText && (
            btn.innerText.toLowerCase().includes('read more') || 
            btn.innerText.toLowerCase().includes('show more') ||
            btn.innerText.toLowerCase().includes('load more'))) {
            try { btn.click(); } catch(e) {}
        }
    }
    await new Promise(r => setTimeout(r, 500));
}
"""

def _save_screenshot(url: str, screenshot_base64: str) -> Optional[str]:
    """Save base64 screenshot to file."""
    try:
        # Generate filename from URL
        safe_name = url[:20].replace('https://', '').replace('http://', '').replace('/', '_')
        filename = f"screenshot_{safe_name}_{int(datetime.now().timestamp())}.png"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        
        # Decode and save
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(screenshot_base64))
        
        return filepath
    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")
        return None


async def _crawl_with_limit(
    crawler: AsyncWebCrawler, 
    url: str, 
    config: CrawlerRunConfig,
    deep_crawl_strategy=None,
    max_retries: int = MAX_RETRIES
) -> CrawlResult:
    """
    Wrap crawler.arun with semaphore, retry logic, and timeout.
    
    **Concurrency Control**: Prevents OOM by limiting parallel crawlers.
    **Retry Logic**: Exponential backoff on transient failures.
    **Timeout**: Per-URL timeout to prevent hanging.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL to crawl
        config: CrawlerRunConfig with extraction strategy
        deep_crawl_strategy: Optional deep crawl strategy (for recursive crawling)
        max_retries: Maximum retry attempts
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            async with _crawler_semaphore:
                logger.debug(f"Acquired crawler slot ({_crawler_semaphore._value}/{MAX_CONCURRENT_CRAWLERS} available) - attempt {attempt + 1}/{max_retries}")
                
                # Wrap with per-URL timeout
                if deep_crawl_strategy:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=config, deep_crawl_strategy=deep_crawl_strategy),
                        timeout=PER_URL_TIMEOUT * 2  # Deep crawl gets extra time
                    )
                else:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=config),
                        timeout=PER_URL_TIMEOUT
                    )
                
                # Validate result quality
                if result.success:
                    return result
                elif attempt < max_retries - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"\u26a0\ufe0f Crawl returned failure for {url}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    return result  # Return the failed result on last attempt
                    
        except asyncio.TimeoutError:
            last_error = f"Timeout after {PER_URL_TIMEOUT}s"
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"\u23f0 Crawl timeout for {url}, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"\u274c Crawl timeout for {url} after {max_retries} attempts")
                
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"\u26a0\ufe0f Crawl error for {url}: {e}, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"\u274c Crawl failed for {url} after {max_retries} attempts: {e}")
    
    # Return a synthetic failed result
    class FailedResult:
        success = False
        url = url
        extracted_content = None
        markdown = None
        screenshot = None
        error_message = last_error
    
    return FailedResult()

def _validate_content_quality(content: str) -> bool:
    """
    Validate that extracted content meets minimum quality standards.
    
    Checks:
    - Minimum length
    - Minimum word count  
    - Medical/research keyword presence
    - Not just boilerplate (cookie notices, nav menus)
    
    Returns:
        True if content passes quality checks
    """
    if not content:
        return False
    
    if len(content) < MIN_CONTENT_LENGTH:
        return False
    
    words = content.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    
    # Check for boilerplate content
    boilerplate_indicators = [
        "cookie", "privacy policy", "terms of service",
        "subscribe to newsletter", "accept all cookies",
        "enable javascript", "browser not supported",
    ]
    content_lower = content.lower()
    boilerplate_score = sum(1 for bp in boilerplate_indicators if bp in content_lower)
    if boilerplate_score >= 3:  # Likely boilerplate
        return False
    
    # Check for minimum medical/research keyword presence
    content_words = set(content_lower.split())
    medical_hits = len(CONTENT_QUALITY_KEYWORDS & content_words)
    if medical_hits < 2:  # At least 2 relevant keywords
        logger.debug(f"Content quality check: only {medical_hits} relevant keywords found")
        # Don't reject outright - might be valid non-medical content
        # but flag as lower quality
    
    return True


def _fallback_plain_text_extraction(result: CrawlResult, url: str) -> Optional[ResearchInsight]:
    """
    Fallback extraction when LLM extraction fails.
    
    Uses the raw markdown/text content and creates a basic insight
    without requiring LLM processing.
    
    Args:
        result: CrawlResult with raw content
        url: Source URL
        
    Returns:
        ResearchInsight or None if content insufficient
    """
    # Try markdown first, then try any available text
    raw_text = ""
    if hasattr(result, 'markdown') and result.markdown:
        raw_text = result.markdown
    elif hasattr(result, 'cleaned_html') and result.cleaned_html:
        import re
        raw_text = re.sub(r'<[^>]+>', ' ', result.cleaned_html)
    elif hasattr(result, 'html') and result.html:
        import re
        raw_text = re.sub(r'<[^>]+>', ' ', result.html)
    
    if not raw_text or len(raw_text) < MIN_CONTENT_LENGTH:
        return None
    
    # Clean up text
    import re
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()
    
    # Extract title from first line or heading
    title = "Untitled"
    title_match = re.search(r'^#\s+(.+?)$', raw_text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()[:200]
    elif raw_text:
        title = raw_text[:100].split('.')[0].strip()
    
    # Extract summary (first substantial paragraph)
    paragraphs = [p.strip() for p in raw_text.split('\n\n') if len(p.strip()) > 50]
    summary = paragraphs[0][:500] if paragraphs else raw_text[:500]
    
    # Extract key findings (sentences with key terms)
    sentences = re.split(r'[.!?]+', raw_text)
    key_findings = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 30:
            words = set(sent.lower().split())
            if len(CONTENT_QUALITY_KEYWORDS & words) >= 2:
                key_findings.append(sent[:200])
                if len(key_findings) >= 5:
                    break
    
    if not key_findings:
        key_findings = [s.strip()[:200] for s in sentences[:3] if len(s.strip()) > 30]
    
    # Extract quotes (sentences in quotes or with key phrases)
    quotes = []
    for match in re.finditer(r'"([^"]{20,200})"', raw_text):
        quotes.append(match.group(1))
        if len(quotes) >= 3:
            break
    
    # Save screenshot if available
    screenshot_path = None
    if hasattr(result, 'screenshot') and result.screenshot:
        screenshot_path = _save_screenshot(url, result.screenshot)
    
    return ResearchInsight(
        source_url=url,
        title=title,
        summary=summary,
        key_findings=key_findings or ["Content extracted via fallback - manual review recommended"],
        relevant_quotes=quotes,
        screenshot_path=screenshot_path,
    )


def _parse_and_add_insight(result: CrawlResult, insights: List[ResearchInsight]):
    """Helper to parse result and add to insights list."""
    if not result.success or not result.extracted_content:
        # Try fallback extraction from raw content
        if result.success and not result.extracted_content:
            logger.info(f"\u26a0\ufe0f LLM extraction empty for {result.url}, trying fallback...")
            fallback = _fallback_plain_text_extraction(result, result.url)
            if fallback:
                insights.append(fallback)
                logger.info(f"\u2705 Fallback extraction successful for {result.url}")
        elif hasattr(result, 'markdown') and result.markdown:
            logger.info(f"\u26a0\ufe0f Crawl failed but has markdown for {result.url}, trying fallback...")
            fallback = _fallback_plain_text_extraction(result, getattr(result, 'url', 'unknown'))
            if fallback:
                insights.append(fallback)
                logger.info(f"\u2705 Fallback extraction from failed crawl for {result.url}")
        return

    try:
        # Save screenshot
        screenshot_path = None
        if result.screenshot:
            screenshot_path = _save_screenshot(result.url, result.screenshot)
        
        data = json.loads(result.extracted_content)
        
        # Handle list or single object
        items = data if isinstance(data, list) else [data]
        
        for item in items:
            item['source_url'] = result.url
            item['screenshot_path'] = screenshot_path
            
            # Extract links if available
            if hasattr(result, 'links'):
                item['source_links'] = [l.get('href', '') for l in result.links[:5]]
            
            # Validate content quality before adding
            summary = item.get('summary', '')
            if _validate_content_quality(summary):
                insights.append(ResearchInsight(**item))
                logger.info(f"\u2705 Extracted high-quality insight from {result.url}")
            else:
                # Still add but log the quality concern
                insights.append(ResearchInsight(**item))
                logger.warning(f"\u26a0\ufe0f Low-quality content from {result.url} - may need review")
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from {result.url}: {e}")
        # Fallback: try to extract insight from raw markdown
        fallback = _fallback_plain_text_extraction(result, result.url)
        if fallback:
            insights.append(fallback)
            logger.info(f"\u2705 Fallback extraction after JSON error for {result.url}")
    except Exception as e:
        logger.warning(f"Error processing insight from {result.url}: {e}")
        # Last resort fallback
        fallback = _fallback_plain_text_extraction(result, result.url)
        if fallback:
            insights.append(fallback)
            logger.info(f"\u2705 Fallback extraction after error for {result.url}")

async def crawl_and_extract(urls: List[str], extraction_prompt: str) -> List[ResearchInsight]:
    """
    Crawls URLs and extracts structured insights.
    Optimized for PDFs (magic=False) and HTML (magic=True).
    """
    logger.info(f"🚀 Starting crawl for {len(urls)} URLs...")
    
    # 1. Configure Browser
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent_mode="random",
        text_mode=False, # Must be False for screenshots!
        user_data_dir=BROWSER_PROFILE_DIR,
    )

    # 2. LLM Extraction Strategy
    llm_strategy = LLMExtractionStrategy(
        provider=LLM_PROVIDER,
        api_token=LLM_API_KEY,
        schema=ResearchInsight.model_json_schema(),
        extraction_type="schema",
        instruction=extraction_prompt,
        chunk_token_threshold=2000,
        overlap_rate=0.1,
        apply_chunking=True,
        input_format="markdown",
        verbose=False
    )

    insights = []
    
    # 3. Crawl with smart routing and concurrency limiting
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls:
            # Smart config: Disable magic for PDFs to save 2-3 seconds
            is_pdf = url.lower().endswith(".pdf")
            
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                magic=not is_pdf,  # ✨ Disable for PDFs
                word_count_threshold=500,
                remove_overlay_elements=True,
                process_iframes=False,
                exclude_external_links=True,
                js_code=EXPAND_CONTENT_SCRIPT if not is_pdf else None,
                wait_for="article, main, .content" if not is_pdf else None,
                screenshot=not is_pdf, # Capture screenshots only for web
                extraction_strategy=llm_strategy,
            )

            # Add Content Filter
            if PruningContentFilter and not is_pdf:
                run_config.content_filter = PruningContentFilter(
                    threshold=0.45,
                    min_word_threshold=50
                )

            try:
                logger.info(f"{'📄' if is_pdf else '🌐'} Crawling: {url}")
                # Use semaphore wrapper to limit concurrent browser instances
                result = await _crawl_with_limit(crawler, url, run_config)
                _parse_and_add_insight(result, insights)
                    
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue

    return insights

async def deep_crawl_research(
    seed_urls: List[str],
    extraction_prompt: str,
    max_depth: int = 1,
    max_pages_per_seed: int = 10,
    score_threshold: float = 0.7
) -> List[ResearchInsight]:
    """
    Perform deep research using native crawl4ai deep crawling (BFS) for Web,
    and optimized fast crawl for PDFs.
    """
    logger.info(f"🔍 Deep crawling {len(seed_urls)} seed URLs...")
    
    # Separate PDFs from regular sites
    pdf_urls = [u for u in seed_urls if u.lower().endswith('.pdf')]
    web_urls = [u for u in seed_urls if not u.lower().endswith('.pdf')]
    
    insights = []
    
    # Browser Config (Screenshots enabled)
    browser_config = BrowserConfig(
        headless=True,
        user_agent_mode="random",
        text_mode=False, # Must be False for screenshots!
        user_data_dir=BROWSER_PROFILE_DIR,
    )
    
    # LLM Strategy
    llm_strategy = LLMExtractionStrategy(
        provider=LLM_PROVIDER,
        api_token=LLM_API_KEY,
        schema=ResearchInsight.model_json_schema(),
        extraction_type="schema",
        instruction=extraction_prompt,
        chunk_token_threshold=2000,
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        
        # 1. Process PDFs (Fast Mode, No Magic, No Deep Crawl)
        if pdf_urls:
            logger.info(f"📄 Processing {len(pdf_urls)} PDF seeds (Fast Mode)...")
            pdf_run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                magic=False,      # CRITICAL: Disable magic for PDFs
                word_count_threshold=500,
                remove_overlay_elements=True,
                extraction_strategy=llm_strategy,
                screenshot=False, # Usually no screenshots for PDFs
            )
            
            # Run flat crawl for PDFs
            results = await crawler.arun_many(pdf_urls, config=pdf_run_config)
            
            for res in results:
                _parse_and_add_insight(res, insights)

        # 2. Process Web Pages (Magic Mode, Deep Crawl)
        if web_urls:
            logger.info(f"🌐 Deep crawling {len(web_urls)} Web seeds (BFS Strategy)...")
            
            # Web Config
            web_run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                magic=True,
                word_count_threshold=500,
                remove_overlay_elements=True,
                extraction_strategy=llm_strategy,
                screenshot=True,
            )
            
            if PruningContentFilter:
                web_run_config.content_filter = PruningContentFilter(
                    threshold=0.45,
                    min_word_threshold=50
                )
            
            if DEEP_CRAWL_AVAILABLE:
                deep_config = DeepCrawlConfig(
                    max_depth=max_depth,
                    score_threshold=score_threshold,
                    same_domain_only=True,
                    max_pages=max_pages_per_seed,
                )
                crawl_strategy = BFSDeepCrawlStrategy(config=deep_config)
                
                for seed_url in web_urls:
                    try:
                        # Use semaphore wrapper to limit concurrent browser instances
                        results = await _crawl_with_limit(
                            crawler,
                            seed_url,
                            web_run_config,
                            deep_crawl_strategy=crawl_strategy
                        )
                        
                        # Handle results (could be list or single)
                        results_list = results if isinstance(results, list) else [results]
                        for res in results_list:
                            _parse_and_add_insight(res, insights)

                    except Exception as e:
                        logger.error(f"Deep crawl failed for {seed_url}: {e}")
            else:
                # Fallback to flat crawl
                logger.warning("⚠️ Deep crawl unavailable, using simple crawl for Web URLs")
                results = await crawler.arun_many(web_urls, config=web_run_config)
                for res in results:
                    _parse_and_add_insight(res, insights)

    logger.info(f"✅ Research complete: {len(insights)} insights")
    return insights
