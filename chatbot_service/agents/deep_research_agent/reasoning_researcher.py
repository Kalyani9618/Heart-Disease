"""
Reasoning Researcher - Chain-of-Thought Research Agent

Transforms linear research into a thinking, adaptive process.
Uses ThinkingAgent to reason about search strategies and retry on failure.

Performance Optimizations (v2.0):
- Search deduplication to prevent redundant API calls
- Reduced max_thinking_rounds (10 -> 4) for faster response
- Early termination when sufficient results found
- Query normalization for better cache hits
"""


import asyncio
import os
import logging
import time as _time
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import re

# Import existing components
try:
    from agents.components.thinking import ThinkingAgent, ThinkingResult
    from tools.web_search import VerifiedWebSearchTool
    from tools.medical_search import MedicalContentSearcher, ContentType
    from .search_tool import get_search_results, get_comprehensive_medical_search
    from .crawler_tool import deep_crawl_research
    from .reporter import synthesize_report
    from .models import ResearchInsight
except ImportError:
    # Direct execution fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from agents.components.thinking import ThinkingAgent, ThinkingResult
    from tools.web_search import VerifiedWebSearchTool
    from tools.medical_search import MedicalContentSearcher, ContentType
    from agents.deep_research_agent.search_tool import get_search_results, get_comprehensive_medical_search
    from agents.deep_research_agent.crawler_tool import deep_crawl_research
    from agents.deep_research_agent.reporter import synthesize_report
    from agents.deep_research_agent.models import ResearchInsight

# Observability & metrics (lazy-loaded, never breaks execution)
_tracer = None
_metrics = None


def _get_tracer():
    global _tracer
    if _tracer is None:
        try:
            from core.observability.tracing import get_tracer
            _tracer = get_tracer()
        except Exception:
            pass
    return _tracer


def _get_metrics():
    global _metrics
    if _metrics is None:
        try:
            from core.monitoring.prometheus_metrics import get_metrics
            _metrics = get_metrics()
        except Exception:
            pass
    return _metrics

logger = logging.getLogger(__name__)


# --- Source Credibility Scoring ---
SOURCE_CREDIBILITY_TIERS = {
    # Tier 1: Government & Academic (highest trust)
    "nih.gov": 0.98, "pubmed.ncbi.nlm.nih.gov": 0.97, "cdc.gov": 0.96,
    "who.int": 0.96, "fda.gov": 0.95, "medlineplus.gov": 0.94,
    "clinicaltrials.gov": 0.95, "ncbi.nlm.nih.gov": 0.96,
    # Tier 2: Major Medical Institutions
    "mayoclinic.org": 0.93, "clevelandclinic.org": 0.92,
    "hopkinsmedicine.org": 0.91, "heart.org": 0.90,
    "mountsinai.org": 0.89, "stanfordhealthcare.org": 0.89,
    # Tier 3: Medical Reference (good but commercial)
    "drugs.com": 0.82, "webmd.com": 0.80, "healthline.com": 0.78,
    "medscape.com": 0.85, "uptodate.com": 0.91,
    # Default for unknown domains
    "_default": 0.50,
}


def _get_source_credibility(url: str) -> float:
    """Get credibility score for a URL based on its domain."""
    try:
        domain = url.split("//")[-1].split("/")[0].replace("www.", "")
        for known_domain, score in SOURCE_CREDIBILITY_TIERS.items():
            if known_domain in domain:
                return score
        return SOURCE_CREDIBILITY_TIERS["_default"]
    except Exception:
        return SOURCE_CREDIBILITY_TIERS["_default"]


def _cross_validate_findings(insights: List[Dict]) -> List[Dict]:
    """
    Cross-validate findings across multiple sources.
    
    Identifies:
    - Corroborated claims (found in 2+ sources)
    - Contradictory claims
    - Unique claims (single source)
    
    Adds validation metadata to each insight.
    """
    if len(insights) < 2:
        for insight in insights:
            insight['validation_status'] = 'single_source'
            insight['corroboration_count'] = 1
        return insights
    
    # Extract key claims from each insight
    for i, insight in enumerate(insights):
        summary_lower = insight.get('summary', '').lower()
        key_points = insight.get('key_points', [])
        
        # Count how many other sources corroborate this finding
        corroboration_count = 0
        for j, other in enumerate(insights):
            if i == j:
                continue
            other_summary = other.get('summary', '').lower()
            
            # Simple keyword overlap as proxy for corroboration
            insight_words = set(summary_lower.split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'of', 'to', 'and', 'for', 'with', 'on', 'at'}
            other_words = set(other_summary.split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'of', 'to', 'and', 'for', 'with', 'on', 'at'}
            
            if len(insight_words) > 0:
                overlap = len(insight_words & other_words) / len(insight_words)
                if overlap > 0.3:  # 30%+ keyword overlap = corroborated
                    corroboration_count += 1
        
        insight['corroboration_count'] = corroboration_count + 1  # Include self
        insight['credibility'] = _get_source_credibility(insight.get('url', ''))
        
        if corroboration_count >= 2:
            insight['validation_status'] = 'strongly_corroborated'
        elif corroboration_count >= 1:
            insight['validation_status'] = 'corroborated'
        else:
            insight['validation_status'] = 'single_source'
    
    return insights


def _normalize_query(query: str) -> str:
    """Normalize search query for deduplication."""
    # Lowercase, strip whitespace, remove extra spaces
    normalized = ' '.join(query.lower().strip().split())
    return normalized


def _query_hash(query: str) -> str:
    """Create a hash for quick query comparison."""
    return hashlib.md5(_normalize_query(query).encode()).hexdigest()[:12]


@dataclass
class ResearchSession:
    """Tracks a complete research session."""
    query: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    urls_searched: List[str] = field(default_factory=list)
    urls_crawled: List[str] = field(default_factory=list)
    insights: List[Dict] = field(default_factory=list)
    reasoning_trace: str = ""
    final_report: Optional[str] = None
    # v2.0: Track executed queries to prevent duplicates
    _executed_queries: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "started_at": self.started_at.isoformat(),
            "urls_searched": self.urls_searched,
            "urls_crawled": self.urls_crawled,
            "insights_count": len(self.insights),
            "has_report": self.final_report is not None,
            "unique_searches": len(self._executed_queries),
        }
    
    def is_query_executed(self, query: str) -> bool:
        """Check if a query has already been executed."""
        return _query_hash(query) in self._executed_queries
    
    def mark_query_executed(self, query: str) -> None:
        """Mark a query as executed."""
        self._executed_queries.add(_query_hash(query))


class ReasoningResearcher:
    """
    Chain-of-Thought Research Agent.
    
    Instead of blindly searching and crawling, this agent:
    1. Thinks about what to search for
    2. Evaluates search results
    3. Reasons about which URLs to crawl
    4. Reflects on findings and adjusts strategy
    
    v2.0 Performance Improvements:
    - Search deduplication prevents redundant API calls
    - Reduced thinking rounds (10 -> 4) for faster completion
    - Early termination when enough URLs/insights found
    - Better prompt to encourage diverse searches
    
    Example:
        researcher = ReasoningResearcher(llm)
        result = await researcher.research("Latest breakthroughs in quantum computing")
        
        print(result.reasoning_trace)  # See the agent's thinking
        print(result.final_report)     # The synthesized report
    """
    
    # v2.0: Configurable limits
    DEFAULT_MAX_ROUNDS = 4  # Reduced from 10 - usually enough for good results
    MIN_URLS_FOR_COMPLETION = 3  # Early exit threshold
    MAX_URLS_TO_COLLECT = 10  # Don't search forever
    
    def __init__(
        self,
        llm,
        max_thinking_rounds: int = DEFAULT_MAX_ROUNDS,
        verbose: bool = True,
    ):
        """
        Initialize the reasoning researcher.
        
        Args:
            llm: Language model (must support ainvoke)
            max_thinking_rounds: Maximum think-act cycles (default: 4)
            verbose: Log thinking process
        """
        self.llm = llm
        self.max_thinking_rounds = max_thinking_rounds
        self.verbose = verbose
        
        # Initialize search tool
        self.search_tool = VerifiedWebSearchTool(use_cache=True)
        
        # Research session tracking
        self.current_session: Optional[ResearchSession] = None
    
    async def research(self, query: str) -> ResearchSession:
        """
        Conduct intelligent research on a topic.
        
        The agent will:
        1. Analyze the query
        2. Search with reasoning
        3. Select URLs to crawl based on relevance
        4. Extract insights
        5. Synthesize report
        
        Args:
            query: Research topic
            
        Returns:
            ResearchSession with all findings and reasoning
        """
        logger.info(f"🧠 Starting reasoning research: {query}")
        _start = _time.perf_counter()
        
        self.current_session = ResearchSession(query=query)
        
        # Create tool wrappers that the ThinkingAgent can call
        tools = self._create_research_tools()
        
        # Initialize thinking agent
        agent = ThinkingAgent(
            llm=self.llm,
            tools=tools,
            max_thinking_rounds=self.max_thinking_rounds,
            verbose=self.verbose,
        )
        
        # Research prompt - v3.0: Structured reasoning, diverse searches, source evaluation
        research_prompt = f"""You are an expert medical research assistant conducting deep research. Your task:

"{query}"

Available tools:
1. perform_search(query, search_pdfs) - Search the web for articles or PDFs
2. search_medical_research(query, content_types) - Search medical sources (PubMed, WHO, CDC, NIH, FDA)
3. perform_deep_crawl(urls, question) - Crawl URLs to extract detailed information
4. finalize_research(summary) - Complete the research with final summary

## RESEARCH METHODOLOGY:

### Phase 1: BROAD SEARCH (2-3 different searches)
- Start with search_medical_research for authoritative medical content
- Use perform_search with different angle/keyword combinations
- Each search MUST use DIFFERENT keywords (no repeats!)
- Good strategies: add "2024"/"2025", "systematic review", "clinical trial", "guideline"

### Phase 2: DEEP EXTRACTION (after collecting 3+ URLs)
- Use perform_deep_crawl on the 2-3 most promising/authoritative URLs
- Prioritize .gov and major institution sources
- Ask specific extraction questions, not generic ones

### Phase 3: CROSS-VALIDATION & SYNTHESIS
- Compare findings across sources for consistency
- Note any contradictions or disagreements between sources
- Weight findings by source credibility:
  - Tier 1: Government/Academic (NIH, CDC, WHO, PubMed) = highest trust
  - Tier 2: Major Medical Institutions (Mayo, Cleveland Clinic) = high trust  
  - Tier 3: Medical Reference sites (WebMD, Healthline) = moderate trust
- Include relevant images/videos when available

### Phase 4: FINALIZE
- Call finalize_research with:
  - Key findings ranked by evidence strength
  - Source credibility assessment
  - Contradictions flagged
  - Clinical implications or actionable takeaways
  - Limitations of the research

## RULES:
- NEVER repeat the same search query
- Complete research within 3-4 tool calls if possible
- If sources disagree, report both perspectives with credibility context
- Always note the evidence level (systematic review > RCT > observational > expert opinion)

Begin your research now."""

        # Run the thinking agent
        result = await agent.run(research_prompt)
        
        # Store reasoning trace
        self.current_session.reasoning_trace = result.get_reasoning_trace()
        
        # Generate final report if not already done
        if not self.current_session.final_report:
            self.current_session.final_report = self._synthesize_final_report()
        
        _elapsed_ms = (_time.perf_counter() - _start) * 1000
        
        # Record tracing span
        tracer = _get_tracer()
        if tracer:
            try:
                from core.observability.tracing import SpanType
                async with tracer.trace_operation(
                    name=f"deep_research:{query[:50]}",
                    operation_type=SpanType.AGENT_STEP,
                    metadata={
                        "query": query[:200],
                        "insights_count": len(self.current_session.insights),
                        "urls_searched": len(self.current_session.urls_searched),
                        "urls_crawled": len(self.current_session.urls_crawled),
                        "latency_ms": round(_elapsed_ms, 1),
                    }
                ) as span:
                    span.add_event("research_complete", {
                        "unique_searches": len(self.current_session._executed_queries),
                        "has_report": self.current_session.final_report is not None,
                    })
            except Exception:
                pass  # Tracing must never break research
        
        # Record Prometheus metrics
        metrics = _get_metrics()
        if metrics:
            try:
                metrics.increment_counter("deep_research_executions")
                metrics.record_histogram("deep_research_latency_ms", _elapsed_ms)
            except Exception:
                pass
        
        logger.info(f"✅ Research complete: {len(self.current_session.insights)} insights in {_elapsed_ms:.0f}ms")
        
        return self.current_session
    
    def _create_research_tools(self) -> List[Callable]:
        """Create tool functions for the ThinkingAgent."""
        
        async def perform_search(query: str, search_pdfs: bool = False) -> str:
            """
            Search the web for articles or academic papers.
            
            Args:
                query: Search query (must be DIFFERENT from previous searches)
                search_pdfs: If True, search specifically for PDF papers
                
            Returns:
                Summary of search results with URLs
            """
            # v2.0: Check for duplicate query
            if self.current_session.is_query_executed(query):
                logger.info(f"🔄 Skipping duplicate search: '{query[:50]}...'")
                return (
                    f"⚠️ This search query was already executed. "
                    f"You have {len(self.current_session.urls_searched)} URLs collected. "
                    f"Please try a DIFFERENT search query with new keywords, "
                    f"or use perform_deep_crawl to extract information from existing URLs, "
                    f"or call finalize_research if you have enough information."
                )
            
            # v2.0: Check if we have enough URLs already
            if len(self.current_session.urls_searched) >= self.MAX_URLS_TO_COLLECT:
                return (
                    f"✅ Already collected {len(self.current_session.urls_searched)} URLs. "
                    f"Please use perform_deep_crawl to extract insights, "
                    f"or call finalize_research to complete."
                )
            
            try:
                # Mark query as executed
                self.current_session.mark_query_executed(query)
                
                urls = await get_search_results(
                    query, 
                    max_results=5, 
                    search_pdfs=search_pdfs
                )
                
                self.current_session.urls_searched.extend(urls)
                
                if urls:
                    result = f"Found {len(urls)} URLs (total collected: {len(self.current_session.urls_searched)}):\n"
                    for i, url in enumerate(urls, 1):
                        type_label = "[PDF]" if url.lower().endswith(".pdf") else "[WEB]"
                        result += f"  {i}. {type_label} {url}\n"
                    
                    # v2.0: Hint for early completion
                    if len(self.current_session.urls_searched) >= self.MIN_URLS_FOR_COMPLETION:
                        result += f"\n💡 You have enough URLs. Consider using perform_deep_crawl or finalize_research."
                    
                    return result
                else:
                    return "No results found. Try a different search query with different keywords."
                    
            except Exception as e:
                return f"Search error: {e}. Try a different approach."
        
        async def perform_deep_crawl(urls: str, question: str) -> str:
            """
            Crawl specific URLs to extract detailed information.
            
            Args:
                urls: Comma-separated URLs to crawl
                question: Specific question to answer from the content
                
            Returns:
                Extracted insights from the URLs
            """
            try:
                url_list = [u.strip() for u in urls.split(",")]
                
                insights = await deep_crawl_research(
                    seed_urls=url_list,
                    extraction_prompt=question,
                    max_depth=1
                )
                
                self.current_session.urls_crawled.extend(url_list)
                
                if insights:
                    result = f"Extracted {len(insights)} insights:\n\n"
                    for insight in insights[:5]:  # Limit to 5 for context window
                        self.current_session.insights.append({
                            "url": insight.url,
                            "summary": insight.summary,
                            "key_points": insight.key_points,
                        })
                        result += f"**From {insight.url}:**\n"
                        result += f"Summary: {insight.summary[:500]}...\n"
                        if insight.key_points:
                            result += f"Key points: {', '.join(insight.key_points[:3])}\n"
                        result += "\n"
                    return result
                else:
                    return "No insights extracted. The URLs may be inaccessible or empty."
                    
            except Exception as e:
                return f"Crawl error: {e}. Try different URLs."
        
        async def search_medical_research(query: str, content_types: str = "all") -> str:
            """
            Search for comprehensive medical content: research papers, articles,
            news, images, and videos from verified medical sources.
            
            Args:
                query: Medical research query (must be DIFFERENT from previous searches)
                content_types: Comma-separated types: 'article,research_paper,news,image,video' or 'all'
                
            Returns:
                Structured results with papers, news, images, and videos
            """
            # Check for duplicate query
            if self.current_session.is_query_executed(f"med_{query}"):
                return (
                    f"⚠️ This medical search query was already executed. "
                    f"Please try DIFFERENT keywords or call finalize_research."
                )
            
            try:
                self.current_session.mark_query_executed(f"med_{query}")
                
                # Parse content types
                types = None
                if content_types and content_types.lower() != "all":
                    types = [ct.strip() for ct in content_types.split(",")]
                
                result_data = await get_comprehensive_medical_search(
                    query=query,
                    max_results=5,
                    content_types=types
                )
                
                # Add URLs for crawling
                self.current_session.urls_searched.extend(result_data.get("urls", []))
                
                # Build readable result
                output_lines = [f"📊 Medical search found {result_data.get('total_results', 0)} results:\n"]
                
                # Research papers
                papers = result_data.get("papers", [])
                if papers:
                    output_lines.append(f"\n📄 **Research Papers ({len(papers)}):**")
                    for i, p in enumerate(papers[:3], 1):
                        output_lines.append(f"  {i}. {p['title']}")
                        if p.get('authors'):
                            output_lines.append(f"     Authors: {p['authors']}")
                        if p.get('journal'):
                            output_lines.append(f"     Journal: {p['journal']}")
                        output_lines.append(f"     URL: {p['url']}")
                
                # Articles
                articles = result_data.get("articles", [])
                if articles:
                    output_lines.append(f"\n📋 **Medical Articles ({len(articles)}):**")
                    for i, a in enumerate(articles[:3], 1):
                        output_lines.append(f"  {i}. {a['title']} [{a['domain']}]")
                        output_lines.append(f"     URL: {a['url']}")
                
                # News
                news = result_data.get("news", [])
                if news:
                    output_lines.append(f"\n📰 **Medical News ({len(news)}):**")
                    for i, n in enumerate(news[:3], 1):
                        cat = f" [{n['category']}]" if n.get('category') else ""
                        output_lines.append(f"  {i}. {n['title']}{cat}")
                        output_lines.append(f"     Source: {n['source']} | URL: {n['url']}")
                
                # Images
                images = result_data.get("images", [])
                if images:
                    output_lines.append(f"\n🖼️ **Medical Images ({len(images)}):**")
                    for i, img in enumerate(images[:3], 1):
                        output_lines.append(f"  {i}. {img['title']} (Source: {img['source']})")
                        output_lines.append(f"     URL: {img['url']}")
                
                # Videos
                videos = result_data.get("videos", [])
                if videos:
                    output_lines.append(f"\n🎬 **Medical Videos ({len(videos)}):**")
                    for i, vid in enumerate(videos[:3], 1):
                        dur = f" ({vid['duration']})" if vid.get('duration') else ""
                        output_lines.append(f"  {i}. {vid['title']}{dur}")
                        output_lines.append(f"     Source: {vid['source']} | URL: {vid['url']}")
                
                output_lines.append(f"\n💡 Total URLs collected: {len(self.current_session.urls_searched)}")
                
                return "\n".join(output_lines)
                
            except Exception as e:
                return f"Medical search error: {e}. Try different keywords."
        
        async def finalize_research(summary: str) -> str:
            """
            Complete the research with a final summary.
            
            Args:
                summary: Your synthesis of all findings
                
            Returns:
                Confirmation that research is complete
            """
            self.current_session.final_report = summary
            return "Research finalized. Summary saved."
        
        # Set function metadata for ThinkingAgent
        perform_search.__name__ = "perform_search"
        perform_search.name = "perform_search"
        perform_search.description = "Search the web for articles or PDFs related to the research topic"
        
        search_medical_research.__name__ = "search_medical_research"
        search_medical_research.name = "search_medical_research"
        search_medical_research.description = "Search for medical research papers, articles, news, images, and videos from verified medical sources (PubMed, WHO, CDC, NIH, FDA, journals)"
        
        perform_deep_crawl.__name__ = "perform_deep_crawl"
        perform_deep_crawl.name = "perform_deep_crawl"
        perform_deep_crawl.description = "Crawl specific URLs to extract detailed information"
        
        finalize_research.__name__ = "finalize_research"
        finalize_research.name = "finalize_research"
        finalize_research.description = "Complete the research with a final summary"
        
        return [perform_search, search_medical_research, perform_deep_crawl, finalize_research]
    
    def _synthesize_final_report(self) -> str:
        """Generate final report from collected insights with cross-validation."""
        if not self.current_session.insights:
            return "No insights were collected during research."
        
        # Cross-validate findings
        validated_insights = _cross_validate_findings(self.current_session.insights)
        
        # Sort by credibility * corroboration
        validated_insights.sort(
            key=lambda x: x.get('credibility', 0.5) * x.get('corroboration_count', 1),
            reverse=True
        )
        
        report_lines = [
            f"# Research Report: {self.current_session.query}",
            f"\n*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n",
            f"## Executive Summary\n",
            f"Analyzed {len(self.current_session.urls_crawled)} sources from "
            f"{len(self.current_session.urls_searched)} search results.\n",
        ]
        
        # Evidence quality summary
        strongly_corroborated = [i for i in validated_insights if i.get('validation_status') == 'strongly_corroborated']
        corroborated = [i for i in validated_insights if i.get('validation_status') == 'corroborated']
        single_source = [i for i in validated_insights if i.get('validation_status') == 'single_source']
        
        report_lines.append("## Evidence Quality Assessment\n")
        report_lines.append(f"- ✅ **Strongly Corroborated** (3+ sources): {len(strongly_corroborated)} findings")
        report_lines.append(f"- ✔️ **Corroborated** (2 sources): {len(corroborated)} findings")
        report_lines.append(f"- ⚠️ **Single Source**: {len(single_source)} findings\n")
        
        report_lines.append("## Key Findings\n")
        
        for i, insight in enumerate(validated_insights, 1):
            # Credibility badge
            cred = insight.get('credibility', 0.5)
            if cred >= 0.9:
                badge = "🏥 HIGH TRUST"
            elif cred >= 0.8:
                badge = "📗 MODERATE TRUST"
            else:
                badge = "📓 LOWER TRUST"
            
            validation = insight.get('validation_status', 'unknown')
            if validation == 'strongly_corroborated':
                val_badge = "✅ Strongly Corroborated"
            elif validation == 'corroborated':
                val_badge = "✔️ Corroborated"
            else:
                val_badge = "⚠️ Single Source"
            
            report_lines.append(f"### Finding {i} [{badge}] [{val_badge}]")
            report_lines.append(f"**Source:** {insight.get('url', 'Unknown')}")
            report_lines.append(f"**Credibility:** {cred:.0%} | **Corroborated by:** {insight.get('corroboration_count', 1)} source(s)\n")
            report_lines.append(insight.get('summary', '')[:1000])
            if insight.get('key_points'):
                report_lines.append("\n**Key Points:**")
                for point in insight['key_points'][:5]:
                    report_lines.append(f"- {point}")
            report_lines.append("\n")
        
        # Reasoning trace
        report_lines.append("## Research Methodology")
        report_lines.append(f"- Search queries executed: {len(self.current_session._executed_queries)}")
        report_lines.append(f"- URLs discovered: {len(self.current_session.urls_searched)}")
        report_lines.append(f"- URLs crawled: {len(self.current_session.urls_crawled)}")
        report_lines.append(f"- Insights extracted: {len(self.current_session.insights)}\n")
        
        report_lines.append("## Reasoning Trace")
        report_lines.append("```")
        report_lines.append(self.current_session.reasoning_trace[:5000])
        report_lines.append("```")
        
        report_lines.append("\n---")
        report_lines.append("*Generated by Deep Research Agent v3.0 — Source-Validated Research*")
        
        return "\n".join(report_lines)


# =============================================================================
# Convenience Function
# =============================================================================

async def run_reasoning_research(query: str, llm=None) -> ResearchSession:
    """
    Run reasoning research with default configuration.
    
    Args:
        query: Research topic
        llm: Optional LLM (will use default if not provided)
        
    Returns:
        ResearchSession with results
    """
    if llm is None:
        # Try to get default LLM
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except ImportError:
            raise ValueError("No LLM provided and langchain_openai not available")
    
    researcher = ReasoningResearcher(llm=llm)
    return await researcher.research(query)
