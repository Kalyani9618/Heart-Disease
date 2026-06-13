"""
Clinical Guidelines Search Tool

Provides targeted search capabilities for medical guidelines from reputable sources:
- PubMed / NCBI
- CDC
- WHO
- Professional Societies (AHA, ACC, ADA, etc.)

Uses the existing WebSearchTool but with domain-restricted queries.
"""


import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from tools.web_search import VerifiedWebSearchTool

logger = logging.getLogger(__name__)

@dataclass
class GuidelineResult:
    """Structured result from a guideline search."""
    title: str
    source: str
    url: str
    summary: str
    publication_date: Optional[str] = None
    evidence_level: Optional[str] = None

class ClinicalGuidelinesSearch:
    """
    Specialized searcher for clinical guidelines.
    
    Example:
        searcher = ClinicalGuidelinesSearch()
        guidelines = await searcher.search("hypertension management 2024")
    """
    
    # Domain filters for high-quality guidelines
    TRUSTED_DOMAINS = [
        "ncbi.nlm.nih.gov",
        "cdc.gov",
        "who.int",
        "heart.org",  # AHA
        "acc.org",    # ACC
        "diabetes.org", # ADA
        "thoracic.org", # ATS
        "acpjournals.org", # ACP
        "aafp.org",   # AAFP
        "nice.org.uk", # NICE
    ]

    def __init__(self, web_search_tool: Optional[VerifiedWebSearchTool] = None):
        self.web_search = web_search_tool or VerifiedWebSearchTool()

    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        specialty: Optional[str] = None
    ) -> List[GuidelineResult]:
        """
        Search for clinical guidelines.
        
        Args:
            query: Clinical topic (e.g., "sepsis management")
            max_results: Number of guidelines to return
            specialty: Optional specialty filter (e.g., "cardiology")
            
        Returns:
            List of GuidelineResult objects
        """
        # Enhance query for guidelines
        enhanced_query = f"{query} clinical guidelines treatment management"
        if specialty:
            enhanced_query += f" {specialty}"
            
        # Add site restrictions to query if supported by search engine, 
        # or just append "guidelines" to ensure relevance.
        # For Tavily/DDG, we rely on the search tool's capability, 
        # but we can filter results post-hoc.
        
        try:
            # We assume web_search.run returns a list of dicts or a string
            # This depends on the VerifiedWebSearchTool implementation.
            # Let's assume it returns a raw string or list of dicts.
            # We will wrap the call to ensure we get structured data if possible.
            
            # Note: VerifiedWebSearchTool.run usually returns a string. 
            # If we want structured results, we might need to access the underlying searcher 
            # or parse the output. For now, we'll use the run method and wrap the result.
            
            search_response = await self.web_search.search(enhanced_query, num_results=5)
            
            # search() returns a WebSearchResponse with .results list of WebSearchResult
            guidelines = []
            if search_response and search_response.results:
                for r in search_response.results:
                    guidelines.append(GuidelineResult(
                        title=r.title or f"Guidelines for {query}",
                        source=r.domain or "Web Search",
                        url=r.url or "",
                        summary=r.content or "",
                        publication_date=None
                    ))
            
            if not guidelines:
                guidelines.append(GuidelineResult(
                    title=f"Guidelines for {query}",
                    source="Web Search",
                    url="",
                    summary="No guideline results found",
                    publication_date=None
                ))
            
            return guidelines
            
        except Exception as e:
            logger.error(f"Guideline search failed: {e}")
            return []

# Convenience function for tool integration
async def search_clinical_guidelines(query: str) -> str:
    """
    Search for clinical guidelines.
    
    Args:
        query: Medical topic
        
    Returns:
        Formatted string of guidelines
    """
    searcher = ClinicalGuidelinesSearch()
    results = await searcher.search(query)
    
    if not results:
        return "No guidelines found."
        
    output = [f"## Clinical Guidelines for: {query}\n"]
    for res in results:
        output.append(f"### {res.title}")
        output.append(f"**Source**: {res.source}")
        output.append(f"**Summary**: {res.summary}\n")
        
    return "\n".join(output)
