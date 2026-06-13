"""CRAG Fallback - Phase 2.2 Implementation"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)



class CRAGFallback:
    """Corrective RAG: Use web search when local knowledge is insufficient."""
    
    def __init__(
        self,
        vector_store,
        web_search_tool,
        lower_threshold: float = 0.3,
        upper_threshold: float = 0.7
    ):
        """Initialize CRAG fallback."""
        self.vector_store = vector_store
        self.web_search = web_search_tool
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
    
    async def retrieve_with_fallback(
        self,
        query: str,
        k: int = 3
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve documents with web fallback.
        
        Returns:
            (documents, retrieval_method)
        """
        # Step 1: Try local vector store
        result = self.vector_store.search(query, top_k=k)
        # Handle both sync and async search methods
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            local_docs = await result
        else:
            local_docs = result
        
        # Calculate average confidence
        avg_confidence = sum(
            doc.get('score', 0) for doc in local_docs
        ) / len(local_docs) if local_docs else 0
        
        logger.info(f"Local retrieval confidence: {avg_confidence:.2f}")
        
        # Step 2: Decision based on confidence
        if avg_confidence >= self.upper_threshold:
            logger.info("Using local retrieval (high confidence)")
            return local_docs, "local"
        
        elif avg_confidence < self.lower_threshold:
            logger.info("Using web search (low local confidence)")
            web_docs = await self.web_search.search(query, num_results=k)
            return web_docs, "web"
        
        else:
            logger.info("Using hybrid local + web retrieval")
            web_docs = await self.web_search.search(query, num_results=k)
            
            combined = self._combine_results(local_docs, web_docs)
            return combined[:k], "hybrid"
    
    def _combine_results(
        self,
        local: List[Dict],
        web: List[Dict]
    ) -> List[Dict]:
        """Combine and deduplicate local + web results."""
        seen_content = set()
        combined = []
        
        for doc in local + web:
            content = doc.get('content', doc.get('text', ''))
            content_hash = hash(content[:50])
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined.append(doc)
        
        return combined
