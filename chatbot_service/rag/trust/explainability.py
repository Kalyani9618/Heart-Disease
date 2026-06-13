"""
Explainable Retrieval - Trust Layer

Adds transparency to RAG by explaining WHY a document was retrieved.
This is critical for medical AI to build trust with clinicians and patients.

Usage:
    explainer = ExplainableRetrieval(llm_gateway)
    docs_with_explanation = await explainer.explain_relevance(query, docs)
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class ExplainableRetrieval:
    """
    Wraps retrieval results with AI-generated explanations.
    """
    
    def __init__(self, llm_gateway):
        self.llm = llm_gateway
        
    async def explain_relevance(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Add 'relevance_explanation' to document metadata.
        """
        # Optimize: Only explain top 3 documents to save latency/tokens
        docs_to_explain = documents[:3]
        remaining_docs = documents[3:]
        
        explained_docs = []
        
        for doc in docs_to_explain:
            try:
                explanation = await self._generate_explanation(query, doc.page_content)
                
                # Create new doc with updated metadata to avoid mutating original
                new_metadata = doc.metadata.copy()
                new_metadata["relevance_explanation"] = explanation
                
                explained_docs.append(Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                ))
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")
                explained_docs.append(doc)
                
        return explained_docs + remaining_docs

    async def _generate_explanation(self, query: str, content: str) -> str:
        """Generate a one-sentence explanation of relevance."""
        prompt = f"""Explain in one short sentence why this medical text is relevant to the query.
        
        Query: {query}
        Text: {content[:300]}...
        
        Explanation: This text is relevant because..."""
        
        # We want a very short, fast response
        response = await self.llm.generate(prompt, content_type="general")
        
        # Cleanup response
        if response.lower().startswith("this text is relevant because"):
            response = response[len("this text is relevant because"):].strip()
            
        return response.strip()
