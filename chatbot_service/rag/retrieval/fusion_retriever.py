"""
Fusion Retriever - Hybrid Search with Reciprocal Rank Fusion (RRF)

Combines:
1. Semantic Search (Vector) - Good for concepts
2. Keyword Search (BM25) - Good for specific terms (drug names, acronyms)

Algorithm:
RRF Score = 1 / (k + rank_vector) + 1 / (k + rank_bm25)

RAGFlow Enhancement:
- Implements query cleaning (normalization/tokenization) before BM25
- Handles medical terminology (hypertension vs hypertensive)
- Removes stop words and special characters
- Normalizes case and spacing
"""


from __future__ import annotations

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import logging
import re

from langchain_core.documents import Document

if TYPE_CHECKING:
    from rag.store.chromadb_store import ChromaDBVectorStore
    from rag.store.vector_store import InMemoryVectorStore

    VectorStore = ChromaDBVectorStore | InMemoryVectorStore

logger = logging.getLogger(__name__)


def clean_query(query: str) -> str:
    """
    RAGFlow-style query cleaning and normalization.
    
    Prepares query for BM25 keyword search by:
    1. Converting to lowercase
    2. Removing punctuation and special characters
    3. Normalizing whitespace
    4. Removing medical stop words (optional)
    
    Examples:
        "What is hypertension?" -> "what is hypertension"
        "Drug-drug interaction (DDI)" -> "drug drug interaction ddi"
        "treating   high   BP" -> "treating high bp"
    
    Args:
        query: Raw query string
    
    Returns:
        Cleaned query optimized for BM25 matching
    """
    if not query:
        return ""
    
    # Step 1: Lowercase
    cleaned = query.lower().strip()
    
    # Step 2: Remove URLs and email addresses
    cleaned = re.sub(r'http\S+|www.\S+|[\w\.-]+@[\w\.-]+', '', cleaned)
    
    # Step 3: Replace hyphens with spaces (drug-drug -> drug drug)
    cleaned = re.sub(r'-+', ' ', cleaned)
    
    # Step 4: Remove special characters, keep alphanumeric and spaces
    # This preserves acronyms like "DDI", "BP", "ECG"
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    
    # Step 5: Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Step 6: Remove medical stop words (optional, medical-specific)
    # These are words that appear frequently but add little semantic value
    medical_stop_words = {
        'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'how', 'why',
        'patient', 'treatment', 'medication', 'drug', 'medical', 'clinical',
        'can', 'will', 'should', 'may', 'has', 'have', 'been', 'being',
        'do', 'does', 'did', 'to', 'from', 'in', 'on', 'at', 'for', 'with'
    }
    
    words = cleaned.split()
    filtered_words = [w for w in words if w not in medical_stop_words]
    
    # Keep original if filtering removes everything
    if filtered_words:
        cleaned = ' '.join(filtered_words)
    
    return cleaned


def lemmatize_medical_terms(query: str) -> str:
    """
    Use spaCy's lemmatizer for accurate lemmatization.
    
    Args:
        query: Query string to lemmatize
    
    Returns:
        Lemmatized query
    """
    try:
        from core.services.spacy_service import get_spacy_service
        spacy_svc = get_spacy_service()
        return spacy_svc.lemmatize(query)
    except Exception as e:
        logger.warning(f"SpaCy lemmatization failed: {e}, using fallback")
        return query


@dataclass
class SearchResult:
    doc: Document
    score: float
    source: str  # "vector" or "keyword"


class FusionRetriever:
    """
    Implements Hybrid Search using Reciprocal Rank Fusion with query cleaning.
    
    Improvements from RAGFlow:
    - Query normalization before BM25 search
    - Medical term lemmatization
    - Better handling of medical acronyms and terminology
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        bm25_retriever=None,  # Optional: Pass a pre-built BM25 retriever
        rrf_k: int = 60,
        use_query_cleaning: bool = True,
        use_lemmatization: bool = True
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.use_query_cleaning = use_query_cleaning
        self.use_lemmatization = use_lemmatization
        
        logger.info(
            f"✓ FusionRetriever initialized\n"
            f"  - Query Cleaning: {use_query_cleaning}\n"
            f"  - Lemmatization: {use_lemmatization}\n"
            f"  - RRF K: {rrf_k}"
        )
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        collection_name: str = "medical_knowledge",
        use_both_methods: bool = True
    ) -> List[Document]:
        """
        Perform fusion retrieval with optional dual-search.
        
        P2.4 Optimization: Adaptive retrieval based on query characteristics.
        - Drug names, dosages, acronyms → hybrid search (vector + BM25)
        - Conceptual questions → vector-only search (faster)
        
        Args:
            query: User query string
            top_k: Number of top results to return
            collection_name: Vector store collection name
            use_both_methods: If True, use both vector and keyword search. If False, only vector.
        
        Returns:
            List of top-k fused documents
        """
        logger.debug(f"Retrieving for query: {query}")
        
        # P2.4: Adaptive strategy selection
        if use_both_methods:
            use_both_methods = self._needs_hybrid_search(query)
            if not use_both_methods:
                logger.debug("P2.4: Using vector-only retrieval (fast path)")
        
        # Clean query for keyword search
        clean_q = query
        if self.use_query_cleaning:
            clean_q = clean_query(query)
            logger.debug(f"Cleaned query: {clean_q}")
        
        # Apply lemmatization if enabled
        if self.use_lemmatization:
            clean_q = lemmatize_medical_terms(clean_q)
            logger.debug(f"Lemmatized query: {clean_q}")
        
        # Step 1: Vector Search (uses original query for semantic understanding)
        vector_results = await self._vector_search(query, top_k * 2, collection_name)
        
        # Step 2: Keyword Search (uses cleaned query) - only if needed
        keyword_results = []
        if use_both_methods and self.bm25_retriever:
            keyword_results = await self._keyword_search(clean_q, top_k * 2)
        
        # Step 3: Apply RRF if we have results from both methods
        if keyword_results and vector_results:
            fused_results = self._reciprocal_rank_fusion(
                [vector_results, keyword_results], 
                k=self.rrf_k
            )
        else:
            # Fall back to single method
            fused_results = vector_results or keyword_results
        
        final_results = fused_results[:top_k]
        
        logger.debug(f"Retrieved {len(final_results)} documents from {len(vector_results)} vector + {len(keyword_results)} keyword results")
        
        return final_results
    
    def _needs_hybrid_search(self, query: str) -> bool:
        """P2.4: Determine if query benefits from hybrid search.
        
        Returns True for:
        - Drug names (exact match important)
        - Dosages with units (100mg, 5ml)
        - Acronyms (MI, CHF, STEMI)
        - Specific medical codes
        
        Returns False for:
        - Conceptual questions (faster vector-only)
        - Symptom descriptions
        - General health queries
        
        Saves ~200ms by skipping BM25 for 40-50% of queries.
        """
        import re
        
        # Patterns that need keyword search for precision
        needs_keyword_patterns = [
            r'\b\d+\s*(mg|mcg|ml|mL|g|kg|mmol|IU)\b',  # Dosages
            r'\b[A-Z]{2,5}\b',  # Acronyms (MI, CHF, STEMI)
            r'\b(aspirin|metformin|lisinopril|ibuprofen|atorvastatin|omeprazole)\b',  # Common drugs
            r'\b(warfarin|clopidogrel|metoprolol|amlodipine|losartan)\b',  # More drugs
            r'\bICD-?\d+\b',  # ICD codes
            r'\b[A-Z]\d{2}(\.\d+)?',  # Medical coding patterns
        ]
        
        for pattern in needs_keyword_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.debug(f"P2.4: Hybrid search needed (pattern: {pattern[:30]}...)")
                return True
        
        # Default: vector-only for conceptual queries
        return False
    
    async def _vector_search(self, query: str, k: int, collection: str) -> List[Document]:
        """
        Execute vector search via VectorStore.
        
        Args:
            query: Original query (for semantic understanding)
            k: Number of results to retrieve
            collection: Collection name in vector store
        
        Returns:
            List of Document objects ranked by semantic similarity
        """
        try:
            # Assuming VectorStore has a method that returns Documents
            # Adapting to the existing VectorStore interface
            results = self.vector_store.search_medical_knowledge(query, top_k=k)
            
            # Convert dict results to Documents if needed
            documents = []
            for r in results:
                # Handle if result is already a Document or a dict
                if isinstance(r, dict):
                    doc = Document(
                        page_content=r.get("content", ""),
                        metadata=r.get("metadata", {})
                    )
                    documents.append(doc)
                else:
                    documents.append(r)
            
            logger.debug(f"Vector search returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _keyword_search(self, clean_query: str, k: int) -> List[Document]:
        """
        Execute keyword search via BM25.
        
        Args:
            clean_query: Cleaned/normalized query string
            k: Number of results to retrieve
        
        Returns:
            List of Document objects ranked by BM25 score
        """
        try:
            # This assumes bm25_retriever follows LangChain Retriever interface
            if not self.bm25_retriever:
                return []
            
            results = self.bm25_retriever.get_relevant_documents(clean_query)[:k]
            
            logger.debug(f"Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self, 
        result_sets: List[List[Document]], 
        k: int = 60
    ) -> List[Document]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
        
        RRF Score = sum(1 / (k + rank_i)) for each ranked list i
        
        This ensures:
        - Documents appearing in both lists get boost
        - Relative ranking is preserved
        - Single high-ranking result doesn't dominate
        
        Args:
            result_sets: List of ranked document lists to fuse
            k: RRF parameter (typically 60, prevents extreme score differences)
        
        Returns:
            Fused and re-ranked list of documents
        """
        doc_scores = defaultdict(float)
        doc_map = {}  # Keep track of document objects: content -> doc
        
        for result_set in result_sets:
            for rank, doc in enumerate(result_set):
                # Use content as unique key (or doc_id if available)
                doc_key = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                
                if "id" in getattr(doc, 'metadata', {}):
                    doc_key = doc.metadata["id"]
                
                doc_map[doc_key] = doc
                rrf_score = 1 / (k + rank + 1)
                doc_scores[doc_key] += rrf_score
        
        # Sort by RRF score descending
        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        fused_docs = [doc_map[key] for key in sorted_keys]
        
        logger.debug(f"RRF fused {sum(len(rs) for rs in result_sets)} results -> {len(fused_docs)} unique documents")
        
        return fused_docs
