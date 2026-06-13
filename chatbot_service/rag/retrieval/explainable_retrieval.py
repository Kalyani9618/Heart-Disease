"""
Explainable Retrieval Module for Medical RAG Pipeline

Provides transparency into WHY documents were retrieved and ranked,
supporting:
- Token-level highlighting of query-document matches
- Reasoning traces for retrieval decisions
- Confidence explanations
- Citation attribution

Key Features:
- Highlights matching medical terms in retrieved passages
- Explains semantic similarity scores
- Provides confidence calibration explanations
- Generates structured citation metadata
"""


import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class MatchType(Enum):
    """Types of matches between query and document."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    SYNONYM = "synonym"
    RELATED = "related"
    PHONETIC = "phonetic"


@dataclass
class TermMatch:
    """Represents a matched term between query and document."""
    query_term: str
    document_term: str
    match_type: MatchType
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""


@dataclass
class RetrievalExplanation:
    """Structured explanation of why a document was retrieved."""
    document_id: str
    rank: int
    overall_score: float
    confidence: float
    term_matches: List[TermMatch] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    highlighted_passage: str = ""
    citation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "document_id": self.document_id,
            "rank": self.rank,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "term_matches": [
                {
                    "query_term": tm.query_term,
                    "document_term": tm.document_term,
                    "match_type": tm.match_type.value,
                    "confidence": tm.confidence,
                    "context": tm.context
                }
                for tm in self.term_matches
            ],
            "reasoning_trace": self.reasoning_trace,
            "highlighted_passage": self.highlighted_passage,
            "citation": self.citation_metadata
        }


class ExplainableRetriever:
    """
    Wraps retrieval operations to provide explanations.
    
    Works with any base retriever and adds explanation layers.
    """
    
    # Medical term patterns for matching
    MEDICAL_PATTERNS = {
        "drug": r'\b(?:aspirin|metformin|lisinopril|atorvastatin|warfarin|digoxin|amlodipine|metoprolol|omeprazole|losartan|hydrochlorothiazide|gabapentin|prednisone|levothyroxine|amoxicillin)\b',
        "condition": r'\b(?:hypertension|diabetes|hyperlipidemia|heart\s*failure|arrhythmia|angina|stroke|myocardial\s*infarction|atrial\s*fibrillation|coronary\s*artery\s*disease|cardiomyopathy)\b',
        "measurement": r'\b(?:blood\s*pressure|heart\s*rate|cholesterol|glucose|HbA1c|eGFR|creatinine|potassium|sodium|BNP)\b',
        "procedure": r'\b(?:echocardiogram|angioplasty|catheterization|ablation|pacemaker|CABG|stent|bypass)\b'
    }
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.5
    LOW_CONFIDENCE = 0.3
    
    def __init__(
        self,
        base_retriever: Any = None,
        highlight_matches: bool = True,
        include_reasoning: bool = True,
        max_context_chars: int = 100
    ):
        """
        Initialize explainable retriever.
        
        Args:
            base_retriever: Underlying retriever to wrap
            highlight_matches: Whether to highlight matching terms
            include_reasoning: Whether to include reasoning traces
            max_context_chars: Max chars around matched terms for context
        """
        self.base_retriever = base_retriever
        self.highlight_matches = highlight_matches
        self.include_reasoning = include_reasoning
        self.max_context_chars = max_context_chars
        
        # Compile patterns
        self._compiled_patterns = {
            category: re.compile(pattern, re.IGNORECASE)
            for category, pattern in self.MEDICAL_PATTERNS.items()
        }
        
        logger.info(
            f"✓ ExplainableRetriever initialized "
            f"(highlight={highlight_matches}, reasoning={include_reasoning})"
        )
    
    def explain_retrieval(
        self,
        query: str,
        documents: List[Dict],
        scores: Optional[List[float]] = None
    ) -> List[RetrievalExplanation]:
        """
        Generate explanations for retrieved documents.
        
        Args:
            query: Original query string
            documents: List of retrieved documents
            scores: Optional list of retrieval scores
            
        Returns:
            List of RetrievalExplanation objects
        """
        explanations = []
        
        # Extract query terms for matching
        query_terms = self._extract_medical_terms(query)
        query_words = set(query.lower().split())
        
        for i, doc in enumerate(documents):
            # Get document content
            content = self._get_document_content(doc)
            doc_id = self._generate_doc_id(doc)
            score = scores[i] if scores and i < len(scores) else doc.get("score", doc.get("rerank_score", 0.0))
            
            # Find term matches
            term_matches = self._find_term_matches(query, query_terms, query_words, content)
            
            # Generate reasoning trace
            reasoning = []
            if self.include_reasoning:
                reasoning = self._generate_reasoning(query, doc, term_matches, score)
            
            # Highlight passage
            highlighted = ""
            if self.highlight_matches and term_matches:
                highlighted = self._highlight_passage(content, term_matches)
            
            # Calculate confidence
            confidence = self._calculate_confidence(score, term_matches)
            
            # Generate citation metadata
            citation = self._generate_citation(doc, i + 1)
            
            explanation = RetrievalExplanation(
                document_id=doc_id,
                rank=i + 1,
                overall_score=float(score),
                confidence=confidence,
                term_matches=term_matches,
                reasoning_trace=reasoning,
                highlighted_passage=highlighted,
                citation_metadata=citation
            )
            
            explanations.append(explanation)
        
        return explanations
    
    def _extract_medical_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract medical terms by category from text."""
        terms = {}
        for category, pattern in self._compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                terms[category] = [m.lower() for m in matches]
        return terms
    
    def _get_document_content(self, doc: Dict) -> str:
        """Extract content from document in various formats."""
        if isinstance(doc, dict):
            return doc.get("content") or doc.get("page_content") or doc.get("text", "")
        elif hasattr(doc, "page_content"):
            return doc.page_content
        elif hasattr(doc, "content"):
            return doc.content
        return str(doc)
    
    def _generate_doc_id(self, doc: Dict) -> str:
        """Generate a unique document ID."""
        if isinstance(doc, dict):
            if "id" in doc:
                return str(doc["id"])
            if "document_id" in doc:
                return str(doc["document_id"])
        
        # Generate hash-based ID from content
        content = self._get_document_content(doc)[:500]
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _find_term_matches(
        self,
        query: str,
        query_terms: Dict[str, List[str]],
        query_words: set,
        content: str
    ) -> List[TermMatch]:
        """Find all term matches between query and document."""
        matches = []
        content_lower = content.lower()
        
        # 1. Exact medical term matches
        for category, terms in query_terms.items():
            for term in terms:
                for match in re.finditer(re.escape(term), content_lower):
                    start, end = match.start(), match.end()
                    context = self._get_context(content, start, end)
                    matches.append(TermMatch(
                        query_term=term,
                        document_term=content[start:end],
                        match_type=MatchType.EXACT,
                        confidence=1.0,
                        start_pos=start,
                        end_pos=end,
                        context=context
                    ))
        
        # 2. Exact word matches (non-medical)
        for word in query_words:
            if len(word) > 3:  # Skip short words
                for match in re.finditer(r'\b' + re.escape(word) + r'\b', content_lower):
                    start, end = match.start(), match.end()
                    # Skip if already matched as medical term
                    if not any(m.start_pos == start for m in matches):
                        context = self._get_context(content, start, end)
                        matches.append(TermMatch(
                            query_term=word,
                            document_term=content[start:end],
                            match_type=MatchType.EXACT,
                            confidence=0.9,
                            start_pos=start,
                            end_pos=end,
                            context=context
                        ))
        
        # 3. Document medical terms (semantic matches)
        doc_terms = self._extract_medical_terms(content)
        for category, terms in doc_terms.items():
            for term in terms:
                # If query has terms in same category, it's a semantic match
                if category in query_terms and term not in query_terms[category]:
                    for match in re.finditer(re.escape(term), content_lower):
                        start, end = match.start(), match.end()
                        if not any(m.start_pos == start for m in matches):
                            context = self._get_context(content, start, end)
                            matches.append(TermMatch(
                                query_term=f"[{category}]",
                                document_term=content[start:end],
                                match_type=MatchType.RELATED,
                                confidence=0.7,
                                start_pos=start,
                                end_pos=end,
                                context=context
                            ))
        
        # Sort by position
        matches.sort(key=lambda m: m.start_pos)
        
        return matches
    
    def _get_context(self, content: str, start: int, end: int) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - self.max_context_chars // 2)
        ctx_end = min(len(content), end + self.max_context_chars // 2)
        
        context = content[ctx_start:ctx_end]
        
        # Add ellipsis if truncated
        if ctx_start > 0:
            context = "..." + context
        if ctx_end < len(content):
            context = context + "..."
            
        return context
    
    def _generate_reasoning(
        self,
        query: str,
        doc: Dict,
        matches: List[TermMatch],
        score: float
    ) -> List[str]:
        """Generate human-readable reasoning trace."""
        reasoning = []
        
        # Score explanation
        if score >= self.HIGH_CONFIDENCE:
            reasoning.append(f"✓ High relevance score ({score:.2f}): Strong semantic match to query")
        elif score >= self.MEDIUM_CONFIDENCE:
            reasoning.append(f"○ Moderate relevance score ({score:.2f}): Partial match to query themes")
        else:
            reasoning.append(f"△ Lower relevance score ({score:.2f}): Tangential connection to query")
        
        # Match explanation
        exact_matches = [m for m in matches if m.match_type == MatchType.EXACT]
        semantic_matches = [m for m in matches if m.match_type in (MatchType.SEMANTIC, MatchType.RELATED)]
        
        if exact_matches:
            terms = list(set(m.query_term for m in exact_matches))[:5]
            reasoning.append(f"✓ Contains exact query terms: {', '.join(terms)}")
        
        if semantic_matches:
            terms = list(set(m.document_term for m in semantic_matches))[:5]
            reasoning.append(f"→ Contains related medical terms: {', '.join(terms)}")
        
        # Source information
        source = doc.get("source") or doc.get("metadata", {}).get("source", "Unknown")
        if source != "Unknown":
            reasoning.append(f"📄 Source: {source}")
        
        return reasoning
    
    def _highlight_passage(self, content: str, matches: List[TermMatch]) -> str:
        """Highlight matched terms in the passage."""
        if not matches:
            return content[:500] + "..." if len(content) > 500 else content
        
        # Sort matches by position (reverse order for insertion)
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)
        
        highlighted = content
        for match in sorted_matches:
            # Use markdown bold for highlighting
            start, end = match.start_pos, match.end_pos
            original_term = highlighted[start:end]
            highlighted = highlighted[:start] + f"**{original_term}**" + highlighted[end:]
        
        # Truncate if too long
        if len(highlighted) > 600:
            # Find a good truncation point around matches
            first_match_pos = min(m.start_pos for m in matches)
            start = max(0, first_match_pos - 100)
            end = min(len(highlighted), start + 500)
            highlighted = "..." + highlighted[start:end] + "..."
        
        return highlighted
    
    def _calculate_confidence(self, score: float, matches: List[TermMatch]) -> float:
        """Calculate overall confidence in the retrieval."""
        base_confidence = min(score, 1.0)
        
        # Boost for exact matches
        exact_count = sum(1 for m in matches if m.match_type == MatchType.EXACT)
        match_boost = min(0.2, exact_count * 0.05)
        
        # Penalty for no matches
        if not matches:
            base_confidence *= 0.7
        
        return min(1.0, base_confidence + match_boost)
    
    def _generate_citation(self, doc: Dict, rank: int) -> Dict[str, Any]:
        """Generate structured citation metadata."""
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        
        return {
            "rank": rank,
            "source": doc.get("source") or metadata.get("source", "Unknown"),
            "title": doc.get("title") or metadata.get("title", ""),
            "section": doc.get("section") or metadata.get("section", ""),
            "page": doc.get("page") or metadata.get("page"),
            "url": doc.get("url") or metadata.get("url"),
            "timestamp": doc.get("timestamp") or metadata.get("timestamp"),
            "confidence_level": "high" if self._calculate_confidence(
                doc.get("score", doc.get("rerank_score", 0.5)), []
            ) >= self.HIGH_CONFIDENCE else "medium"
        }
    
    async def retrieve_with_explanations(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> Tuple[List[Dict], List[RetrievalExplanation]]:
        """
        Retrieve documents with explanations.
        
        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional arguments for base retriever
            
        Returns:
            Tuple of (documents, explanations)
        """
        if not self.base_retriever:
            logger.warning("No base retriever configured")
            return [], []
        
        # Perform retrieval
        if hasattr(self.base_retriever, "retrieve"):
            documents = await self.base_retriever.retrieve(query, top_k=top_k, **kwargs)
        elif hasattr(self.base_retriever, "search"):
            documents = await self.base_retriever.search(query, k=top_k, **kwargs)
        elif callable(self.base_retriever):
            documents = await self.base_retriever(query, top_k=top_k, **kwargs)
        else:
            logger.error("Base retriever has no compatible method")
            return [], []
        
        # Generate explanations
        scores = [
            doc.get("score") or doc.get("rerank_score") or 0.0
            for doc in documents
        ]
        explanations = self.explain_retrieval(query, documents, scores)
        
        return documents, explanations


def format_explanations_for_response(
    explanations: List[RetrievalExplanation],
    include_highlights: bool = True,
    max_explanations: int = 3
) -> str:
    """
    Format explanations as human-readable text for LLM context.
    
    Args:
        explanations: List of retrieval explanations
        include_highlights: Whether to include highlighted passages
        max_explanations: Maximum number to include
        
    Returns:
        Formatted string for inclusion in prompts
    """
    if not explanations:
        return "No retrieval explanations available."
    
    lines = ["**Retrieval Explanation:**", ""]
    
    for exp in explanations[:max_explanations]:
        lines.append(f"**[{exp.rank}] Score: {exp.overall_score:.2f} (Confidence: {exp.confidence:.0%})**")
        
        if exp.reasoning_trace:
            for reason in exp.reasoning_trace:
                lines.append(f"  {reason}")
        
        if include_highlights and exp.highlighted_passage:
            lines.append(f"  📝 Passage: {exp.highlighted_passage[:200]}...")
        
        lines.append("")
    
    return "\n".join(lines)
