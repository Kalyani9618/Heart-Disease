"""
Tiered Retrieval Orchestrator

This module implements the core retrieval strategy:
1. Detect query intent
2. Route to appropriate tier(s)
3. Merge results with confidence-based ranking
4. Provide fallback mechanism

Tiered Strategy:
- Tier 1 (High Precision): StatPearls + Textbooks
  - For: Clinical guidelines, established facts, anatomy/pathology
  - Strategy: Always search first, high confidence results stop search
  
- Tier 2 (High Recall): PubMed
  - For: Recent research, validation studies, novel findings
  - Strategy: Use only if Tier 1 insufficient or explicitly requested
"""


import logging
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import re

from rag.retrieval.models import (
    MedicalDocument,
    SourceTier,
)

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Detected user intent from query"""
    GUIDELINE = "guideline"  # "What's the standard treatment for..."
    DIAGNOSIS = "diagnosis"  # "How to diagnose..."
    RESEARCH = "research"  # "Recent studies on...", "Meta-analysis"
    DEFINITION = "definition"  # "What is...", "Define..."
    COMPARISON = "comparison"  # "Difference between..."
    VALIDATION = "validation"  # "Evidence for...", "Studies show..."
    UNKNOWN = "unknown"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior"""
    tier1_confidence_threshold: float = 0.85  # Stop if avg confidence >= this
    tier2_fallback_threshold: float = 0.65  # Use Tier 2 if < this
    max_results_tier1: int = 10
    max_results_tier2: int = 5
    max_combined_results: int = 15
    always_search_tier1: bool = True  # Always start with Tier 1
    stream_pubmed_if_needed: bool = True  # Stream PubMed on demand


class IntentDetector:
    """Detects user intent from query text"""
    
    # Intent detection cache for repeated queries
    _intent_cache: Dict[int, QueryIntent] = {}
    
    GUIDELINE_PATTERNS = [
        r"standard (treatment|management|therapy|protocol)",
        r"(how to|best way to) treat",
        r"clinical guideline",
        r"recommended (treatment|approach)",
        r"first[- ]?line",
        r"gold standard",
    ]
    
    DIAGNOSIS_PATTERNS = [
        r"(how to|diagnose|diagnosis)",
        r"diagnostic (criteria|test|workup)",
        r"differential diagnosis",
        r"rule out",
    ]
    
    RESEARCH_PATTERNS = [
        r"(recent|latest|new) (studies?|research)",
        r"meta[- ]?analysis",
        r"systematic review",
        r"clinical trial",
        r"evidence for",
    ]
    
    DEFINITION_PATTERNS = [
        r"what (is|are)",
        r"define",
        r"definition of",
        r"explain",
    ]
    
    COMPARISON_PATTERNS = [
        r"(difference|difference) between",
        r"compare",
        r"versus",
        r"vs\.?",
    ]
    
    @classmethod
    def detect(cls, query: str) -> QueryIntent:
        """
        Detect user intent from query.
        
        Args:
            query: User's natural language query
        
        Returns:
            QueryIntent enum value
        """
        # Cache check
        query_hash = hash(query.lower().strip()[:100])
        if query_hash in cls._intent_cache:
            return cls._intent_cache[query_hash]
        
        query_lower = query.lower()
        
        # Check each pattern group in order of specificity
        patterns = [
            (QueryIntent.RESEARCH, cls.RESEARCH_PATTERNS),
            (QueryIntent.GUIDELINE, cls.GUIDELINE_PATTERNS),
            (QueryIntent.DIAGNOSIS, cls.DIAGNOSIS_PATTERNS),
            (QueryIntent.COMPARISON, cls.COMPARISON_PATTERNS),
            (QueryIntent.DEFINITION, cls.DEFINITION_PATTERNS),
        ]
        
        for intent, pattern_list in patterns:
            for pattern in pattern_list:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    logger.debug(f"Detected intent: {intent.value} from pattern: {pattern}")
                    # Cache result (limit cache size)
                    if len(cls._intent_cache) > 500:
                        cls._intent_cache.clear()
                    cls._intent_cache[query_hash] = intent
                    return intent
        
        # Cache unknown intent too
        cls._intent_cache[query_hash] = QueryIntent.UNKNOWN
        return QueryIntent.UNKNOWN
    
    @classmethod
    def should_search_tier2(cls, intent: QueryIntent) -> bool:
        """
        Determine if Tier 2 (PubMed) should be searched for this intent.
        
        Args:
            intent: Detected query intent
        
        Returns:
            True if Tier 2 should be included
        """
        # Always search Tier 2 for research queries
        if intent in [QueryIntent.RESEARCH, QueryIntent.VALIDATION]:
            return True
        
        # May search Tier 2 for other intents if Tier 1 insufficient
        return False


class TieredRetriever:
    """
    Main retrieval orchestrator that routes queries through tiers.
    
    Strategy:
    1. Detect query intent
    2. Search Tier 1 (StatPearls + Textbooks)
    3. Evaluate confidence
    4. If insufficient, search Tier 2 (PubMed) based on intent
    5. Merge and rank results
    """
    
    def __init__(
        self,
        tier1_vector_db,  # e.g., Chroma client with Tier 1 docs
        tier2_vector_db=None,  # e.g., PubMedStreamer
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize retriever.
        
        Args:
            tier1_vector_db: Vector database with StatPearls + Textbooks
            tier2_vector_db: PubMed streamer or vector database
            config: RetrievalConfig with thresholds
        """
        self.tier1_db = tier1_vector_db
        self.tier2_db = tier2_vector_db
        self.config = config or RetrievalConfig()
        self.intent_detector = IntentDetector()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        force_tier2: bool = False,
    ) -> List[MedicalDocument]:
        """
        Retrieve documents using tiered strategy.
        
        Args:
            query: User query
            top_k: Max documents to return
            force_tier2: Force inclusion of Tier 2 results
        
        Returns:
            List of MedicalDocument objects, ranked by relevance
        """
        logger.info(f"Retrieving for query: {query[:80]}...")
        
        # Step 1: Detect intent
        intent = self.intent_detector.detect(query)
        logger.info(f"  Detected intent: {intent.value}")
        
        # Step 2: Search Tier 1
        logger.info("  Searching Tier 1 (StatPearls + Textbooks)...")
        tier1_results = self._search_tier1(query, top_k=self.config.max_results_tier1)
        logger.info(f"    Found {len(tier1_results)} results")
        
        # Step 3: Evaluate confidence
        has_high_confidence = self._has_high_confidence(tier1_results)
        logger.info(f"    Confidence: {'✓ High' if has_high_confidence else '✗ Low'}")
        
        # Step 4: Decide on Tier 2
        should_search_tier2 = (
            force_tier2 or
            (not has_high_confidence) or
            (intent == QueryIntent.RESEARCH and self.config.stream_pubmed_if_needed)
        )
        
        tier2_results = []
        if should_search_tier2 and self.tier2_db:
            logger.info("  Searching Tier 2 (PubMed)...")
            tier2_results = self._search_tier2(query, top_k=self.config.max_results_tier2)
            logger.info(f"    Found {len(tier2_results)} results")
        
        # Step 5: Merge and rank
        merged = self._merge_results(tier1_results, tier2_results, intent)
        
        # Limit to requested top_k
        final_results = merged[:top_k]
        
        logger.info(f"  Returning {len(final_results)} documents")
        return final_results
    
    def _search_tier1(self, query: str, top_k: int) -> List[MedicalDocument]:
        """Search Tier 1 databases"""
        try:
            # Query with metadata filtering to only Tier 1
            results = self.tier1_db.search(
                query=query,
                top_k=top_k,
                where_filter={
                    "tier": {"$in": [
                        SourceTier.TIER_1_STATPEARLS.value,
                        SourceTier.TIER_1_TEXTBOOKS.value,
                    ]}
                }
            )
            return results
        except Exception as e:
            logger.error(f"Error searching Tier 1: {e}")
            return []
    
    def _search_tier2(self, query: str, top_k: int) -> List[MedicalDocument]:
        """Search Tier 2 (PubMed) databases"""
        try:
            if hasattr(self.tier2_db, 'stream_search'):
                # PubMed streamer
                results = self.tier2_db.stream_search(query, top_k=top_k)
            else:
                # Standard vector DB
                results = self.tier2_db.search(
                    query=query,
                    top_k=top_k,
                    where_filter={"tier": SourceTier.TIER_2_PUBMED.value}
                )
            return results
        except Exception as e:
            logger.error(f"Error searching Tier 2: {e}")
            return []
    
    def _has_high_confidence(self, results: List[MedicalDocument]) -> bool:
        """
        Check if results have high average confidence.
        
        Args:
            results: List of retrieved documents
        
        Returns:
            True if avg confidence >= threshold
        """
        if not results:
            return False
        
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        return avg_confidence >= self.config.tier1_confidence_threshold
    
    def _merge_results(
        self,
        tier1: List[MedicalDocument],
        tier2: List[MedicalDocument],
        intent: QueryIntent,
    ) -> List[MedicalDocument]:
        """
        Merge Tier 1 and Tier 2 results with intelligent ranking.
        
        Strategy:
        - Tier 1 results always come first (more authoritative)
        - Tier 2 results supplement Tier 1
        - Higher confidence scores ranked higher within tier
        - For research queries, Tier 2 gets more prominence
        
        Args:
            tier1: Tier 1 search results
            tier2: Tier 2 search results
            intent: Query intent
        
        Returns:
            Merged and ranked results
        """
        # Remove duplicates (same document from both tiers)
        tier1_ids = {doc.document_id for doc in tier1}
        tier2_dedup = [doc for doc in tier2 if doc.document_id not in tier1_ids]
        
        # For research queries, mix tiers more
        if intent in [QueryIntent.RESEARCH, QueryIntent.VALIDATION]:
            # Interleave results: 1 from Tier 1, 1 from Tier 2, repeat
            merged = []
            for i in range(max(len(tier1), len(tier2_dedup))):
                if i < len(tier1):
                    merged.append(tier1[i])
                if i < len(tier2_dedup):
                    merged.append(tier2_dedup[i])
        else:
            # For non-research, Tier 1 dominates
            merged = tier1 + tier2_dedup
        
        # Sort by confidence within same tier
        tier1_merged = [d for d in merged if d.tier == SourceTier.TIER_1_STATPEARLS or d.tier == SourceTier.TIER_1_TEXTBOOKS]
        tier2_merged = [d for d in merged if d.tier == SourceTier.TIER_2_PUBMED]
        
        tier1_sorted = sorted(tier1_merged, key=lambda x: -x.confidence_score)
        tier2_sorted = sorted(tier2_merged, key=lambda x: -x.confidence_score)
        
        return tier1_sorted + tier2_sorted
    
    def retrieve_with_explanation(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[MedicalDocument], Dict[str, Any]]:
        """
        Retrieve documents and return detailed explanation of retrieval process.
        
        Returns:
            (documents, explanation_dict)
        """
        intent = self.intent_detector.detect(query)
        tier1_results = self._search_tier1(query, top_k=self.config.max_results_tier1)
        
        explanation = {
            "query": query,
            "detected_intent": intent.value,
            "tier1_results": len(tier1_results),
            "tier1_avg_confidence": sum(r.confidence_score for r in tier1_results) / len(tier1_results) if tier1_results else 0,
            "strategy_used": "tier1_only",
        }
        
        # Decide on Tier 2
        if not self._has_high_confidence(tier1_results):
            explanation["strategy_used"] = "tier1_then_tier2"
            tier2_results = self._search_tier2(query, top_k=self.config.max_results_tier2)
            explanation["tier2_results"] = len(tier2_results)
            final_results = self._merge_results(tier1_results, tier2_results, intent)
        else:
            explanation["tier2_results"] = 0
            final_results = tier1_results
        
        return final_results[:top_k], explanation


class RetrievalEvaluator:
    """Evaluate retrieval quality and effectiveness"""
    
    def __init__(self, retriever: TieredRetriever):
        self.retriever = retriever
    
    def evaluate_intent_detection(self, test_queries: Dict[str, QueryIntent]) -> float:
        """
        Evaluate intent detection accuracy.
        
        Args:
            test_queries: Dict of query -> expected_intent
        
        Returns:
            Accuracy (0.0-1.0)
        """
        detector = self.retriever.intent_detector
        correct = 0
        
        for query, expected_intent in test_queries.items():
            detected = detector.detect(query)
            if detected == expected_intent:
                correct += 1
        
        accuracy = correct / len(test_queries) if test_queries else 0.0
        logger.info(f"Intent detection accuracy: {accuracy:.1%}")
        return accuracy
    
    def evaluate_confidence_calibration(
        self,
        sample_queries: List[str],
        sample_size: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate if confidence scores are well-calibrated.
        
        Returns:
            Dict with calibration metrics
        """
        metrics = {
            "avg_tier1_confidence": 0.0,
            "avg_tier2_confidence": 0.0,
        }
        
        for query in sample_queries[:sample_size]:
            results, _ = self.retriever.retrieve_with_explanation(query)
            
            tier1_docs = [d for d in results if d.tier in [
                SourceTier.TIER_1_STATPEARLS,
                SourceTier.TIER_1_TEXTBOOKS,
            ]]
            tier2_docs = [d for d in results if d.tier == SourceTier.TIER_2_PUBMED]
            
            if tier1_docs:
                metrics["avg_tier1_confidence"] = sum(d.confidence_score for d in tier1_docs) / len(tier1_docs)
            if tier2_docs:
                metrics["avg_tier2_confidence"] = sum(d.confidence_score for d in tier2_docs) / len(tier2_docs)
        
        return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test intent detection
    print("Testing Intent Detection:")
    print("=" * 50)
    
    test_queries = {
        "What is the standard treatment for atrial fibrillation?": QueryIntent.GUIDELINE,
        "How do you diagnose heart failure?": QueryIntent.DIAGNOSIS,
        "Recent studies on aspirin for cardiovascular prevention": QueryIntent.RESEARCH,
        "Define myocardial infarction": QueryIntent.DEFINITION,
        "Difference between stable and unstable angina": QueryIntent.COMPARISON,
    }
    
    detector = IntentDetector()
    
    for query, expected in test_queries.items():
        detected = detector.detect(query)
        match = "✓" if detected == expected else "✗"
        print(f"{match} {query}")
        print(f"   Expected: {expected.value}, Got: {detected.value}\n")

