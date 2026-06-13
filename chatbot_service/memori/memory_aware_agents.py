"""
Memory-Enhanced NLP Agents

This module extends the existing NLP agents (IntentRecognizer, SentimentAnalyzer, etc.)
with memory-aware context injection, allowing them to make better decisions based on
patient history and conversation context.

Features:
- Contextual intent recognition with prior conversation history
- Sentiment analysis with emotional trend awareness
- Entity extraction with domain knowledge accumulation
- Risk assessment with longitudinal health data
- All with comprehensive logging and metrics

Complexity:
- add_context: O(1) to O(n) depending on context size
- recognize_with_context: O(n) where n = context size
- All operations have timeouts to prevent performance degradation

Integration Points:
- MemoryManager for retrieving patient context
- Existing NLP components for core functionality
- Metrics collection for observability
"""

import logging
import asyncio
import json
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass, field

# Import from local memori module
from .memory_manager import (
    PatientMemory,
    MemoryResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol Definitions for External Components
# ============================================================================

@runtime_checkable
class IntentRecognizer(Protocol):
    """Protocol for intent recognizer implementations."""
    def recognize(self, text: str) -> Any:
        """Recognize intent from text."""
        ...


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Protocol for sentiment analyzer implementations."""
    def analyze(self, text: str) -> Any:
        """Analyze sentiment from text."""
        ...


@runtime_checkable
class RiskAssessor(Protocol):
    """Protocol for risk assessor implementations."""
    def assess_risk(self, metrics: Dict[str, Any]) -> Any:
        """Assess health risk from metrics."""
        ...


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class IntentResult:
    """Base intent recognition result."""
    type: str
    confidence: float
    keywords: List[str] = field(default_factory=list)


@dataclass
class SentimentResult:
    """Base sentiment analysis result."""
    sentiment: str
    confidence: float
    positive_score: float = 0.0
    negative_score: float = 0.0
    neutral_score: float = 0.0


@dataclass
class ConversationContext:
    """Structured conversation context from memory."""

    patient_id: str
    session_id: str
    recent_conversations: List[Dict[str, Any]]
    recent_health_data: List[Dict[str, Any]]
    prior_intents: List[str]
    emotional_trend: Optional[str]  # improving, declining, stable
    last_risk_assessment: Optional[Dict[str, Any]]

    def __bool__(self) -> bool:
        """Check if context has meaningful data."""
        return bool(
            self.recent_conversations or self.recent_health_data or self.prior_intents
        )


@dataclass
class ContextualIntentResult(IntentResult):
    """Intent result with context information."""

    context_used: bool = False
    prior_related_intents: List[str] = field(default_factory=list)
    context_confidence_boost: float = 0.0  # How much context improved confidence


@dataclass
class ContextualSentimentResult(SentimentResult):
    """Sentiment result with emotional trend awareness."""

    emotional_trend: Optional[str] = None  # improving, declining, stable
    context_used: bool = False
    prior_sentiments: List[str] = field(default_factory=list)
    trend_confidence: float = 0.0


# ============================================================================
# Context Retrieval
# ============================================================================


class ContextRetriever:
    """
    Retrieves and synthesizes conversation context from memory.

    Responsibility: Load relevant context from PatientMemory and structure it
    for agent consumption.

    Complexity:
    - retrieve_context: O(m) where m = number of memory searches
    - All operations bounded by timeout (30s default)
    """

    REQUEST_TIMEOUT = 30  # seconds

    @staticmethod
    async def retrieve_context(
        patient_memory: PatientMemory,
        query: str,
        include_health_data: bool = True,
        conversation_limit: int = 5,
    ) -> ConversationContext:
        """
        Retrieve structured context from patient memory.

        Args:
            patient_memory: PatientMemory instance for patient
            query: Current query/user message for context matching
            include_health_data: Whether to retrieve health data
            conversation_limit: Number of recent conversations to retrieve

        Returns:
            ConversationContext with relevant memories

        Complexity: O(m) where m = memory search operations
        """
        try:
            start_time = time.time()
            patient_id = patient_memory.patient_id
            session_id = patient_memory.session_id

            # Run all searches in parallel
            search_tasks = [
                patient_memory.search(
                    query=query,
                    data_type="conversation",
                    limit=conversation_limit,
                    timeout=ContextRetriever.REQUEST_TIMEOUT,
                ),
                patient_memory.search(
                    query="prior intents",
                    data_type="conversation",
                    limit=10,
                    timeout=ContextRetriever.REQUEST_TIMEOUT,
                ),
            ]

            if include_health_data:
                search_tasks.append(
                    patient_memory.search(
                        query="recent health measurements vitals",
                        data_type="health_data",
                        limit=10,
                        timeout=ContextRetriever.REQUEST_TIMEOUT,
                    )
                )

            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process results
            relevant_conversations = (
                [_parse_conversation(r) for r in results[0]]
                if not isinstance(results[0], Exception)
                else []
            )

            prior_intent_results = (
                [_extract_intent_from_conversation(r) for r in results[1]]
                if not isinstance(results[1], Exception)
                else []
            )
            prior_intents = [i for i in prior_intent_results if i]

            recent_health_data = []
            if include_health_data and len(results) > 2:
                recent_health_data = (
                    [_parse_health_data(r) for r in results[2]]
                    if not isinstance(results[2], Exception)
                    else []
                )

            # Calculate emotional trend from recent sentiments
            emotional_trend = _calculate_emotional_trend(relevant_conversations)

            # Get last risk assessment if available
            last_risk = None
            try:
                risk_results = await patient_memory.search(
                    query="risk assessment",
                    data_type="health_data",
                    limit=1,
                    timeout=ContextRetriever.REQUEST_TIMEOUT,
                )
                if risk_results:
                    last_risk = json.loads(risk_results[0].content)
            except Exception as e:
                logger.warning(f"Error retrieving risk assessment: {e}")

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Context retrieved: patient_id={patient_id}, "
                f"conversations={len(relevant_conversations)}, "
                f"health_records={len(recent_health_data)}, "
                f"latency_ms={latency_ms:.2f}"
            )

            return ConversationContext(
                patient_id=patient_id,
                session_id=session_id,
                recent_conversations=relevant_conversations,
                recent_health_data=recent_health_data,
                prior_intents=prior_intents,
                emotional_trend=emotional_trend,
                last_risk_assessment=last_risk,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Context retrieval timeout for {patient_memory.patient_id}")
            return ConversationContext(
                patient_id=patient_memory.patient_id,
                session_id=patient_memory.session_id,
                recent_conversations=[],
                recent_health_data=[],
                prior_intents=[],
                emotional_trend=None,
                last_risk_assessment=None,
            )
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return ConversationContext(
                patient_id=patient_memory.patient_id,
                session_id=patient_memory.session_id,
                recent_conversations=[],
                recent_health_data=[],
                prior_intents=[],
                emotional_trend=None,
                last_risk_assessment=None,
            )


def _parse_conversation(memory_result: MemoryResult) -> Dict[str, Any]:
    """Parse conversation from memory result."""
    try:
        return json.loads(memory_result.content)
    except Exception as e:
        logger.warning(f"Error parsing conversation: {e}")
        return {}


def _extract_intent_from_conversation(memory_result: MemoryResult) -> Optional[str]:
    """Extract intent from conversation memory."""
    try:
        conv = json.loads(memory_result.content)
        return conv.get("intent")
    except Exception:
        return None


def _parse_health_data(memory_result: MemoryResult) -> Dict[str, Any]:
    """Parse health data from memory result."""
    try:
        return {
            "data": json.loads(memory_result.content),
            "type": memory_result.metadata.get("data_type", "unknown"),
            "severity": memory_result.metadata.get("severity"),
            "timestamp": memory_result.timestamp,
        }
    except Exception as e:
        logger.warning(f"Error parsing health data: {e}")
        return {}


def _calculate_emotional_trend(conversations: List[Dict[str, Any]]) -> Optional[str]:
    """
    Calculate emotional trend from recent conversations.

    Returns:
    - "improving" if sentiment is becoming more positive
    - "declining" if sentiment is becoming more negative
    - "stable" if sentiment is relatively constant
    """
    if not conversations:
        return None

    sentiments = []
    for conv in conversations:
        sentiment = conv.get("sentiment")
        if sentiment:
            sentiments.append(sentiment)

    if len(sentiments) < 2:
        return "stable"

    # Simple trend calculation (can be enhanced with more sophisticated analysis)
    sentiment_scores = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0,
        "distressed": -2.0,
        "urgent": -3.0,
    }

    scores = [sentiment_scores.get(s, 0.0) for s in sentiments]

    # Compare recent vs. older
    recent_avg = sum(scores[-3:]) / min(3, len(scores))
    older_avg = (
        sum(scores[:-3]) / max(1, len(scores) - 3) if len(scores) > 3 else recent_avg
    )

    if recent_avg > older_avg + 0.5:
        return "improving"
    elif recent_avg < older_avg - 0.5:
        return "declining"
    else:
        return "stable"


# ============================================================================
# Memory-Enhanced Intent Recognizer
# ============================================================================


class MemoryAwareIntentRecognizer:
    """
    Intent recognizer with memory-aware context injection.

    Enhances IntentRecognizer by:
    1. Retrieving conversation context from memory
    2. Considering prior intents for consistency
    3. Boosting confidence based on contextual relevance
    4. Storing intent for future context

    Complexity: O(n) where n = context size
    """

    def __init__(self, base_recognizer: IntentRecognizer):
        """
        Initialize memory-aware recognizer.

        Args:
            base_recognizer: Underlying IntentRecognizer instance
        """
        self.base_recognizer = base_recognizer
        self.context_retriever = ContextRetriever()

        # Metrics
        self.context_uses = 0
        self.context_helped = 0  # Times context improved confidence

    async def recognize_with_context(
        self,
        text: str,
        patient_memory: PatientMemory,
    ) -> ContextualIntentResult:
        """
        Recognize intent with memory context.

        Args:
            text: User input text
            patient_memory: PatientMemory instance

        Returns:
            ContextualIntentResult with context information
        """
        try:
            start_time = time.time()

            # Get base intent (fast path, no memory)
            base_intent = await asyncio.to_thread(self.base_recognizer.recognize, text)

            # Try to retrieve context
            try:
                context = await asyncio.wait_for(
                    self.context_retriever.retrieve_context(
                        patient_memory,
                        query=text,
                        include_health_data=False,
                        conversation_limit=5,
                    ),
                    timeout=10,  # Quick timeout for context
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Context retrieval failed: {e}")
                context = ConversationContext(
                    patient_id=patient_memory.patient_id,
                    session_id=patient_memory.session_id,
                    recent_conversations=[],
                    recent_health_data=[],
                    prior_intents=[],
                    emotional_trend=None,
                    last_risk_assessment=None,
                )

            # Enhance intent using context
            confidence_boost = 0.0
            if context.prior_intents:
                self.context_uses += 1
                # If prior intents match current intent, boost confidence
                if base_intent.type in context.prior_intents:
                    confidence_boost = 0.1
                    self.context_helped += 1

            latency_ms = (time.time() - start_time) * 1000

            # Create contextual result
            result = ContextualIntentResult(
                type=base_intent.type,
                confidence=min(1.0, base_intent.confidence + confidence_boost),
                keywords=base_intent.keywords,
                context_used=bool(context),
                prior_related_intents=context.prior_intents,
                context_confidence_boost=confidence_boost,
            )

            logger.debug(
                f"Intent recognized: intent={result.type}, "
                f"confidence={result.confidence:.2f}, "
                f"boost={confidence_boost:.2f}, "
                f"context_used={result.context_used}, "
                f"latency_ms={latency_ms:.2f}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error in memory-aware intent recognition: {e}", exc_info=True
            )
            # Fallback to base recognition
            base_intent = await asyncio.to_thread(self.base_recognizer.recognize, text)
            return ContextualIntentResult(
                type=base_intent.type,
                confidence=base_intent.confidence,
                keywords=base_intent.keywords,
                context_used=False,
                prior_related_intents=[],
                context_confidence_boost=0.0,
            )


# ============================================================================
# Memory-Enhanced Sentiment Analyzer
# ============================================================================


class MemoryAwareSentimentAnalyzer:
    """
    Sentiment analyzer with emotional trend awareness.

    Enhances SentimentAnalyzer by:
    1. Tracking emotional trends over conversations
    2. Detecting mood swings and concerning patterns
    3. Adjusting sentiment classification based on history
    4. Providing early warning for distressed patients

    Complexity: O(n) where n = conversation history size
    """

    def __init__(self, base_analyzer: SentimentAnalyzer):
        """
        Initialize memory-aware sentiment analyzer.

        Args:
            base_analyzer: Underlying SentimentAnalyzer instance
        """
        self.base_analyzer = base_analyzer
        self.context_retriever = ContextRetriever()

    async def analyze_with_context(
        self,
        text: str,
        patient_memory: PatientMemory,
    ) -> ContextualSentimentResult:
        """
        Analyze sentiment with emotional trend awareness.

        Args:
            text: User input text
            patient_memory: PatientMemory instance

        Returns:
            ContextualSentimentResult with trend information
        """
        try:
            # Get base sentiment (fast)
            base_sentiment = await asyncio.to_thread(self.base_analyzer.analyze, text)

            # Try to retrieve emotional context
            try:
                context = await asyncio.wait_for(
                    self.context_retriever.retrieve_context(
                        patient_memory,
                        query="emotional health mood feeling",
                        include_health_data=False,
                        conversation_limit=10,
                    ),
                    timeout=10,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Context retrieval failed: {e}")
                context = ConversationContext(
                    patient_id=patient_memory.patient_id,
                    session_id=patient_memory.session_id,
                    recent_conversations=[],
                    recent_health_data=[],
                    prior_intents=[],
                    emotional_trend=None,
                    last_risk_assessment=None,
                )

            # Extract prior sentiments
            prior_sentiments = []
            for conv in context.recent_conversations:
                sentiment = conv.get("sentiment")
                if sentiment:
                    prior_sentiments.append(sentiment)

            # Adjust sentiment based on trend
            adjusted_sentiment = base_sentiment.sentiment
            trend_confidence = 0.0

            if context.emotional_trend == "declining" and len(prior_sentiments) >= 3:
                # If trend is declining and currently negative, increase concern
                if base_sentiment.sentiment in ["negative", "distressed"]:
                    adjusted_sentiment = "distressed"
                    trend_confidence = 0.3

            # Create contextual result
            result = ContextualSentimentResult(
                sentiment=adjusted_sentiment,
                confidence=base_sentiment.confidence,
                positive_score=base_sentiment.positive_score,
                negative_score=base_sentiment.negative_score,
                neutral_score=base_sentiment.neutral_score,
                emotional_trend=context.emotional_trend,
                context_used=bool(context),
                prior_sentiments=prior_sentiments,
                trend_confidence=trend_confidence,
            )

            logger.debug(
                f"Sentiment analyzed: sentiment={result.sentiment}, "
                f"confidence={result.confidence:.2f}, "
                f"trend={result.emotional_trend}, "
                f"context_used={result.context_used}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error in memory-aware sentiment analysis: {e}", exc_info=True
            )
            # Fallback to base sentiment
            base_sentiment = await asyncio.to_thread(self.base_analyzer.analyze, text)
            return ContextualSentimentResult(
                sentiment=base_sentiment.sentiment,
                confidence=base_sentiment.confidence,
                positive_score=base_sentiment.positive_score,
                negative_score=base_sentiment.negative_score,
                neutral_score=base_sentiment.neutral_score,
                emotional_trend=None,
                context_used=False,
                prior_sentiments=[],
                trend_confidence=0.0,
            )


# ============================================================================
# Memory-Enhanced Risk Assessor
# ============================================================================


class MemoryAwareRiskAssessor:
    """
    Risk assessor with longitudinal health data integration.

    Enhances RiskAssessor by:
    1. Accessing prior risk assessments and trends
    2. Using health data history for risk calculation
    3. Detecting deteriorating health patterns
    4. Providing trend-aware risk interpretation

    Complexity: O(n) where n = health data history size
    """

    def __init__(self, base_assessor: RiskAssessor):
        """
        Initialize memory-aware risk assessor.

        Args:
            base_assessor: Underlying RiskAssessor instance
        """
        self.base_assessor = base_assessor
        self.context_retriever = ContextRetriever()

    async def assess_risk_with_context(
        self,
        metrics: Dict[str, Any],
        patient_memory: PatientMemory,
    ) -> Dict[str, Any]:
        """
        Assess risk with longitudinal health data context.

        Args:
            metrics: Current health metrics
            patient_memory: PatientMemory instance

        Returns:
            Risk assessment with context and trend information
        """
        try:
            # Get base risk assessment
            base_assessment = await asyncio.to_thread(
                self.base_assessor.assess_risk, metrics
            )

            # Retrieve health history
            try:
                context = await asyncio.wait_for(
                    self.context_retriever.retrieve_context(
                        patient_memory,
                        query="vital signs health measurements disease risk",
                        include_health_data=True,
                        conversation_limit=0,
                    ),
                    timeout=10,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Context retrieval failed: {e}")
                context = ConversationContext(
                    patient_id=patient_memory.patient_id,
                    session_id=patient_memory.session_id,
                    recent_conversations=[],
                    recent_health_data=[],
                    prior_intents=[],
                    emotional_trend=None,
                    last_risk_assessment=None,
                )

            # Analyze health trends
            health_trend = _analyze_health_trend(context.recent_health_data)

            # Create enriched assessment
            base_dict = (
                base_assessment.dict()
                if hasattr(base_assessment, "dict")
                else base_assessment
            )
            result = {
                **base_dict,
                "prior_risk_assessment": context.last_risk_assessment,
                "health_trend": health_trend,
                "longitudinal_data_count": len(context.recent_health_data),
                "context_used": bool(context),
            }

            # Adjust recommendations based on trend
            if health_trend == "deteriorating":
                result["recommendations"] = result.get("recommendations", []) + [
                    "Urgent medical consultation recommended due to declining health trend"
                ]

            logger.debug(
                f"Risk assessed: risk_level={result.get('risk_level')}, "
                f"health_trend={health_trend}, "
                f"prior_risk={context.last_risk_assessment is not None}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in memory-aware risk assessment: {e}", exc_info=True)
            # Fallback to base assessment
            base_assessment = await asyncio.to_thread(
                self.base_assessor.assess_risk, metrics
            )
            return (
                base_assessment.dict()
                if hasattr(base_assessment, "dict")
                else base_assessment
            )


def _analyze_health_trend(health_data: List[Dict[str, Any]]) -> Optional[str]:
    """
    Analyze health data trend.

    Returns:
    - "improving" if key metrics improving
    - "deteriorating" if key metrics worsening
    - "stable" if relatively constant
    """
    if len(health_data) < 2:
        return None

    # Extract blood pressure trends as example
    bp_readings = []
    for record in health_data:
        data = record.get("data", {})
        if "heart_rate" in data or "blood_pressure_systolic" in data:
            bp_readings.append(data)

    if len(bp_readings) < 2:
        return "stable"

    # Simple trend: compare averages
    first_half_avg = sum(
        r.get("heart_rate", 0) for r in bp_readings[: len(bp_readings) // 2]
    ) / max(1, len(bp_readings) // 2)

    second_half_avg = sum(
        r.get("heart_rate", 0) for r in bp_readings[len(bp_readings) // 2 :]
    ) / max(1, len(bp_readings) - len(bp_readings) // 2)

    if second_half_avg > first_half_avg + 5:
        return "deteriorating"
    elif second_half_avg < first_half_avg - 5:
        return "improving"
    else:
        return "stable"


# ============================================================================
# Multi-Tier Cache for Memory Operations
# ============================================================================


class MultiTierCache:
    """
    Multi-tier cache for memory-aware agents.
    
    Provides L1 (in-memory) and optional L2 (Redis) caching 
    for frequently accessed memory contexts.
    Uses OrderedDict for O(1) LRU eviction instead of list scanning.
    """
    
    def __init__(self, l1_max_size: int = 100, l2_ttl: int = 300):
        self.l1_cache: OrderedDict[str, Any] = OrderedDict()
        self.l1_max_size = l1_max_size
        self.l2_ttl = l2_ttl
        self._redis_client = None
        self._hits = 0
        self._misses = 0
        
        # Try to connect to Redis for L2
        try:
            import redis
            self._redis_client = redis.Redis(
                host='localhost', port=6379, db=0, 
                decode_responses=True, socket_timeout=2
            )
            self._redis_client.ping()
            logger.info("MultiTierCache: L2 Redis connected")
        except Exception:
            self._redis_client = None
            logger.debug("MultiTierCache: L2 Redis unavailable, using L1 only")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)."""
        # L1 check
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)
            self._hits += 1
            return self.l1_cache[key]
        
        # L2 check
        if self._redis_client:
            try:
                val = self._redis_client.get(f"mtc:{key}")
                if val:
                    result = json.loads(val)
                    self._set_l1(key, result)
                    self._hits += 1
                    return result
            except Exception:
                pass
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in both cache tiers."""
        self._set_l1(key, value)
        
        if self._redis_client:
            try:
                self._redis_client.setex(
                    f"mtc:{key}", self.l2_ttl, json.dumps(value, default=str)
                )
            except Exception:
                pass
    
    def _set_l1(self, key: str, value: Any) -> None:
        """Set value in L1 with O(1) LRU eviction using OrderedDict."""
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)
            self.l1_cache[key] = value
        else:
            if len(self.l1_cache) >= self.l1_max_size:
                self.l1_cache.popitem(last=False)  # O(1) eviction
            self.l1_cache[key] = value
    
    def invalidate(self, key: str) -> None:
        """Remove key from all cache tiers."""
        self.l1_cache.pop(key, None)
        if self._redis_client:
            try:
                self._redis_client.delete(f"mtc:{key}")
            except Exception:
                pass
    
    def clear(self) -> None:
        """Clear all caches."""
        self.l1_cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "l1_size": len(self.l1_cache),
            "l1_max_size": self.l1_max_size,
            "l2_available": self._redis_client is not None,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0.0,
        }


# ============================================================================
# Batch Memory Processor
# ============================================================================


class BatchMemoryProcessor:
    """
    Process multiple memory operations in batch for efficiency.
    
    Reduces database round-trips by batching memory reads/writes.
    Supports parallel retrieval and actual data persistence.
    """
    
    def __init__(
        self,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        memory_manager: Any = None,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._memory_manager = memory_manager
        self._pending_writes: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._cache = MultiTierCache()
        self._write_lock = asyncio.Lock() if asyncio else threading.Lock()
        self._flush_count = 0
        self._total_writes = 0
        self._failed_writes = 0
    
    async def batch_retrieve(
        self, 
        patient_ids: List[str], 
        memory_manager: Any = None
    ) -> Dict[str, List[Any]]:
        """
        Retrieve memories for multiple patients in parallel.
        
        Uses asyncio.gather for concurrent retrieval, significantly
        reducing total fetch time for multi-patient operations.
        
        Args:
            patient_ids: List of patient IDs to retrieve
            memory_manager: MemoryManager instance
            
        Returns:
            Dict mapping patient_id to their memories
        """
        mgr = memory_manager or self._memory_manager
        results: Dict[str, List[Any]] = {}
        uncached_ids = []
        
        # Check cache first
        for pid in patient_ids:
            cached = self._cache.get(f"patient:{pid}")
            if cached is not None:
                results[pid] = cached
            else:
                uncached_ids.append(pid)
        
        # Fetch uncached from database in parallel
        if uncached_ids and mgr:
            async def _fetch_patient(pid: str):
                try:
                    memories = await mgr.retrieve(pid)
                    return pid, memories if memories else []
                except Exception as e:
                    logger.warning(f"Failed to retrieve memories for {pid}: {e}")
                    return pid, []

            fetch_results = await asyncio.gather(
                *[_fetch_patient(pid) for pid in uncached_ids],
                return_exceptions=True
            )
            
            for result in fetch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Batch retrieval error: {result}")
                    continue
                pid, memories = result
                results[pid] = memories
                self._cache.set(f"patient:{pid}", memories)
        
        return results
    
    def queue_write(self, operation: Dict[str, Any]) -> None:
        """Queue a write operation for batch processing."""
        self._pending_writes.append(operation)
        
        if len(self._pending_writes) >= self.batch_size:
            # Schedule flush - handle both async and sync contexts
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.flush())
            except RuntimeError:
                # No running event loop, flush synchronously via thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, self.flush())
    
    async def flush(self) -> int:
        """
        Flush pending write operations to the memory manager.
        
        Actually persists each queued operation to the database
        through the memory manager, with error handling per-operation.
        
        Returns:
            Number of successfully processed operations
        """
        if not self._pending_writes:
            return 0
        
        writes = self._pending_writes.copy()
        self._pending_writes.clear()
        self._last_flush = time.time()
        
        processed = 0
        mgr = self._memory_manager
        
        for op in writes:
            try:
                op_type = op.get("type", "store")
                patient_id = op.get("patient_id", op.get("user_id", "unknown"))
                content = op.get("content", "")
                memory_type = op.get("memory_type", "general")
                metadata = op.get("metadata", {})
                session_id = op.get("session_id", "default")
                
                if mgr and op_type == "store":
                    await mgr.store_memory(
                        patient_id=patient_id,
                        memory_type=memory_type,
                        content=content,
                        session_id=session_id,
                        metadata=metadata,
                    )
                elif mgr and op_type == "conversation":
                    patient_memory = await mgr.get_patient_memory(patient_id, session_id)
                    await patient_memory.add_conversation(
                        user_message=op.get("user_message", ""),
                        assistant_message=op.get("assistant_message", ""),
                        intent=op.get("intent"),
                        sentiment=op.get("sentiment"),
                        metadata=metadata,
                    )
                elif mgr and op_type == "health_data":
                    patient_memory = await mgr.get_patient_memory(patient_id, session_id)
                    await patient_memory.add_health_data(
                        data_type=op.get("data_type", "unknown"),
                        data=op.get("data", {}),
                        severity=op.get("severity"),
                        metadata=metadata,
                    )
                else:
                    logger.warning(
                        f"Unknown write operation type '{op_type}' or no memory manager"
                    )
                    continue
                
                processed += 1
                self._total_writes += 1
                
                # Invalidate cache for this patient
                self._cache.invalidate(f"patient:{patient_id}")
                
            except Exception as e:
                self._failed_writes += 1
                logger.error(f"Failed to process write operation: {e}")
        
        self._flush_count += 1
        logger.debug(
            f"Batch flush complete: {processed}/{len(writes)} operations persisted"
        )
        return processed
    
    def stats(self) -> Dict[str, Any]:
        """Return processor statistics."""
        return {
            "pending_writes": len(self._pending_writes),
            "batch_size": self.batch_size,
            "last_flush": self._last_flush,
            "flush_count": self._flush_count,
            "total_writes": self._total_writes,
            "failed_writes": self._failed_writes,
            "cache_stats": self._cache.stats(),
        }


# ============================================================================
# Convenience Factory
# ============================================================================


def create_memory_aware_agents(
    intent_recognizer: IntentRecognizer,
    sentiment_analyzer: SentimentAnalyzer,
    risk_assessor: RiskAssessor,
) -> Tuple[
    MemoryAwareIntentRecognizer, MemoryAwareSentimentAnalyzer, MemoryAwareRiskAssessor
]:
    """
    Create memory-aware versions of core NLP agents.

    Args:
        intent_recognizer: Base IntentRecognizer
        sentiment_analyzer: Base SentimentAnalyzer
        risk_assessor: Base RiskAssessor

    Returns:
        Tuple of memory-aware agents
    """
    return (
        MemoryAwareIntentRecognizer(intent_recognizer),
        MemoryAwareSentimentAnalyzer(sentiment_analyzer),
        MemoryAwareRiskAssessor(risk_assessor),
    )
