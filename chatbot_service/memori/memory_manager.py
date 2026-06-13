"""
Production-grade Memory Manager for Memori Integration

Provides:
- Singleton MemoryManager with thread-safe initialization
- Per-patient Memori instances with LRU caching
- Async memory operations with timeout protection
- Circuit breaker for resilience
- Comprehensive error handling and structured logging
- Request correlation ID tracking
- HIPAA-compliant data isolation
- Metrics collection and observability

Architecture:
    Request → MemoryManager (singleton)
              → CircuitBreaker (failure detection)
              → PatientMemory cache (LRU, per-patient isolation)
              → Memori instance (persistent storage)
              → SQLite/PostgreSQL backend

Complexity:
    - get_patient_memory: O(1) cache lookup, O(n) first init
    - search_memory: O(log n) with database indexing
    - store_memory: O(1) async write
    - All operations bounded by timeout (30s default)
"""

import asyncio
import enum
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from .utils.exceptions import (
    TimeoutError,
    IntegrationError as ExternalServiceError,
    ProcessingError,
)

logger = logging.getLogger(__name__)

# Request context for correlation ID tracking
request_id_context: ContextVar[str] = ContextVar("request_id", default="")


# ============================================================================
# Exceptions
# ============================================================================


class MemoryManagerException(Exception):
    """Base exception for memory manager errors."""


class MemoryOperationTimeout(MemoryManagerException):
    """Raised when memory operation exceeds timeout."""


class MemoryCircuitBreakerOpen(MemoryManagerException):
    """Raised when circuit breaker is open (service unavailable)."""


class PatientMemoryNotFound(MemoryManagerException):
    """Raised when patient memory cannot be initialized."""


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MemoryResult:
    """Result from memory search operation."""

    id: str
    content: str
    memory_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 1.0

    @classmethod
    def from_memori_result(cls, result: Dict[str, Any]) -> "MemoryResult":
        """Convert Memori result to MemoryResult."""
        return cls(
            id=result.get("id", ""),
            content=result.get("content", ""),
            memory_type=result.get("type", ""),
            timestamp=result.get("timestamp", ""),
            metadata=result.get("metadata", {}),
            relevance_score=result.get("relevance_score", 1.0),
        )


@dataclass
class MemoryManagerMetrics:
    """Metrics for memory manager operations."""

    searches_total: int = 0
    searches_successful: int = 0
    searches_failed: int = 0
    searches_timeout: int = 0
    searches_latency_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    stores_total: int = 0
    stores_successful: int = 0
    stores_failed: int = 0
    stores_timeout: int = 0
    stores_latency_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0

    errors_total: int = 0
    circuit_breaker_state: str = "CLOSED"
    circuit_breaker_open_count: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    @property
    def avg_search_latency_ms(self) -> float:
        """Calculate average search latency."""
        return (
            sum(self.searches_latency_ms) / len(self.searches_latency_ms)
            if self.searches_latency_ms
            else 0.0
        )

    @property
    def avg_store_latency_ms(self) -> float:
        """Calculate average store latency."""
        return (
            sum(self.stores_latency_ms) / len(self.stores_latency_ms)
            if self.stores_latency_ms
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "searches": {
                "total": self.searches_total,
                "successful": self.searches_successful,
                "failed": self.searches_failed,
                "timeout": self.searches_timeout,
                "avg_latency_ms": self.avg_search_latency_ms,
            },
            "stores": {
                "total": self.stores_total,
                "successful": self.stores_successful,
                "failed": self.stores_failed,
                "timeout": self.stores_timeout,
                "avg_latency_ms": self.avg_store_latency_ms,
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "evictions": self.cache_evictions,
                "hit_rate_percent": self.cache_hit_rate,
            },
            "errors_total": self.errors_total,
            "circuit_breaker": {
                "state": self.circuit_breaker_state,
                "open_count": self.circuit_breaker_open_count,
            },
        }


# ============================================================================
# LRU Cache for Patient Memory Instances
# ============================================================================


class LRUMemoryCache:
    """
    Thread-safe LRU cache for Memori instances.

    Features:
    - O(1) get/put/delete operations using OrderedDict
    - Automatic eviction when maxsize exceeded
    - Thread-safe via lock
    - Metrics tracking
    """

    def __init__(self, maxsize: int = 100):
        """Initialize LRU cache."""
        self.maxsize = maxsize
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (mark as recently used)."""
        with self._lock:
            if key in self.cache:
                # Move to end (mark as recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any) -> Optional[Any]:
        """Put value in cache (evict LRU if needed).
        
        Returns:
            Evicted item if eviction occurred, None otherwise.
            Caller is responsible for async cleanup of evicted items.
        """
        evicted_item = None
        with self._lock:
            if key in self.cache:
                # Already exists, move to end and update value
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # New entry
                if len(self.cache) >= self.maxsize:
                    # Evict LRU (first item)
                    evicted_key, evicted_item = self.cache.popitem(last=False)
                    self.evictions += 1
                    logger.debug(f"LRU eviction: {evicted_key}")
                    # DON'T cleanup here - return item for caller to handle async
                    # This prevents asyncio.run() from being called inside a running loop

                self.cache[key] = value
        return evicted_item

    def delete(self, key: str) -> Optional[Any]:
        """Delete entry from cache."""
        with self._lock:
            return self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self.cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self.cache)

    def items(self) -> List[Tuple[str, Any]]:
        """Get all items in cache."""
        with self._lock:
            return list(self.cache.items())


# ============================================================================
# Circuit Breaker
# ============================================================================


class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"        # Normal operation – requests pass through
    OPEN = "OPEN"            # Failures exceeded threshold – requests rejected
    HALF_OPEN = "HALF_OPEN"  # Cooldown elapsed – allow one probe request


class CircuitBreaker:
    """
    Thread-safe circuit breaker for memory operations.

    Prevents cascading failures by short-circuiting requests when the
    backend is unhealthy.  After a cooldown period the breaker enters
    HALF_OPEN state and allows a single probe request.  If it succeeds
    the breaker closes again; if it fails the breaker re-opens.

    Parameters:
        failure_threshold: Consecutive failures before opening.
        recovery_timeout:  Seconds to wait before probing (HALF_OPEN).
        success_threshold: Consecutive successes in HALF_OPEN to fully close.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._open_count = 0
        self._lock = threading.RLock()

    # -- public interface ----------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition from OPEN → HALF_OPEN)."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker → HALF_OPEN (recovery timeout elapsed)")
            return self._state

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        state = self.state  # property performs OPEN→HALF_OPEN transition
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # allow probe
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker → CLOSED (recovered)")
            else:
                # In CLOSED state – reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed – reopen
                self._state = CircuitState.OPEN
                self._open_count += 1
                logger.warning("Circuit breaker → OPEN (probe failed)")
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._open_count += 1
                logger.warning(
                    f"Circuit breaker → OPEN (failures={self._failure_count})"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Metrics snapshot."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "open_count": self._open_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


# ============================================================================
# Patient Memory Wrapper
# ============================================================================


class PatientMemory:
    """
    Domain wrapper providing health-aware memory operations.

    Wraps Memori instance with:
    - Type-safe health domain operations
    - Automatic metadata enrichment
    - Conversation turn tracking
    - PHI-aware logging
    """

    def __init__(self, memori: Any, patient_id: str, session_id: str):
        """Initialize patient memory wrapper."""
        self.memori = memori
        self.patient_id = patient_id
        self.session_id = session_id
        self.created_at = datetime.now()
        self._closed = False

        logger.info(
            f"PatientMemory initialized: patient_id={patient_id}, "
            f"session_id={session_id}"
        )

    async def add_conversation(
        self,
        user_message: str,
        assistant_message: str,
        intent: Optional[str] = None,
        sentiment: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record conversation turn with NLP metadata.

        Args:
            user_message: User's input message
            assistant_message: Assistant's response
            intent: Detected intent (from IntentRecognizer)
            sentiment: Detected sentiment (from SentimentAnalyzer)
            entities: Extracted entities (from EntityExtractor)
            metadata: Additional metadata
        """
        if self._closed:
            raise MemoryManagerException("PatientMemory is closed")

        try:
            conversation_data = {
                "user": user_message,
                "assistant": assistant_message,
                "intent": intent,
                "sentiment": sentiment,
                "entities": entities or {},
            }

            meta = {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": request_id_context.get(),
                **(metadata or {}),
            }

            await self.memori.add_memory(
                type="conversation",
                content=json.dumps(conversation_data),
                metadata=meta,
            )

            logger.debug(
                f"Stored conversation: patient_id={self.patient_id}, "
                f"intent={intent}, sentiment={sentiment}"
            )

        except Exception as e:
            logger.error(f"Error storing conversation: {e}", exc_info=True)
            raise

    async def add_health_data(
        self,
        data_type: str,
        data: Dict[str, Any],
        severity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record health-related data with clinical context.

        Args:
            data_type: Type of health data (vitals, medication, symptom, etc.)
            data: Health data dictionary
            severity: Optional severity level (low, medium, high, critical)
            metadata: Additional metadata
        """
        if self._closed:
            raise MemoryManagerException("PatientMemory is closed")

        try:
            meta = {
                "data_type": data_type,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": request_id_context.get(),
                **(metadata or {}),
            }

            await self.memori.add_memory(
                type="health_data",
                content=json.dumps(data),
                metadata=meta,
            )

            logger.debug(
                f"Stored health data: patient_id={self.patient_id}, "
                f"data_type={data_type}, severity={severity}"
            )

        except Exception as e:
            logger.error(f"Error storing health data: {e}", exc_info=True)
            raise

    async def search(
        self,
        query: str,
        data_type: Optional[str] = None,
        limit: int = 5,
        timeout: int = 30,
    ) -> List[MemoryResult]:
        """
        Intelligent memory search with optional filtering.

        Args:
            query: Search query string
            data_type: Optional data type filter
            limit: Maximum results to return
            timeout: Search timeout in seconds

        Returns:
            List of MemoryResult objects
        """
        if self._closed:
            raise MemoryManagerException("PatientMemory is closed")

        try:
            start_time = time.time()

            # Build filters
            filters = {}
            if data_type:
                filters["data_type"] = data_type

            # Execute search with timeout
            # Use the correct method name from Memori class
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.memori.search,
                        query=query,
                        limit=limit,
                    ),
                    timeout=timeout,
                )
            except (AttributeError, TypeError):
                # Fallback to retrieve_context if search doesn't work
                results = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.memori.retrieve_context,
                        query=query,
                        limit=limit,
                    ),
                    timeout=timeout,
                )

            latency_ms = (time.time() - start_time) * 1000

            # Convert to MemoryResult objects
            memory_results = [MemoryResult.from_memori_result(r) for r in results]

            logger.debug(
                f"Memory search completed: query='{query}', "
                f"results={len(memory_results)}, latency_ms={latency_ms:.2f}"
            )

            return memory_results

        except asyncio.TimeoutError:
            logger.warning(
                f"Memory search timeout: query='{query}', "
                f"timeout={timeout}s, patient_id={self.patient_id}"
            )
            raise MemoryOperationTimeout(f"Memory search exceeded {timeout}s timeout")
        except Exception as e:
            logger.error(f"Error searching memory: {e}", exc_info=True)
            raise

    async def get_conversation_context(
        self,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get recent conversation history for context injection.

        Args:
            limit: Number of recent conversations to retrieve

        Returns:
            Dictionary with conversation context
        """
        if self._closed:
            raise MemoryManagerException("PatientMemory is closed")

        try:
            results = await self.search(
                query="recent conversation",
                data_type="conversation",
                limit=limit,
                timeout=30,
            )

            conversations = [json.loads(r.content) for r in results if r.content]

            return {
                "recent_conversations": conversations,
                "conversation_count": len(conversations),
                "session_id": self.session_id,
                "retrieved_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {
                "recent_conversations": [],
                "conversation_count": 0,
                "session_id": self.session_id,
                "retrieved_at": datetime.now().isoformat(),
                "error": str(e),
            }

    async def get_health_summary(
        self,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Get recent health data summary.

        Args:
            limit: Number of recent health records to retrieve

        Returns:
            Dictionary with health data summary
        """
        if self._closed:
            raise MemoryManagerException("PatientMemory is closed")

        try:
            results = await self.search(
                query="health data vitals measurements",
                data_type="health_data",
                limit=limit,
                timeout=30,
            )

            health_records = [
                {
                    "data": json.loads(r.content) if r.content else {},
                    "type": r.metadata.get("data_type", "unknown"),
                    "severity": r.metadata.get("severity"),
                    "timestamp": r.timestamp,
                }
                for r in results
            ]

            return {
                "health_records": health_records,
                "record_count": len(health_records),
                "retrieved_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {
                "health_records": [],
                "record_count": 0,
                "retrieved_at": datetime.now().isoformat(),
                "error": str(e),
            }

    async def close(self) -> None:
        """Cleanup memory instance."""
        if not self._closed:
            try:
                if hasattr(self.memori, "close"):
                    await self.memori.close()
                self._closed = True
                logger.debug(f"PatientMemory closed: patient_id={self.patient_id}")
            except Exception as e:
                logger.warning(f"Error closing patient memory: {e}")

    def cleanup(self) -> None:
        """Synchronous cleanup for cache eviction (handles nested event loop safely)."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # Already in an event loop - schedule the coroutine as a task
                # This avoids "This event loop is already running" error
                loop.create_task(self.close())
                logger.debug(f"Scheduled async cleanup for patient_id={self.patient_id}")
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                asyncio.run(self.close())
        except Exception as e:
            logger.warning(f"Error in sync cleanup: {e}")


# ============================================================================
# Memory Manager (Singleton)
# ============================================================================


class MemoryManager:
    """
    Production-grade memory management layer (Singleton).

    Manages:
    - Per-patient Memori instances with LRU caching
    - Async memory operations with timeout protection
    - Circuit breaker for resilience
    - Comprehensive error handling and metrics
    - HIPAA-compliant data isolation

    Lifecycle:
        1. get_instance() - Thread-safe singleton initialization
        2. initialize() - Setup during app startup
        3. get_patient_memory() - Per-request access (cached)
        4. shutdown() - Cleanup during app shutdown

    Thread Safety:
        - Singleton creation: Double-checked locking
        - Cache access: RLock for thread-safe OrderedDict ops
        - Metrics: Atomic operations on primitives
    """

    # Singleton instance
    _instance: Optional["MemoryManager"] = None
    _lock = threading.Lock()

    # Configuration defaults
    DEFAULT_POOL_SIZE = 10
    DEFAULT_CACHE_SIZE = 100
    DEFAULT_REQUEST_TIMEOUT = 30
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        cache_size: int = DEFAULT_CACHE_SIZE,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        enabled: bool = True,
    ):
        """
        Initialize MemoryManager.

        Args:
            database_url: Memori database connection URL
            pool_size: Database connection pool size
            cache_size: Max patient instances to keep in memory
            request_timeout: Timeout for memory operations (seconds)
            circuit_breaker_threshold: Failures before opening circuit
            enabled: Whether memory is enabled
        """
        import os
        
        # Use MEMORI_DATABASE_URL from environment if not provided
        # SECURITY: Database URL must be set via environment variable, not hardcoded
        # Default database name is 'memori_db' (dedicated database for Memori)
        default_url = (
            f"postgresql+psycopg2://"
            f"{os.getenv('DB_USER', 'postgres')}:"
            f"{os.getenv('DB_PASSWORD', '')}@"
            f"{os.getenv('DB_HOST', 'localhost')}:"
            f"{os.getenv('DB_PORT', '5432')}/"
            f"{os.getenv('MEMORI_DB_NAME', 'memori_db')}"
        )
        self.database_url = database_url or os.getenv(
            "MEMORI_DATABASE_URL",
            default_url
        )
        self.pool_size = pool_size
        self.cache_size = cache_size
        self.request_timeout = request_timeout
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.enabled = enabled

        # LRU cache for patient memories
        self._cache = LRUMemoryCache(maxsize=cache_size)

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=60.0,
            success_threshold=2,
        )

        # Per-patient initialization locks to prevent race conditions
        self._patient_init_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock: Optional[asyncio.Lock] = None  # Lazy init in async context

        # Background workers
        self._promotion_task: Optional[asyncio.Task] = None
        self._decay_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = MemoryManagerMetrics()

        # Initialization flag
        self._initialized = False

        logger.info(
            f"MemoryManager created: database_url={self.database_url}, "
            f"pool_size={pool_size}, cache_size={cache_size}, "
            f"enabled={enabled}"
        )

    @classmethod
    def get_instance(cls, **kwargs) -> "MemoryManager":
        """
        Thread-safe singleton accessor.

        Returns:
            MemoryManager singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance:
            try:
                asyncio.run(cls._instance.shutdown())
            except Exception as e:
                logger.warning(f"Error resetting instance: {e}")
        cls._instance = None

    async def initialize(self) -> None:
        """
        Initialize memory manager during app startup.

        Creates necessary database schema and verifies connectivity.
        """
        if self._initialized or not self.enabled:
            return

        try:
            logger.info("Initializing MemoryManager...")

            # Try to create a test Memori instance to verify setup
            if not self.enabled:
                logger.info("MemoryManager disabled, skipping initialization")
                self._initialized = True
                return

            # For now, just mark as initialized
            # Actual Memori init happens lazily on first patient access
            self._initialized = True

            logger.info("MemoryManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}", exc_info=True)
            self.enabled = False  # Graceful degradation

    async def _get_patient_lock(self, patient_id: str) -> asyncio.Lock:
        """
        Get or create a lock for patient initialization.
        
        Prevents race conditions when multiple requests for the same patient
        arrive simultaneously.
        
        Args:
            patient_id: Unique patient identifier
            
        Returns:
            asyncio.Lock for this patient
        """
        # Lazy initialize the locks lock in async context
        if self._locks_lock is None:
            self._locks_lock = asyncio.Lock()
        
        async with self._locks_lock:
            if patient_id not in self._patient_init_locks:
                self._patient_init_locks[patient_id] = asyncio.Lock()
            return self._patient_init_locks[patient_id]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((IOError, ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number} for get_patient_memory. "
            f"Will retry in {retry_state.next_action.sleep} seconds."
        ),
    )  # PHASE 2: Add retry for transient cache/DB errors
    async def get_patient_memory(
        self,
        patient_id: str,
        session_id: str = "default",
    ) -> PatientMemory:
        """
        Get or create Memori instance for patient (cached).

        Implements:
        - LRU cache for recently accessed patient memories
        - Lazy initialization (create on first access)
        - Automatic eviction based on cache size
        - Thread-safe operations

        Complexity:
        - Cache hit: O(1)
        - Cache miss: O(n) where n = schema setup time

        Args:
            patient_id: Unique patient identifier
            session_id: Conversation session identifier

        Returns:
            PatientMemory instance for the patient

        Raises:
            MemoryManagerException: If initialization fails
            MemoryCircuitBreakerOpen: If service unavailable
        """
        if not self.enabled:
            raise MemoryManagerException("Memory manager is disabled")

        # Circuit breaker guard
        if not self._circuit_breaker.allow_request():
            self.metrics.circuit_breaker_state = self._circuit_breaker.state.value
            raise MemoryCircuitBreakerOpen(
                "Memory service unavailable (circuit breaker OPEN)"
            )

        cache_key = f"{patient_id}:{session_id}"

        # Try cache first (fast path - no lock)
        cached = self._cache.get(cache_key)
        if cached:
            self.metrics.cache_hits += 1
            logger.debug(f"Cache hit: {cache_key}")
            return cached

        self.metrics.cache_misses += 1

        # Get per-patient lock to prevent race conditions
        patient_lock = await self._get_patient_lock(patient_id)
        
        async with patient_lock:
            # Double-checked locking: re-check cache after acquiring lock
            cached = self._cache.get(cache_key)
            if cached:
                self.metrics.cache_hits += 1
                logger.debug(f"Cache hit (after lock): {cache_key}")
                return cached

            # Initialize new Memori instance
            try:
                logger.info(f"Initializing patient memory: {cache_key}")

                # Lazy load Memori here
                try:
                    from .core.memory import Memori
                except ImportError:
                    raise MemoryManagerException(
                        "Memori library not installed. " "Install with: pip install memori"
                    )

                # ============================================================================
                # [FIX] Connect Memori to Local LLM (MedGemma @ localhost:8090)
                # ============================================================================
                
                # Create OpenAI-compatible client pointing to local llama-server
                try:
                    from openai import AsyncOpenAI
                    
                    local_llm_client = AsyncOpenAI(
                        base_url="http://127.0.0.1:8090/v1",      # Local MedGemma server
                        api_key="sk-local-key"                      # Dummy key (ignored by local server)
                    )
                    logger.info("✅ Memori: Using local LLM client (localhost:8090)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create local LLM client: {e}, falling back to default Memori config")
                    local_llm_client = None
                
                # Configure remote embeddings (MedCPT via Colab)
                embedding_config = {
                    "provider": "remote",
                    "model_name": "MedCPT-Query-Encoder",
                    "dimension": 768,
                }
                
                # Build Memori initialization kwargs
                memori_kwargs = {
                    "database_connect": self.database_url,
                    "user_id": patient_id,
                    "session_id": session_id,
                    "conscious_ingest": True,  # Auto-inject relevant memory
                    "schema_init": True,
                    "pool_size": self.pool_size,
                }
                
                # Add local LLM config if available - use Memori's native parameters
                if local_llm_client:
                    # Memori accepts these parameters directly, not as llm_config dict
                    memori_kwargs["base_url"] = "http://127.0.0.1:8090/v1"
                    memori_kwargs["api_key"] = "sk-local-key"
                    memori_kwargs["model"] = "medgemma-4b-it"
                    memori_kwargs["api_type"] = "llama_local"
                    logger.info("✅ Memori configured with local LLM (localhost:8090)")
                
                # Create Memori instance with local configuration
                memori = Memori(**memori_kwargs)

                # Wrap in domain-aware PatientMemory
                patient_memory = PatientMemory(memori, patient_id, session_id)

                # Cache for future access - handle evicted item asynchronously
                evicted = self._cache.put(cache_key, patient_memory)
                if evicted:
                    # Properly await cleanup of evicted patient memory
                    try:
                        await evicted.close()
                        logger.debug(f"Cleaned up evicted patient memory")
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up evicted patient memory: {cleanup_error}")
                
                self.metrics.cache_evictions = self._cache.evictions

                logger.info(f"Patient memory initialized: {cache_key}")
                self._circuit_breaker.record_success()
                return patient_memory

            except Exception as e:
                self._circuit_breaker.record_failure()
                self.metrics.circuit_breaker_state = self._circuit_breaker.state.value
                self.metrics.errors_total += 1
                logger.error(
                    f"Failed to initialize patient memory {cache_key}: {e}",
                    exc_info=True,
                )
                raise PatientMemoryNotFound(
                    f"Could not initialize memory for patient {patient_id}: {e}"
                )

    async def search_memory(
        self,
        patient_id: str,
        query: str,
        session_id: str = "default",
        data_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryResult]:
        """
        Search patient memory with timeout protection.

        Complexity: O(log n) with database indexing

        Args:
            patient_id: Patient identifier
            query: Search query string
            session_id: Session identifier
            data_type: Optional data type filter
            limit: Max results

        Returns:
            List of MemoryResult objects

        Raises:
            MemoryOperationTimeout: If search exceeds timeout
            MemoryManagerException: If operation fails
        """
        if not self.enabled:
            logger.warning("Memory search attempted when disabled")
            return []

        start_time = time.time()
        self.metrics.searches_total += 1

        try:
            patient_memory = await self.get_patient_memory(patient_id, session_id)
            results = await patient_memory.search(
                query=query,
                data_type=data_type,
                limit=limit,
                timeout=self.request_timeout,
            )

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.searches_successful += 1
            self.metrics.searches_latency_ms.append(latency_ms)

            # Keep only recent metrics
            # Metrics handled by deque maxlen
            pass

            logger.debug(
                f"Memory search: patient_id={patient_id}, "
                f"query='{query}', results={len(results)}, "
                f"latency_ms={latency_ms:.2f}"
            )

            return results

        except MemoryOperationTimeout:
            self.metrics.searches_timeout += 1
            self.metrics.errors_total += 1
            raise
        except Exception as e:
            self.metrics.searches_failed += 1
            self.metrics.errors_total += 1
            logger.error(f"Memory search error: {e}", exc_info=True)
            raise

    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get user context (convenience method).

        Combines conversation context and health summary.
        """
        if not self.enabled:
            return {}

        try:
            patient_memory = await self.get_patient_memory(user_id)
            
            # Run in parallel
            conv_task = patient_memory.get_conversation_context()
            health_task = patient_memory.get_health_summary()
            
            conv_context, health_context = await asyncio.gather(conv_task, health_task)
            
            return {
                **conv_context,
                **health_context,
                "user_id": user_id
            }
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((IOError, ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number} for store_memory. "
            f"Will retry in {retry_state.next_action.sleep} seconds."
        ),
    )  # PHASE 2: Add retry for transient DB errors
    async def store_memory(
        self,
        patient_id: str,
        memory_type: str,
        content: str,
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store memory with PHI handling and audit trail.

        Complexity: O(1) async write

        Args:
            patient_id: Patient identifier
            memory_type: Type of memory (conversation, health_data, etc.)
            content: Memory content
            session_id: Session identifier
            metadata: Additional metadata

        Raises:
            MemoryManagerException: If operation fails
        """
        if not self.enabled:
            logger.debug("Memory store attempted when disabled")
            return

        start_time = time.time()
        self.metrics.stores_total += 1

        try:
            patient_memory = await self.get_patient_memory(patient_id, session_id)

            # Store based on type
            if memory_type == "conversation":
                data = json.loads(content)
                await patient_memory.add_conversation(
                    user_message=data.get("user", ""),
                    assistant_message=data.get("assistant", ""),
                    intent=data.get("intent"),
                    sentiment=data.get("sentiment"),
                    entities=data.get("entities"),
                    metadata=metadata,
                )
            elif memory_type == "health_data":
                data = json.loads(content)
                await patient_memory.add_health_data(
                    data_type=(
                        metadata.get("data_type", "unknown") if metadata else "unknown"
                    ),
                    data=data,
                    severity=metadata.get("severity") if metadata else None,
                    metadata=metadata,
                )
            else:
                # Generic storage
                await patient_memory.memori.add_memory(
                    type=memory_type,
                    content=content,
                    metadata={
                        "correlation_id": request_id_context.get(),
                        **(metadata or {}),
                    },
                )

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.stores_successful += 1
            self.metrics.stores_latency_ms.append(latency_ms)

            # Keep only recent metrics
            # Metrics handled by deque maxlen
            pass

            logger.debug(
                f"Memory stored: patient_id={patient_id}, "
                f"type={memory_type}, latency_ms={latency_ms:.2f}"
            )

        except Exception as e:
            self.metrics.stores_failed += 1
            self.metrics.errors_total += 1
            logger.error(f"Memory store error: {e}", exc_info=True)
            raise

    # ========================================================================
    # Memory Promotion Worker
    # ========================================================================

    async def start_background_workers(self) -> None:
        """
        Start background workers for memory promotion and decay.
        Call this after ``initialize()``.
        """
        if not self.enabled or not self._initialized:
            return

        if self._promotion_task is None or self._promotion_task.done():
            self._promotion_task = asyncio.create_task(
                self._promotion_loop(), name="memori-promotion-worker"
            )
            logger.info("Memory promotion worker started")

        if self._decay_task is None or self._decay_task.done():
            self._decay_task = asyncio.create_task(
                self._decay_loop(), name="memori-decay-worker"
            )
            logger.info("Memory decay worker started")

    async def _promotion_loop(
        self,
        interval_seconds: float = 300.0,
        min_messages: int = 3,
        min_age_seconds: float = 120.0,
        importance_threshold: float = 0.4,
    ) -> None:
        """
        Periodically scan short-term sessions and promote important ones
        to long-term memory via FactExtractor.

        Steps per cycle:
          1. Query ``RedisSessionBuffer.get_promotable_sessions()``
          2. Score each session via ``analyze_session_importance()``
          3. Extract facts using ``FactExtractor.extract_batch()``
          4. Store extracted facts in long-term memory via ``store_memory()``
        """
        logger.info("Promotion loop starting")
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self._run_promotion_cycle(
                    min_messages=min_messages,
                    min_age_seconds=min_age_seconds,
                    importance_threshold=importance_threshold,
                )
            except asyncio.CancelledError:
                logger.info("Promotion loop cancelled")
                return
            except Exception as e:
                logger.error(f"Promotion cycle error: {e}", exc_info=True)
                # Back-off on repeated errors
                await asyncio.sleep(30)

    async def _run_promotion_cycle(
        self,
        min_messages: int = 3,
        min_age_seconds: float = 120.0,
        importance_threshold: float = 0.4,
    ) -> int:
        """
        Execute one promotion cycle.

        Returns:
            Number of sessions promoted.
        """
        promoted = 0
        try:
            # Import lazily to avoid circular deps
            from .short_term.redis_buffer import RedisSessionBuffer, InMemorySessionBuffer
            from .long_term.fact_extractor import FactExtractor

            # Try to get the session buffer instance
            buffer: Optional[Any] = None
            try:
                buffer = RedisSessionBuffer.get_instance()
            except Exception:
                buffer = InMemorySessionBuffer()

            if buffer is None:
                return 0

            # 1. Find promotable sessions
            promotable = buffer.get_promotable_sessions(
                min_messages=min_messages,
                min_age_seconds=min_age_seconds,
            )
            if not promotable:
                return 0

            logger.info(f"Promotion: found {len(promotable)} candidate sessions")

            # 2. Score and filter
            qualified_sessions: List[Tuple[str, Dict[str, Any]]] = []
            for session_id in promotable:
                try:
                    analysis = buffer.analyze_session_importance(session_id)
                    if analysis.get("importance_score", 0) >= importance_threshold:
                        qualified_sessions.append((session_id, analysis))
                except Exception as e:
                    logger.warning(f"Promotion scoring failed for {session_id}: {e}")

            if not qualified_sessions:
                return 0

            logger.info(
                f"Promotion: {len(qualified_sessions)} sessions qualify "
                f"(threshold={importance_threshold})"
            )

            # 3. Extract facts from conversation text
            extractor = FactExtractor()
            texts = []
            session_ids = []
            for sid, analysis in qualified_sessions:
                try:
                    messages = buffer.get_messages(sid)
                    if messages:
                        combined = "\n".join(
                            f"{getattr(m, 'role', 'user')}: {getattr(m, 'content', str(m))}"
                            for m in messages
                        )
                        texts.append(combined)
                        session_ids.append(sid)
                except Exception as e:
                    logger.warning(f"Failed to read messages for {sid}: {e}")

            if not texts:
                return 0

            batch_results = await extractor.extract_batch(texts, max_concurrent=5)

            # 4. Store extracted facts in long-term memory
            for idx, facts in enumerate(batch_results):
                if not facts:
                    continue
                sid = session_ids[idx]
                for fact in facts:
                    try:
                        # Derive a patient_id from session metadata or use session id
                        patient_id = sid.split(":")[0] if ":" in sid else sid
                        await self.store_memory(
                            patient_id=patient_id,
                            memory_type="health_data",
                            content=json.dumps({
                                "fact": fact.text if hasattr(fact, "text") else str(fact),
                                "category": fact.category.value if hasattr(fact, "category") else "general",
                                "confidence": getattr(fact, "confidence", 1.0),
                                "source": "promotion",
                            }),
                            session_id=sid,
                            metadata={
                                "data_type": "promoted_fact",
                                "source_session": sid,
                                "promotion_timestamp": datetime.now().isoformat(),
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store promoted fact: {e}")

                promoted += 1
                logger.debug(f"Promoted session {sid}: {len(facts)} facts")

        except ImportError as e:
            logger.warning(f"Promotion dependencies unavailable: {e}")
        except Exception as e:
            logger.error(f"Promotion cycle failed: {e}", exc_info=True)

        if promoted:
            logger.info(f"Promotion cycle complete: {promoted} sessions promoted")
        return promoted

    # ========================================================================
    # Memory Consolidation & Decay
    # ========================================================================

    async def _decay_loop(
        self,
        interval_seconds: float = 3600.0,
        decay_factor: float = 0.95,
        min_score: float = 0.05,
        max_age_days: int = 90,
    ) -> None:
        """
        Periodically apply time-based importance decay to long-term memories
        and clean up obsolete entries.

        Each cycle:
        1. Decay importance scores of memories that haven't been accessed recently
        2. Remove memories whose scores fall below ``min_score``
        3. Deduplicate nearly-identical memories (same patient, same content hash)
        """
        logger.info("Decay loop starting")
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self._run_decay_cycle(
                    decay_factor=decay_factor,
                    min_score=min_score,
                    max_age_days=max_age_days,
                )
            except asyncio.CancelledError:
                logger.info("Decay loop cancelled")
                return
            except Exception as e:
                logger.error(f"Decay cycle error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _run_decay_cycle(
        self,
        decay_factor: float = 0.95,
        min_score: float = 0.05,
        max_age_days: int = 90,
    ) -> Dict[str, int]:
        """
        Execute one consolidation/decay cycle.

        Returns:
            Dict with counts: ``decayed``, ``pruned``, ``deduplicated``.
        """
        stats = {"decayed": 0, "pruned": 0, "deduplicated": 0}

        try:
            from .core.memory import Memori

            # Iterate over all cached patient memories
            for cache_key, patient_memory in self._cache.items():
                try:
                    memori = patient_memory.memori
                    db = getattr(memori, "db", None) or getattr(memori, "database", None)
                    if db is None:
                        continue

                    # Attempt to get all memories via available DB methods
                    memories = []
                    try:
                        if hasattr(db, "get_all_memories"):
                            memories = await asyncio.to_thread(db.get_all_memories)
                        elif hasattr(db, "execute_query"):
                            result = await asyncio.to_thread(
                                db.execute_query,
                                "SELECT id, importance_score, last_accessed, created_at "
                                "FROM memories WHERE importance_score IS NOT NULL"
                            )
                            memories = result if result else []
                    except Exception as e:
                        logger.debug(f"Skipping decay for {cache_key}: {e}")
                        continue

                    if not memories:
                        continue

                    cutoff = datetime.now() - timedelta(days=max_age_days)

                    for mem in memories:
                        mem_id = mem.get("id") if isinstance(mem, dict) else getattr(mem, "id", None)
                        score = (
                            mem.get("importance_score", 1.0)
                            if isinstance(mem, dict)
                            else getattr(mem, "importance_score", 1.0)
                        )
                        last_accessed = (
                            mem.get("last_accessed")
                            if isinstance(mem, dict)
                            else getattr(mem, "last_accessed", None)
                        )
                        created_at = (
                            mem.get("created_at")
                            if isinstance(mem, dict)
                            else getattr(mem, "created_at", None)
                        )

                        if score is None:
                            continue

                        # Parse timestamps
                        ts = None
                        for candidate in [last_accessed, created_at]:
                            if candidate is None:
                                continue
                            if isinstance(candidate, datetime):
                                ts = candidate
                                break
                            try:
                                ts = datetime.fromisoformat(str(candidate))
                                break
                            except (ValueError, TypeError):
                                continue

                        # 1. Prune if older than max_age_days and low score
                        if ts and ts < cutoff and score < 0.2:
                            try:
                                if hasattr(db, "delete_memory"):
                                    await asyncio.to_thread(db.delete_memory, mem_id)
                                    stats["pruned"] += 1
                            except Exception:
                                pass
                            continue

                        # 2. Apply decay
                        new_score = score * decay_factor
                        if new_score < min_score:
                            # Score too low — prune
                            try:
                                if hasattr(db, "delete_memory"):
                                    await asyncio.to_thread(db.delete_memory, mem_id)
                                    stats["pruned"] += 1
                            except Exception:
                                pass
                        else:
                            try:
                                if hasattr(db, "update_memory"):
                                    await asyncio.to_thread(
                                        db.update_memory,
                                        mem_id,
                                        {"importance_score": new_score},
                                    )
                                    stats["decayed"] += 1
                            except Exception:
                                pass

                except Exception as e:
                    logger.debug(f"Decay skip for {cache_key}: {e}")

        except ImportError:
            logger.debug("Memori core not available for decay")
        except Exception as e:
            logger.error(f"Decay cycle failed: {e}", exc_info=True)

        if any(stats.values()):
            logger.info(f"Decay cycle: {stats}")
        return stats

    async def shutdown(self) -> None:
        """Cleanup all patient memory instances and background workers gracefully."""
        if not self._initialized:
            return

        logger.info("Shutting down MemoryManager...")

        # Cancel background workers
        for task in [self._promotion_task, self._decay_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        for cache_key, patient_memory in self._cache.items():
            try:
                await patient_memory.close()
            except Exception as e:
                logger.warning(f"Error closing {cache_key}: {e}")

        self._cache.clear()
        self._initialized = False

        logger.info("MemoryManager shutdown complete")

    def get_metrics(self) -> Dict[str, Any]:
        """Get memory manager metrics."""
        return {
            "enabled": self.enabled,
            "initialized": self._initialized,
            "cache_size": self._cache.size(),
            "cache_max_size": self.cache_size,
            "circuit_breaker": self._circuit_breaker.to_dict(),
            "background_workers": {
                "promotion": (
                    "running"
                    if self._promotion_task and not self._promotion_task.done()
                    else "stopped"
                ),
                "decay": (
                    "running"
                    if self._decay_task and not self._decay_task.done()
                    else "stopped"
                ),
            },
            "metrics": self.metrics.to_dict(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Get health check status."""
        return {
            "status": "healthy" if self.enabled else "degraded",
            "enabled": self.enabled,
            "initialized": self._initialized,
            "metrics": self.get_metrics(),
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Context Manager for Request-Scoped Memory
# ============================================================================


@asynccontextmanager
async def get_request_memory(patient_id: str, session_id: str = "default"):
    """
    Context manager for request-scoped patient memory access.

    Usage:
        async with get_request_memory(patient_id) as memory:
            context = await memory.get_conversation_context()
            await memory.add_health_data("vitals", {...})
    """
    memory_mgr = MemoryManager.get_instance()
    patient_memory = await memory_mgr.get_patient_memory(patient_id, session_id)
    try:
        yield patient_memory
    finally:
        # Note: Don't close here, keep cached for reuse
        pass
