"""
Performance Optimization for Memori Integration

Provides:
1. Database query optimization with pagination and lazy loading
2. Multi-tier caching strategy (in-memory + Redis optional)
3. Batch memory operations for bulk processing
4. Connection pool sizing recommendations based on workload
5. Compression for large memory entries
6. Query result deduplication and filtering
7. Bulk insertion with batching

Optimizations:
- Paginated search results (default 50 per page)
- Lazy loading of conversation context
- Redis caching layer (optional, falls back gracefully)
- Batch store operations (50-500 entries per batch)
- ZSTD compression for entries > 1KB
- Query result deduplication
- Connection pool adaptive sizing

Performance Targets:
- Single search: < 100ms (with caching < 50ms)
- Batch store (100 entries): < 500ms
- Cache hit rate: > 80% after warmup
- Memory footprint: < 500MB for 1000 active patients
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional compression library
try:
    import zstandard as zstd

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    logger.debug("zstandard not available, compression disabled")


# ============================================================================
# Compression Utilities
# ============================================================================


class CompressionManager:
    """
    Manages compression for large memory entries.

    Uses ZSTD (Zstandard) for better compression ratio than gzip:
    - Compression ratio: ~40-60% for text
    - Speed: Fast compression/decompression
    - Minimum size to compress: 1KB

    Complexity: O(n) where n = entry size
    """

    COMPRESSION_THRESHOLD = 1024  # 1KB
    COMPRESSION_LEVEL = 3  # Balance speed and ratio

    @staticmethod
    def compress(data: str) -> Tuple[bytes, bool]:
        """
        Compress data if beneficial.

        Args:
            data: String to compress

        Returns:
            (compressed_data, was_compressed)
        """
        data_bytes = data.encode("utf-8")

        # Only compress if over threshold
        if len(data_bytes) < CompressionManager.COMPRESSION_THRESHOLD:
            return data_bytes, False

        # If zstandard not available, return uncompressed
        if not HAS_ZSTD:
            return data_bytes, False

        try:
            cctx = zstd.ZstdCompressor(level=CompressionManager.COMPRESSION_LEVEL)
            compressed = cctx.compress(data_bytes)

            # Only use compression if it saves space
            if len(compressed) < len(data_bytes) * 0.85:  # 15% savings threshold
                return compressed, True

            return data_bytes, False

        except Exception as e:
            logger.warning(f"Compression failed, storing uncompressed: {e}")
            return data_bytes, False

    @staticmethod
    def decompress(data: bytes, was_compressed: bool) -> str:
        """
        Decompress data if it was compressed.

        Args:
            data: Bytes to decompress
            was_compressed: Whether data was compressed

        Returns:
            Decompressed string
        """
        if not was_compressed:
            return data.decode("utf-8")

        try:
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(data)
            return decompressed.decode("utf-8")

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data.decode("utf-8", errors="ignore")


# ============================================================================
# Batch Operations
# ============================================================================


@dataclass
class BatchMemoryOperation:
    """Single operation in a batch."""

    patient_id: str
    memory_type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher = process first


class BatchMemoryProcessor:
    """
    Process memory operations in batches.

    Improves throughput for bulk operations:
    - Reduces connection overhead
    - Enables transaction batching
    - Improves database insert performance

    Complexity: O(n) where n = batch size
    """

    def __init__(self, batch_size: int = 100, max_batch_time_ms: int = 5000):
        self.batch_size = batch_size
        self.max_batch_time_ms = max_batch_time_ms
        self.pending_operations: List[BatchMemoryOperation] = []
        self.last_flush_time = datetime.utcnow()

    def add_operation(self, operation: BatchMemoryOperation):
        """Add operation to batch."""
        self.pending_operations.append(operation)

        # Flush if batch is full
        if len(self.pending_operations) >= self.batch_size:
            asyncio.create_task(self.flush())

    async def flush(self) -> int:
        """
        Flush pending operations.

        Returns:
            Number of operations flushed
        """
        if not self.pending_operations:
            return 0

        operations = self.pending_operations[:]
        self.pending_operations = []
        self.last_flush_time = datetime.utcnow()

        try:
            # Sort by priority (higher first)
            operations.sort(key=lambda op: op.priority, reverse=True)

            # Group by patient for better cache locality
            by_patient: Dict[str, List[BatchMemoryOperation]] = {}
            for op in operations:
                if op.patient_id not in by_patient:
                    by_patient[op.patient_id] = []
                by_patient[op.patient_id].append(op)

            # Process each patient's operations together
            tasks = []
            for patient_id, patient_ops in by_patient.items():
                for op in patient_ops:
                    # Note: Memory manager would handle actual storage
                    tasks.append(self._store_single(op))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful operations
            successful = sum(1 for r in results if r is True)
            logger.info(
                f"Batch flush: {successful}/{len(operations)} operations "
                f"({successful/len(operations)*100:.1f}%)"
            )

            return successful

        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            return 0

    async def _store_single(self, operation: BatchMemoryOperation) -> bool:
        """
        Store single operation.

        In real implementation, this would call memory_manager.store_memory()
        """
        # Placeholder - would call memory_manager
        await asyncio.sleep(0.001)
        return True

    async def periodic_flush(self):
        """Flush periodically even if batch not full."""
        while True:
            await asyncio.sleep(self.max_batch_time_ms / 1000.0)

            time_since_flush = (
                datetime.utcnow() - self.last_flush_time
            ).total_seconds() * 1000
            if time_since_flush > self.max_batch_time_ms and self.pending_operations:
                await self.flush()


# ============================================================================
# Multi-Tier Caching Strategy
# ============================================================================


class CacheEntry:
    """Single cache entry with TTL."""

    def __init__(self, data: Any, ttl_seconds: int = 3600):
        self.data = data
        self.created_at = datetime.utcnow()
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
        self.last_accessed = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def access(self) -> Any:
        """Access entry and update stats."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        return self.data


class MultiTierCache:
    """
    Multi-tier caching strategy:
    1. L1: In-memory LRU cache (hot data)
    2. L2: Redis cache (shared, persistent)
    3. L3: Database (authoritative)

    Falls back gracefully if Redis unavailable.

    Complexity:
    - L1 hit: O(1)
    - L2 hit: O(1)
    - L3 hit: O(log n)
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        redis_enabled: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_max_size = l1_max_size
        self.redis_enabled = redis_enabled
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None

        # Stats
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0

        # Try to initialize Redis if enabled
        if redis_enabled:
            self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection (optional)."""
        try:
            try:
                import redis
            except ImportError:
                logger.warning("redis package not installed, Redis cache disabled")
                self.redis_enabled = False
                return

            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized")

        except Exception as e:
            logger.warning(f"Redis initialization failed, falling back to L1 only: {e}")
            self.redis_enabled = False
            self.redis_client = None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (L1 -> L2 -> None).

        Returns value and updates stats.
        """
        # Try L1 (in-memory)
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if not entry.is_expired():
                self.l1_hits += 1
                return entry.access()
            else:
                del self.l1_cache[key]

        self.l1_misses += 1

        # Try L2 (Redis) if enabled
        if self.redis_enabled and self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.l2_hits += 1
                    # Promote to L1
                    self.set(key, json.loads(value), ttl_seconds=3600)
                    return json.loads(value)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")
                self.l2_misses += 1

        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Store value in cache (L1 + L2)."""
        # Always store in L1
        if len(self.l1_cache) >= self.l1_max_size:
            # Evict least recently used
            lru_key = min(
                self.l1_cache.keys(),
                key=lambda k: self.l1_cache[k].last_accessed,
            )
            del self.l1_cache[lru_key]

        self.l1_cache[key] = CacheEntry(value, ttl_seconds)

        # Try L2 (Redis) if enabled
        if self.redis_enabled and self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    ttl_seconds,
                    json.dumps(value, default=str),
                )
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")

    async def delete(self, key: str):
        """Delete value from all tiers."""
        if key in self.l1_cache:
            del self.l1_cache[key]

        if self.redis_enabled and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.debug(f"Redis delete failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        l1_total = self.l1_hits + self.l1_misses
        l1_hit_rate = (self.l1_hits / l1_total * 100) if l1_total > 0 else 0

        l2_total = self.l2_hits + self.l2_misses
        l2_hit_rate = (self.l2_hits / l2_total * 100) if l2_total > 0 else 0

        return {
            "l1": {
                "size": len(self.l1_cache),
                "max_size": self.l1_max_size,
                "hits": self.l1_hits,
                "misses": self.l1_misses,
                "hit_rate_percent": round(l1_hit_rate, 2),
            },
            "l2": {
                "enabled": self.redis_enabled,
                "hits": self.l2_hits,
                "misses": self.l2_misses,
                "hit_rate_percent": round(l2_hit_rate, 2),
            },
            "combined_hit_rate_percent": round(
                (
                    ((self.l1_hits + self.l2_hits) / (l1_total + l2_total) * 100)
                    if (l1_total + l2_total) > 0
                    else 0
                ),
                2,
            ),
        }


# ============================================================================
# Paginated Search Results
# ============================================================================


@dataclass
class PaginatedResults:
    """Paginated search results."""

    items: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": self.items,
            "pagination": {
                "page": self.page,
                "page_size": self.page_size,
                "total_count": self.total_count,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_previous": self.has_previous,
            },
        }


class PaginationHelper:
    """Helper for paginated queries."""

    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE = 500

    @staticmethod
    def validate_pagination(
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Tuple[int, int]:
        """
        Validate pagination parameters.

        Ensures page >= 1 and page_size is reasonable.
        """
        page = max(1, page)
        page_size = min(max(1, page_size), PaginationHelper.MAX_PAGE_SIZE)
        return page, page_size

    @staticmethod
    def paginate_results(
        all_results: List[Dict[str, Any]],
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> PaginatedResults:
        """
        Paginate results.

        Complexity: O(n) where n = total results
        """
        page, page_size = PaginationHelper.validate_pagination(page, page_size)

        total_count = len(all_results)
        total_pages = (total_count + page_size - 1) // page_size

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        items = all_results[start_idx:end_idx]

        return PaginatedResults(
            items=items,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


# ============================================================================
# Query Optimization
# ============================================================================


class QueryOptimizer:
    """
    Optimize memory queries.

    Strategies:
    1. Result deduplication
    2. Filtering before return
    3. Lazy loading of details
    4. Index-aware sorting

    Complexity: O(n log n) for sorting, O(n) for filtering
    """

    @staticmethod
    def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on content hash.

        Useful for merged results from multiple sources.
        """
        seen = set()
        unique = []

        for result in results:
            # Use conversation_id + timestamp as unique key
            key = (
                result.get("conversation_id", ""),
                result.get("timestamp", ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(result)

        return unique

    @staticmethod
    def filter_by_date_range(
        results: List[Dict[str, Any]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter results by date range.

        Expected format: "2025-12-08T10:30:00"
        """
        if not start_date and not end_date:
            return results

        filtered = []
        for result in results:
            timestamp = result.get("timestamp", "")
            if timestamp:
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                filtered.append(result)

        return filtered

    @staticmethod
    def filter_by_memory_type(
        results: List[Dict[str, Any]],
        memory_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Filter results by memory type."""
        if not memory_types:
            return results

        return [r for r in results if r.get("memory_type") in memory_types]

    @staticmethod
    def rank_by_relevance(
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Rank results by relevance to query.

        Simple scoring:
        - Exact phrase match: +2
        - Word match: +1
        - Recent: +0.5
        """
        query_words = query.lower().split()

        def score(result: Dict[str, Any]) -> float:
            score_value = 0.0
            content = result.get("content", "").lower()

            # Phrase match
            if query.lower() in content:
                score_value += 2.0

            # Word matches
            for word in query_words:
                if word in content:
                    score_value += 1.0

            # Recency bonus
            timestamp = result.get("timestamp", "")
            if timestamp:
                # Newer = higher score
                try:
                    days_old = (
                        datetime.utcnow()
                        - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    ).days
                    recency_bonus = max(0.5 - (days_old * 0.05), 0)
                    score_value += recency_bonus
                except ValueError as e:
                    logger.debug(f"Error parsing timestamp for recency bonus: {e}")

            return score_value

        results_with_scores = [(r, score(r)) for r in results]
        results_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [r for r, _ in results_with_scores]


# ============================================================================
# Connection Pool Configuration
# ============================================================================


@dataclass
class PoolConfig:
    """Connection pool configuration recommendations."""

    environment: str  # dev, staging, prod
    expected_patients: int  # Number of active patients
    concurrent_requests: int  # Expected concurrent requests
    pool_size: int  # Base pool size
    max_overflow: int  # Additional connections on demand
    pool_timeout: int  # Timeout in seconds
    pool_recycle: int  # Recycle connection after seconds

    @staticmethod
    def for_development() -> "PoolConfig":
        """Configuration for development."""
        return PoolConfig(
            environment="dev",
            expected_patients=10,
            concurrent_requests=5,
            pool_size=2,
            max_overflow=2,
            pool_timeout=10,
            pool_recycle=3600,
        )

    @staticmethod
    def for_staging() -> "PoolConfig":
        """Configuration for staging."""
        return PoolConfig(
            environment="staging",
            expected_patients=100,
            concurrent_requests=20,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
        )

    @staticmethod
    def for_production() -> "PoolConfig":
        """Configuration for production."""
        return PoolConfig(
            environment="prod",
            expected_patients=1000,
            concurrent_requests=50,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
        )

    @staticmethod
    def for_high_traffic() -> "PoolConfig":
        """Configuration for high-traffic production."""
        return PoolConfig(
            environment="prod",
            expected_patients=10000,
            concurrent_requests=200,
            pool_size=30,
            max_overflow=50,
            pool_timeout=30,
            pool_recycle=1800,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment,
            "expected_patients": self.expected_patients,
            "concurrent_requests": self.concurrent_requests,
            "pool": {
                "size": self.pool_size,
                "max_overflow": self.max_overflow,
                "timeout_seconds": self.pool_timeout,
                "recycle_seconds": self.pool_recycle,
            },
        }


# ============================================================================
# Lazy Loading
# ============================================================================


class LazyLoadedConversation:
    """
    Lazy load conversation context.

    Only loads details when accessed, reducing memory footprint.
    """

    def __init__(
        self,
        conversation_id: str,
        summary: str,
        loader_fn,
    ):
        self.conversation_id = conversation_id
        self.summary = summary
        self._full_conversation = None
        self._loader_fn = loader_fn
        self._loaded = False

    async def get_full_conversation(self) -> Dict[str, Any]:
        """Load full conversation on demand."""
        if not self._loaded:
            self._full_conversation = await self._loader_fn(self.conversation_id)
            self._loaded = True

        return self._full_conversation or {}

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return summary without loading full conversation."""
        return {
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "is_loaded": self._loaded,
        }


# ============================================================================
# Performance Monitoring
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Track performance metrics."""

    search_p50_ms: float = 0.0
    search_p95_ms: float = 0.0
    search_p99_ms: float = 0.0
    store_p50_ms: float = 0.0
    store_p95_ms: float = 0.0
    store_p99_ms: float = 0.0
    cache_hit_rate_percent: float = 0.0
    memory_usage_mb: float = 0.0
    batch_throughput_ops_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "search_latency_ms": {
                "p50": round(self.search_p50_ms, 2),
                "p95": round(self.search_p95_ms, 2),
                "p99": round(self.search_p99_ms, 2),
            },
            "store_latency_ms": {
                "p50": round(self.store_p50_ms, 2),
                "p95": round(self.store_p95_ms, 2),
                "p99": round(self.store_p99_ms, 2),
            },
            "cache_hit_rate_percent": round(self.cache_hit_rate_percent, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "batch_throughput_ops_per_sec": round(self.batch_throughput_ops_per_sec, 2),
        }


# ============================================================================
# Summary
# ============================================================================

"""
INTEGRATION WITH main.py:

1. Add imports:
   ```python
   from memory_performance import (
       MultiTierCache,
       BatchMemoryProcessor,
       CompressionManager,
       QueryOptimizer,
       PoolConfig,
   )
   ```

2. Initialize components:
   ```python
   # In app startup
   cache = MultiTierCache(l1_max_size=1000, redis_enabled=True)
   batch_processor = BatchMemoryProcessor(batch_size=100)

   # Configure pool
   pool_config = PoolConfig.for_production()
   ```

3. Use compression for large entries:
   ```python
   from memory_middleware import with_memory_context

   @app.post("/nlp/process")
   async def nlp_process(request: NLPRequest):
       # Large conversation context
       compressed, was_compressed = CompressionManager.compress(context)
       ```

4. Batch memory operations:
   ```python
   # For bulk updates
   batch = BatchMemoryProcessor()
   for item in bulk_data:
       batch.add_operation(BatchMemoryOperation(
           patient_id=item["patient_id"],
           memory_type="conversation",
           content=item["text"]
       ))
   await batch.flush()
   ```

5. Use paginated searches:
   ```python
   @app.get("/memory/search")
   async def search_memory(
       patient_id: str,
       query: str,
       page: int = 1,
       page_size: int = 50
   ):
       results = await memory_mgr.search_memory(patient_id, query)

       # Optimize results
       results = QueryOptimizer.deduplicate_results(results)
       results = QueryOptimizer.rank_by_relevance(results, query)

       paginated = PaginationHelper.paginate_results(
           results, page, page_size
       )
       return paginated.to_dict()
   ```

6. Monitor performance:
   ```python
   # In health check endpoint
   cache_stats = cache.get_stats()
   return {
       "cache": cache_stats,
       "recommendations": pool_config.to_dict()
   }
   ```

PERFORMANCE TARGETS:
- Search latency: p50 < 50ms, p95 < 100ms, p99 < 200ms
- Store latency: p50 < 100ms, p95 < 300ms, p99 < 500ms
- Cache hit rate: > 80% after warmup
- Memory footprint: < 500MB for 1000 active patients
- Batch throughput: > 1000 ops/sec for 100-entry batches
"""
