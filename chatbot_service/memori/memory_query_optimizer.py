"""
Memory Query Optimizer - Production-grade optimization for memory operations.

Implements:
- Tiered caching (L1 in-memory + L2 Redis) for memory retrieval
- Batch operations for memory writes
- Optimized relevance scoring queries
- Query result pagination and streaming
- Memory aggregation caching
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory query optimization."""
    
    # Cache settings
    l1_cache_size: int = 500  # Max entries in L1 cache
    l1_cache_ttl: int = 300  # L1 cache TTL in seconds
    l2_cache_ttl: int = 600  # L2 (Redis) cache TTL in seconds
    
    # Batch settings
    batch_size: int = 100  # Max items per batch write
    batch_timeout: float = 1.0  # Max wait time before flushing batch
    
    # Query settings
    default_page_size: int = 50
    max_page_size: int = 200
    query_timeout: float = 30.0  # Query timeout in seconds
    
    # Relevance scoring
    recency_weight: float = 0.3
    frequency_weight: float = 0.2
    importance_weight: float = 0.5
    
    # Memory aggregation
    aggregation_cache_ttl: int = 900  # 15 minutes
    max_memories_per_user: int = 10000


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 500, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # Expired, remove
                    del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: T) -> None:
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                # Remove oldest (first) item
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        count = 0
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
                count += 1
        return count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
                "ttl": self.ttl
            }


class MemoryTieredCache:
    """Two-tier caching: L1 (in-memory) + L2 (Redis)."""
    
    def __init__(
        self,
        config: MemoryOptimizationConfig,
        redis_client: Optional[Any] = None
    ):
        self.config = config
        self.l1 = LRUCache[Any](
            max_size=config.l1_cache_size,
            ttl=config.l1_cache_ttl
        )
        self.redis = redis_client
        self._cache_prefix = "memory:"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from L1, then L2 if miss."""
        # L1 lookup
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # L2 lookup (Redis)
        if self.redis:
            try:
                redis_key = f"{self._cache_prefix}{key}"
                data = await self.redis.get(redis_key)
                if data:
                    value = json.loads(data)
                    # Populate L1
                    self.l1.set(key, value)
                    return value
            except Exception as e:
                logger.warning(f"Redis L2 cache get failed: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set in both L1 and L2."""
        # L1
        self.l1.set(key, value)
        
        # L2 (Redis)
        if self.redis:
            try:
                redis_key = f"{self._cache_prefix}{key}"
                ttl = ttl or self.config.l2_cache_ttl
                await self.redis.setex(
                    redis_key,
                    ttl,
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis L2 cache set failed: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete from both L1 and L2."""
        self.l1.delete(key)
        
        if self.redis:
            try:
                redis_key = f"{self._cache_prefix}{key}"
                await self.redis.delete(redis_key)
            except Exception as e:
                logger.warning(f"Redis L2 cache delete failed: {e}")
    
    async def invalidate_user(self, user_id: str) -> int:
        """Invalidate all cache entries for a user."""
        pattern = f"user:{user_id}"
        count = self.l1.invalidate_pattern(pattern)
        
        if self.redis:
            try:
                # Scan and delete matching keys
                cursor = b"0"
                while cursor:
                    cursor, keys = await self.redis.scan(
                        cursor,
                        match=f"{self._cache_prefix}*{pattern}*",
                        count=100
                    )
                    if keys:
                        await self.redis.delete(*keys)
                        count += len(keys)
                    if cursor == b"0":
                        break
            except Exception as e:
                logger.warning(f"Redis pattern invalidation failed: {e}")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            "l1": self.l1.get_stats(),
            "l2_available": self.redis is not None
        }


@dataclass
class MemoryBatchItem:
    """Item in a memory batch operation."""
    user_id: str
    memory_type: str
    content: Dict[str, Any]
    importance: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BatchMemoryWriter:
    """Batches memory writes for efficiency."""
    
    def __init__(
        self,
        config: MemoryOptimizationConfig,
        db_pool: Any,
        flush_callback: Optional[Callable] = None
    ):
        self.config = config
        self.db_pool = db_pool
        self.flush_callback = flush_callback
        
        self._batch: List[MemoryBatchItem] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._last_flush = time.time()
        
        # Statistics
        self._total_items = 0
        self._total_batches = 0
        self._total_flush_time = 0.0
    
    async def add(self, item: MemoryBatchItem) -> None:
        """Add item to batch, flush if needed."""
        async with self._lock:
            self._batch.append(item)
            
            # Flush if batch is full
            if len(self._batch) >= self.config.batch_size:
                await self._flush_batch()
            else:
                # Schedule delayed flush
                self._schedule_flush()
    
    async def add_many(self, items: List[MemoryBatchItem]) -> None:
        """Add multiple items to batch."""
        async with self._lock:
            self._batch.extend(items)
            
            # Flush in chunks if over batch size
            while len(self._batch) >= self.config.batch_size:
                await self._flush_batch()
            
            # Schedule delayed flush for remainder
            if self._batch:
                self._schedule_flush()
    
    def _schedule_flush(self) -> None:
        """Schedule a delayed flush."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._delayed_flush())
    
    async def _delayed_flush(self) -> None:
        """Flush after timeout."""
        await asyncio.sleep(self.config.batch_timeout)
        async with self._lock:
            if self._batch:
                await self._flush_batch()
    
    async def _flush_batch(self) -> None:
        """Execute batch insert."""
        if not self._batch:
            return
        
        batch = self._batch[:self.config.batch_size]
        self._batch = self._batch[self.config.batch_size:]
        
        start_time = time.time()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Build batch insert
                await self._execute_batch_insert(conn, batch)
            
            self._total_items += len(batch)
            self._total_batches += 1
            
            # Call callback if provided
            if self.flush_callback:
                await self.flush_callback(len(batch))
            
        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            # Re-queue failed items (with limit)
            if len(self._batch) < self.config.batch_size * 3:
                self._batch = batch + self._batch
        
        finally:
            elapsed = time.time() - start_time
            self._total_flush_time += elapsed
            self._last_flush = time.time()
    
    async def _execute_batch_insert(
        self,
        conn: Any,
        batch: List[MemoryBatchItem]
    ) -> None:
        """Execute batch insert using COPY or multi-value INSERT."""
        
        # Group by user for efficient processing
        user_batches: Dict[str, List[MemoryBatchItem]] = {}
        for item in batch:
            if item.user_id not in user_batches:
                user_batches[item.user_id] = []
            user_batches[item.user_id].append(item)
        
        # Use executemany for batch insert
        insert_sql = """
            INSERT INTO patient_memories 
            (user_id, memory_type, content, importance, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $6)
            ON CONFLICT (user_id, memory_type, content_hash) 
            DO UPDATE SET 
                importance = GREATEST(patient_memories.importance, EXCLUDED.importance),
                access_count = patient_memories.access_count + 1,
                updated_at = EXCLUDED.updated_at
        """
        
        values = [
            (
                item.user_id,
                item.memory_type,
                json.dumps(item.content),
                item.importance,
                json.dumps(item.metadata) if item.metadata else None,
                item.timestamp
            )
            for item in batch
        ]
        
        await conn.executemany(insert_sql, values)
    
    async def flush(self) -> int:
        """Force flush all pending items."""
        async with self._lock:
            count = len(self._batch)
            while self._batch:
                await self._flush_batch()
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch writer statistics."""
        avg_flush_time = (
            self._total_flush_time / self._total_batches
            if self._total_batches > 0 else 0
        )
        return {
            "pending_items": len(self._batch),
            "total_items_written": self._total_items,
            "total_batches": self._total_batches,
            "avg_flush_time_ms": avg_flush_time * 1000,
            "time_since_last_flush": time.time() - self._last_flush
        }


class OptimizedMemoryQueries:
    """Optimized memory query operations."""
    
    def __init__(
        self,
        config: MemoryOptimizationConfig,
        db_pool: Any,
        redis_client: Optional[Any] = None
    ):
        self.config = config
        self.db_pool = db_pool
        self.cache = MemoryTieredCache(config, redis_client)
        self.batch_writer = BatchMemoryWriter(config, db_pool)
        
        # Query deduplication
        self._pending_queries: Dict[str, asyncio.Future] = {}
        self._query_lock = asyncio.Lock()
    
    def _make_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = ":".join(str(a) for a in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get_user_memories(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get user memories with caching and pagination."""
        
        cache_key = self._make_cache_key(
            "user_memories", user_id, memory_types, limit, offset
        )
        
        # Check cache
        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Query deduplication
        async with self._query_lock:
            if cache_key in self._pending_queries:
                return await self._pending_queries[cache_key]
            
            future: asyncio.Future = asyncio.Future()
            self._pending_queries[cache_key] = future
        
        try:
            # Build optimized query
            sql = """
                SELECT 
                    id, user_id, memory_type, content, importance,
                    metadata, access_count, created_at, updated_at
                FROM patient_memories
                WHERE user_id = $1
            """
            params = [user_id]
            param_idx = 2
            
            if memory_types:
                placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(memory_types)))
                sql += f" AND memory_type IN ({placeholders})"
                params.extend(memory_types)
                param_idx += len(memory_types)
            
            sql += f"""
                ORDER BY 
                    importance DESC,
                    updated_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([limit, offset])
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            
            result = [dict(row) for row in rows]
            
            # Cache result
            if use_cache:
                await self.cache.set(cache_key, result)
            
            # Resolve pending queries
            future.set_result(result)
            return result
            
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self._query_lock:
                self._pending_queries.pop(cache_key, None)
    
    async def get_relevant_memories(
        self,
        user_id: str,
        context: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        recency_hours: int = 168  # 1 week
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to context with optimized scoring.
        
        Uses combined relevance scoring:
        - Recency: How recent the memory is
        - Frequency: How often it's been accessed
        - Importance: Explicit importance score
        """
        
        cache_key = self._make_cache_key(
            "relevant_memories", user_id, context[:100], memory_types, limit
        )
        
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Optimized query with scoring
        sql = """
            WITH scored_memories AS (
                SELECT 
                    id, user_id, memory_type, content, importance,
                    metadata, access_count, created_at, updated_at,
                    -- Recency score (0-1)
                    CASE 
                        WHEN updated_at > NOW() - INTERVAL '1 hour' THEN 1.0
                        WHEN updated_at > NOW() - INTERVAL '1 day' THEN 0.8
                        WHEN updated_at > NOW() - INTERVAL '1 week' THEN 0.5
                        ELSE 0.2
                    END * $2 AS recency_score,
                    -- Frequency score (normalized)
                    LEAST(access_count / 100.0, 1.0) * $3 AS frequency_score,
                    -- Importance score
                    importance * $4 AS importance_score
                FROM patient_memories
                WHERE user_id = $1
                    AND updated_at > NOW() - INTERVAL '%s hours'
        """ % recency_hours
        
        params = [
            user_id,
            self.config.recency_weight,
            self.config.frequency_weight,
            self.config.importance_weight
        ]
        param_idx = 5
        
        if memory_types:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(memory_types)))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
            param_idx += len(memory_types)
        
        sql += f"""
            )
            SELECT *,
                (recency_score + frequency_score + importance_score) AS total_score
            FROM scored_memories
            ORDER BY total_score DESC
            LIMIT ${param_idx}
        """
        params.append(limit)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        result = [dict(row) for row in rows]
        
        # Update access counts asynchronously
        asyncio.create_task(self._update_access_counts([r['id'] for r in result]))
        
        # Cache with shorter TTL for relevance queries
        await self.cache.set(cache_key, result, ttl=60)
        
        return result
    
    async def _update_access_counts(self, memory_ids: List[int]) -> None:
        """Update access counts for retrieved memories."""
        if not memory_ids:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE patient_memories 
                    SET access_count = access_count + 1
                    WHERE id = ANY($1)
                """, memory_ids)
        except Exception as e:
            logger.warning(f"Failed to update access counts: {e}")
    
    async def get_memory_summary(
        self,
        user_id: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get aggregated memory summary for a user."""
        
        cache_key = self._make_cache_key("memory_summary", user_id)
        
        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        sql = """
            SELECT 
                COUNT(*) as total_memories,
                COUNT(DISTINCT memory_type) as unique_types,
                AVG(importance) as avg_importance,
                MAX(updated_at) as last_memory_at,
                json_object_agg(
                    memory_type, 
                    type_count
                ) as type_distribution
            FROM (
                SELECT 
                    memory_type,
                    COUNT(*) as type_count
                FROM patient_memories
                WHERE user_id = $1
                GROUP BY memory_type
            ) sub
        """
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(sql, user_id)
        
        result = dict(row) if row else {
            "total_memories": 0,
            "unique_types": 0,
            "avg_importance": 0,
            "last_memory_at": None,
            "type_distribution": {}
        }
        
        # Cache with longer TTL
        await self.cache.set(
            cache_key, 
            result, 
            ttl=self.config.aggregation_cache_ttl
        )
        
        return result
    
    async def store_memory(
        self,
        user_id: str,
        memory_type: str,
        content: Dict[str, Any],
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        use_batch: bool = True
    ) -> None:
        """Store a memory, optionally using batch writer."""
        
        item = MemoryBatchItem(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            metadata=metadata
        )
        
        if use_batch:
            await self.batch_writer.add(item)
        else:
            # Direct insert
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO patient_memories 
                    (user_id, memory_type, content, importance, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                    ON CONFLICT (user_id, memory_type, content_hash) 
                    DO UPDATE SET 
                        importance = GREATEST(patient_memories.importance, EXCLUDED.importance),
                        access_count = patient_memories.access_count + 1,
                        updated_at = NOW()
                """, user_id, memory_type, json.dumps(content), importance, 
                    json.dumps(metadata) if metadata else None)
        
        # Invalidate user cache
        await self.cache.invalidate_user(user_id)
    
    async def store_memories_bulk(
        self,
        memories: List[Dict[str, Any]]
    ) -> int:
        """Store multiple memories efficiently."""
        
        items = [
            MemoryBatchItem(
                user_id=m['user_id'],
                memory_type=m['memory_type'],
                content=m['content'],
                importance=m.get('importance', 1.0),
                metadata=m.get('metadata')
            )
            for m in memories
        ]
        
        await self.batch_writer.add_many(items)
        await self.batch_writer.flush()
        
        # Invalidate affected users
        user_ids = set(m['user_id'] for m in memories)
        for user_id in user_ids:
            await self.cache.invalidate_user(user_id)
        
        return len(memories)
    
    async def delete_old_memories(
        self,
        user_id: str,
        days_old: int = 365,
        keep_important: bool = True,
        importance_threshold: float = 0.8
    ) -> int:
        """Delete old memories with optional importance preservation."""
        
        sql = """
            DELETE FROM patient_memories
            WHERE user_id = $1
                AND updated_at < NOW() - INTERVAL '%s days'
        """ % days_old
        
        params = [user_id]
        
        if keep_important:
            sql += " AND importance < $2"
            params.append(importance_threshold)
        
        sql += " RETURNING id"
        
        async with self.db_pool.acquire() as conn:
            result = await conn.fetch(sql, *params)
        
        deleted_count = len(result)
        
        if deleted_count > 0:
            await self.cache.invalidate_user(user_id)
            logger.info(f"Deleted {deleted_count} old memories for user {user_id}")
        
        return deleted_count
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Full-text search in memory content."""
        
        cache_key = self._make_cache_key(
            "search_memories", user_id, query[:50], memory_types, limit
        )
        
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Use PostgreSQL full-text search if available
        sql = """
            SELECT 
                id, user_id, memory_type, content, importance,
                metadata, access_count, created_at, updated_at,
                ts_rank(
                    to_tsvector('english', content::text),
                    plainto_tsquery('english', $2)
                ) as relevance
            FROM patient_memories
            WHERE user_id = $1
                AND to_tsvector('english', content::text) @@ plainto_tsquery('english', $2)
        """
        params = [user_id, query]
        param_idx = 3
        
        if memory_types:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(memory_types)))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
            param_idx += len(memory_types)
        
        sql += f" ORDER BY relevance DESC LIMIT ${param_idx}"
        params.append(limit)
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            
            result = [dict(row) for row in rows]
        except Exception as e:
            # Fallback to ILIKE search if full-text not available
            logger.warning(f"Full-text search failed, using fallback: {e}")
            result = await self._fallback_search(user_id, query, memory_types, limit)
        
        await self.cache.set(cache_key, result, ttl=120)
        return result
    
    async def _fallback_search(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[str]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback search using ILIKE."""
        
        sql = """
            SELECT 
                id, user_id, memory_type, content, importance,
                metadata, access_count, created_at, updated_at
            FROM patient_memories
            WHERE user_id = $1
                AND content::text ILIKE $2
        """
        params = [user_id, f"%{query}%"]
        param_idx = 3
        
        if memory_types:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(memory_types)))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
            param_idx += len(memory_types)
        
        sql += f" ORDER BY importance DESC, updated_at DESC LIMIT ${param_idx}"
        params.append(limit)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        return [dict(row) for row in rows]
    
    async def flush_pending(self) -> int:
        """Flush all pending batch writes."""
        return await self.batch_writer.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "cache": self.cache.get_stats(),
            "batch_writer": self.batch_writer.get_stats()
        }


class MemoryPerformanceMonitor:
    """Monitor memory operation performance."""
    
    def __init__(self):
        self._query_times: List[Tuple[str, float, datetime]] = []
        self._write_times: List[Tuple[int, float, datetime]] = []
        self._lock = threading.Lock()
        self._max_history = 1000
    
    def record_query(self, operation: str, duration: float) -> None:
        """Record a query operation."""
        with self._lock:
            self._query_times.append((operation, duration, datetime.utcnow()))
            if len(self._query_times) > self._max_history:
                self._query_times = self._query_times[-self._max_history:]
    
    def record_write(self, count: int, duration: float) -> None:
        """Record a write operation."""
        with self._lock:
            self._write_times.append((count, duration, datetime.utcnow()))
            if len(self._write_times) > self._max_history:
                self._write_times = self._write_times[-self._max_history:]
    
    def get_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics for the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_queries = [
                (op, dur) for op, dur, ts in self._query_times
                if ts > cutoff
            ]
            recent_writes = [
                (cnt, dur) for cnt, dur, ts in self._write_times
                if ts > cutoff
            ]
        
        query_durations = [dur for _, dur in recent_queries]
        write_durations = [dur for _, dur in recent_writes]
        
        return {
            "period_minutes": minutes,
            "queries": {
                "count": len(recent_queries),
                "avg_duration_ms": (
                    sum(query_durations) / len(query_durations) * 1000
                    if query_durations else 0
                ),
                "max_duration_ms": max(query_durations) * 1000 if query_durations else 0,
                "by_operation": self._group_by_operation(recent_queries)
            },
            "writes": {
                "count": len(recent_writes),
                "total_items": sum(cnt for cnt, _ in recent_writes),
                "avg_duration_ms": (
                    sum(write_durations) / len(write_durations) * 1000
                    if write_durations else 0
                )
            }
        }
    
    def _group_by_operation(
        self, 
        queries: List[Tuple[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Group query metrics by operation type."""
        grouped: Dict[str, List[float]] = {}
        for op, dur in queries:
            if op not in grouped:
                grouped[op] = []
            grouped[op].append(dur)
        
        return {
            op: {
                "count": len(durs),
                "avg_ms": sum(durs) / len(durs) * 1000,
                "max_ms": max(durs) * 1000
            }
            for op, durs in grouped.items()
        }


# Factory function for creating optimized memory queries
def create_optimized_memory_queries(
    db_pool: Any,
    redis_client: Optional[Any] = None,
    config: Optional[MemoryOptimizationConfig] = None
) -> OptimizedMemoryQueries:
    """Create an OptimizedMemoryQueries instance with default config."""
    if config is None:
        config = MemoryOptimizationConfig()
    
    return OptimizedMemoryQueries(
        config=config,
        db_pool=db_pool,
        redis_client=redis_client
    )
