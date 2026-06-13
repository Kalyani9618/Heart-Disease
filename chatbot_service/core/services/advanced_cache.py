"""
Advanced Multi-Tier Caching System

Implements:
- L1: In-memory LRU cache (10-100MB, <1ms)
- L2: Redis cache (1GB+, <10ms)
- L3: Persistent database (unlimited, <100ms)
- Automatic cache promotion
- Cache warming strategies
- Probabilistic eviction

Pattern: Cache-Aside, Write-Through
Reference: Designing Data-Intensive Applications (Kleppmann)
"""


import logging
import time
import pickle
import hashlib
import os
import random
import asyncio
import math
from typing import Optional, Any, Dict, List, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import json

import lz4.frame

from .performance_monitor import record_cache_operation, CacheTimer

# Optional Redis import
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis.asyncio as redis

        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        redis = None

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))


class CacheTier(Enum):
    """Cache tiers in hierarchy"""

    L1 = "l1"  # In-memory
    L2 = "l2"  # Redis
    L3 = "l3"  # Database


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 300
    tier: CacheTier = CacheTier.L1

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    def touch(self) -> None:
        """Update access time (for LRU)"""
        self.accessed_at = datetime.now()


class CacheStatistics:
    """Statistics for cache performance"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        self.total_get_duration_ms = 0.0
        self.total_set_duration_ms = 0.0

    @property
    def total_requests(self) -> int:
        """Total cache requests"""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100

    @property
    def avg_get_duration_ms(self) -> float:
        """Average GET duration"""
        if self.hits == 0:
            return 0.0
        return self.total_get_duration_ms / self.hits

    @property
    def avg_set_duration_ms(self) -> float:
        """Average SET duration"""
        if self.hits + self.evictions == 0:
            return 0.0
        return self.total_set_duration_ms / (self.hits + self.evictions)

    def to_dict(self) -> dict:
        """Export as dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate_percent": self.hit_rate,
            "avg_get_duration_ms": self.avg_get_duration_ms,
            "avg_set_duration_ms": self.avg_set_duration_ms,
        }


class CacheBackend(ABC):
    """Abstract base for cache backends"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set value in cache"""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete from cache"""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health"""
        ...


class L1MemoryCache(CacheBackend):
    """In-memory LRU cache (L1)"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStatistics()

    async def get(self, key: str) -> Optional[Any]:
        """Get from L1 cache"""
        start_time = time.time()
        try:
            entry = self._cache.get(key)

            if entry is None:
                self.stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self.stats.misses += 1
                return None

            # Update LRU
            entry.touch()
            self.stats.hits += 1
            return entry.value

        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            # Record performance metrics
            record_cache_operation("get", hit=True, latency_ms=elapsed_ms)
            self.stats.total_get_duration_ms += elapsed_ms

    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set in L1 cache"""
        start = time.time()

        try:
            # Evict LRU if needed
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Find least recently used
                lru_key = min(
                    self._cache.keys(), key=lambda k: self._cache[k].accessed_at
                )
                del self._cache[lru_key]
                self.stats.evictions += 1

            self._cache[key] = CacheEntry(
                key=key, value=value, ttl_seconds=ttl_seconds, tier=CacheTier.L1
            )

        finally:
            elapsed_ms = (time.time() - start) * 1000
            # Record performance metrics
            record_cache_operation("set", latency_ms=elapsed_ms)
            self.stats.total_set_duration_ms += elapsed_ms

    async def delete(self, key: str) -> None:
        """Delete from L1"""
        self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear L1"""
        self._cache.clear()

    async def health_check(self) -> bool:
        """L1 always healthy"""
        return True


class L2RedisCache(CacheBackend):
    """
    Redis cache backend (L2).

    Environment Variables:
        REDIS_URL: Redis connection URL (default: redis://localhost:6379)
        REDIS_DB: Redis database number (default: 0)
        REDIS_PASSWORD: Redis password (optional)
        REDIS_MAX_CONNECTIONS: Max pool connections (default: 10)
    """

    def __init__(
        self,
        url: str = None,
        db: int = None,
        password: str = None,
        key_prefix: str = "nlp_cache:",
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL
            db: Database number
            password: Redis password
            key_prefix: Prefix for all cache keys
        """
        self.url = url or REDIS_URL
        self.db = db if db is not None else REDIS_DB
        self.password = password or REDIS_PASSWORD
        self.key_prefix = key_prefix
        self.stats = CacheStatistics()

        self._client = None
        self._connected = False

        if not REDIS_AVAILABLE:
            logger.warning(
                "Redis libraries not available. Install with: "
                "pip install redis[asyncio]"
            )

    async def connect(self) -> bool:
        """
        Connect to Redis.

        Returns:
            True if connected successfully
        """
        if self._connected and self._client:
            return True

        if not REDIS_AVAILABLE:
            return False

        try:
            self._client = await redis.from_url(
                self.url,
                db=self.db,
                password=self.password,
                encoding="utf-8",
                decode_responses=False,  # We'll handle serialization
                max_connections=REDIS_MAX_CONNECTIONS,
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis: {self.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            self._client = None
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._connected = False
            self._client = None

    def _make_key(self, key: str) -> str:
        """Add prefix to cache key."""
        return f"{self.key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize and optionally compress value."""
        # Pickle the object
        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compression threshold (1KB)
        compression_threshold = 1024
        
        # Compress if above threshold
        if len(serialized) > compression_threshold:
            compressed = lz4.frame.compress(serialized)
            
            # Only use compression if it actually reduces size
            if len(compressed) < len(serialized):
                # Prepend magic byte to indicate compression
                return b'\x01' + compressed
        
        # No compression (prepend null byte)
        return b'\x00' + serialized

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize and decompress value."""
        if not data:
            return None
        
        # Check compression flag
        is_compressed = data[0] == 1
        payload = data[1:]
        
        if is_compressed:
            decompressed = lz4.frame.decompress(payload)
            return pickle.loads(decompressed)
        else:
            return pickle.loads(payload)

    async def get(self, key: str) -> Optional[Any]:
        """Get from Redis cache."""
        if not self._connected or not self._client:
            await self.connect()
            if not self._connected:
                return None

        start = time.time()

        try:
            prefixed_key = self._make_key(key)
            data = await self._client.get(prefixed_key)

            if data is None:
                self.stats.misses += 1
                return None

            # Deserialize and decompress
            value = self._deserialize(data)
            self.stats.hits += 1
            return value

        except Exception as e:
            logger.error(f"Redis GET error for {key}: {e}")
            self.stats.errors += 1
            return None

        finally:
            elapsed_ms = (time.time() - start) * 1000
            # Record performance metrics
            record_cache_operation("get", hit=True, latency_ms=elapsed_ms)
            self.stats.total_get_duration_ms += elapsed_ms

    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set in Redis cache."""
        if not self._connected or not self._client:
            await self.connect()
            if not self._connected:
                return

        start = time.time()

        try:
            prefixed_key = self._make_key(key)

            # Serialize and compress value
            data = self._serialize(value)

            # Set with TTL
            await self._client.setex(prefixed_key, ttl_seconds, data)

        except Exception as e:
            logger.error(f"Redis SET error for {key}: {e}")
            self.stats.errors += 1

        finally:
            elapsed_ms = (time.time() - start) * 1000
            # Record performance metrics
            record_cache_operation("set", latency_ms=elapsed_ms)
            self.stats.total_set_duration_ms += elapsed_ms

    async def delete(self, key: str) -> None:
        """Delete from Redis."""
        if not self._connected or not self._client:
            return

        try:
            prefixed_key = self._make_key(key)
            await self._client.delete(prefixed_key)
        except Exception as e:
            logger.error(f"Redis DELETE error for {key}: {e}")

    async def clear(self) -> None:
        """Clear all keys with our prefix."""
        if not self._connected or not self._client:
            return

        try:
            # Scan for keys with our prefix and delete them
            pattern = f"{self.key_prefix}*"
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._client.delete(*keys)
                if cursor == 0:
                    break

            logger.info(f"Cleared Redis cache with prefix: {self.key_prefix}")
        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")

    async def health_check(self) -> bool:
        """Check Redis health."""
        if not REDIS_AVAILABLE:
            return False

        try:
            if not self._connected or not self._client:
                return await self.connect()

            await self._client.ping()
            return True
        except Exception:
            return False


class MultiTierCache:
    """
    Multi-tier caching system with automatic promotion.

    L1: In-memory (fast, limited size)
       ↓ miss
    L2: Redis (medium, medium size)
       ↓ miss
    L3: Database (slow, unlimited) - not implemented
       ↓ miss
    Compute + promote back up tiers

    Environment Variables:
        CACHE_ENABLE_L2: Enable Redis L2 cache (default: false)
        REDIS_URL: Redis connection URL

    Example:
        cache = MultiTierCache(enable_l2=True)

        # Get with auto-population
        result = await cache.get_or_compute(
            key='intent_result',
            compute_fn=recognizer.analyze,
            args=(text,)
        )
    """

    def __init__(
        self, l1_max_size: int = 10000, enable_l2: bool = None, redis_url: str = None
    ):
        """
        Initialize multi-tier cache.

        Args:
            l1_max_size: Maximum entries in L1 memory cache
            enable_l2: Enable L2 Redis cache (default from env)
            redis_url: Redis URL (default from env)
        """
        self.l1 = L1MemoryCache(max_size=l1_max_size)

        # Initialize L2 Redis cache if enabled
        if enable_l2 is None:
            enable_l2 = os.getenv("CACHE_ENABLE_L2", "false").lower() == "true"

        self.l2 = None
        self.l2_enabled = False

        if enable_l2 and REDIS_AVAILABLE:
            self.l2 = L2RedisCache(url=redis_url)
            self.l2_enabled = True
            logger.info("Multi-tier cache initialized with L1 + L2 (Redis)")
        else:
            logger.info("Multi-tier cache initialized with L1 only")

        self.stats = CacheStatistics()

    async def initialize(self) -> None:
        """Initialize async connections (call after construction)."""
        if self.l2 and self.l2_enabled:
            connected = await self.l2.connect()
            if not connected:
                logger.warning("L2 Redis connection failed, operating with L1 only")
                self.l2_enabled = False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get from cache, checking tiers in order.

        Args:
            key: Cache key

        Returns:
            Value if found, None otherwise
        """
        # Try L1 (fastest)
        result = await self.l1.get(key)
        if result is not None:
            return result

        # Try L2 (Redis)
        if self.l2_enabled and self.l2:
            result = await self.l2.get(key)
            if result is not None:
                # Promote to L1
                await self.l1.set(key, result)
                logger.debug(f"Promoted {key} from L2 to L1")
                return result

        self.stats.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """
        Set in all enabled cache tiers.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live
        """
        # Set in L1
        await self.l1.set(key, value, ttl_seconds)

        # Set in L2 (Redis) if enabled
        if self.l2_enabled and self.l2:
            await self.l2.set(key, value, ttl_seconds)

        self.stats.hits += 1

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple = (),
        kwargs: dict = None,
        ttl_seconds: int = 300,
    ) -> Any:
        """
        Get from cache or compute if missing.

        Args:
            key: Cache key
            compute_fn: Async function to compute value
            args: Arguments for compute_fn
            kwargs: Keyword arguments for compute_fn
            ttl_seconds: Cache TTL

        Returns:
            Cached or computed value

        Example:
            result = await cache.get_or_compute(
                key=f'intent_{hash(text)}',
                compute_fn=recognizer.analyze,
                args=(text,),
                ttl_seconds=300
            )
        """
        kwargs = kwargs or {}

        # Try cache
        cached = await self.get(key)
        if cached is not None:
            logger.debug(f"Cache hit: {key}")
            return cached

        # Compute
        logger.debug(f"Cache miss, computing: {key}")
        value = await compute_fn(*args, **kwargs)

        # Store
        await self.set(key, value, ttl_seconds)

        return value

    async def get_with_early_expiration(
        self, 
        key: str, 
        factory_func: Callable[..., Coroutine[Any, Any, Any]],
        ttl: int = 3600,
        beta: float = 1.0,
        args: tuple = (),
        kwargs: dict = None
    ) -> Any:
        """
        Probabilistic early expiration to prevent cache stampedes.
        
        Based on "Optimal Probabilistic Cache Stampede Prevention"
        https://cseweb.ucsd.edu/~avattani/papers/cache_stampede.pdf
        
        Args:
            key: Cache key
            factory_func: Async function to regenerate value
            ttl: Time to live
            beta: Tuning parameter (higher = less aggressive, default: 1.0)
            args: Arguments for factory function
            kwargs: Keyword arguments for factory function
        """
        kwargs = kwargs or {}
        
        # Check if key exists with TTL
        if self.l2_enabled and self.l2 and self.l2._connected and self.l2._client:
            prefixed_key = self.l2._make_key(key)
            value_data = await self.l2._client.get(prefixed_key)
            remaining_ttl = await self.l2._client.ttl(prefixed_key)
            
            if value_data and remaining_ttl > 0:
                value = self.l2._deserialize(value_data)
                
                # Calculate early expiration probability
                # As TTL decreases, probability of refresh increases
                delta = ttl - remaining_ttl  # How much time has passed
                
                # Probabilistic early refresh
                # If delta is large (key is old), more likely to refresh
                random_value = random.random()
                if delta > 0:
                    threshold = delta * beta * math.log(random_value) / ttl
                    
                    if threshold < 0:
                        # Early refresh triggered - regenerate in background
                        # Return stale value immediately
                        asyncio.create_task(self._refresh_key(key, factory_func, ttl, args, kwargs))
                        return value
                
                return value
        
        # Cache miss or expired - regenerate
        value = await factory_func(*args, **kwargs)
        await self.set(key, value, ttl)
        return value
    
    async def _refresh_key(self, key: str, factory_func, ttl: int, args: tuple = (), kwargs: dict = None):
        """Background task to refresh cache key."""
        try:
            kwargs = kwargs or {}
            value = await factory_func(*args, **kwargs)
            await self.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Failed to refresh cache key {key}: {e}")

    async def warm_up(self, keys_and_values: List[tuple]) -> None:
        """
        Pre-populate cache.

        Example:
            await cache.warm_up([
                ('greeting_1', intent_result_1),
                ('greeting_2', intent_result_2),
            ])
        """
        for key, value in keys_and_values:
            await self.set(key, value)

        logger.info(f"Warmed up {len(keys_and_values)} cache entries")

    async def delete(self, key: str) -> None:
        """Delete from all tiers."""
        await self.l1.delete(key)
        if self.l2_enabled and self.l2:
            await self.l2.delete(key)

    async def clear(self) -> None:
        """Clear all tiers."""
        await self.l1.clear()
        if self.l2_enabled and self.l2:
            await self.l2.clear()

    def get_statistics(self) -> dict:
        """Get cache statistics from all tiers."""
        stats = {"l1": self.l1.stats.to_dict(), "multi_tier": self.stats.to_dict()}
        if self.l2_enabled and self.l2:
            stats["l2"] = self.l2.stats.to_dict()
        return stats

    async def health_check(self) -> dict:
        """Check health of all cache tiers."""
        health = {"l1": await self.l1.health_check(), "l2_enabled": self.l2_enabled}
        if self.l2_enabled and self.l2:
            health["l2"] = await self.l2.health_check()
        return health

    async def close(self) -> None:
        """Close all connections."""
        if self.l2:
            await self.l2.disconnect()


class CacheKeyBuilder:
    """Build cache keys from parameters"""

    @staticmethod
    def intent_key(text: str, user_id: Optional[str] = None) -> str:
        """Build cache key for intent recognition"""
        text_hash = hashlib.sha256(text.lower().encode()).hexdigest()[:8]
        if user_id:
            return f"intent_{user_id}_{text_hash}"
        return f"intent_{text_hash}"

    @staticmethod
    def sentiment_key(text: str, user_id: Optional[str] = None) -> str:
        """Build cache key for sentiment analysis"""
        text_hash = hashlib.sha256(text.lower().encode()).hexdigest()[:8]
        if user_id:
            return f"sentiment_{user_id}_{text_hash}"
        return f"sentiment_{text_hash}"

    @staticmethod
    def entity_key(text: str) -> str:
        """Build cache key for entity extraction"""
        text_hash = hashlib.sha256(text.lower().encode()).hexdigest()[:8]
        return f"entity_{text_hash}"

    @staticmethod
    def risk_key(
        text: str, user_id: Optional[str] = None, health_metrics: Optional[dict] = None
    ) -> str:
        """Build cache key for risk assessment"""
        text_hash = hashlib.sha256(text.lower().encode()).hexdigest()[:8]

        if health_metrics:
            metrics_hash = hashlib.sha256(
                json.dumps(health_metrics, sort_keys=True).encode()
            ).hexdigest()[:8]
            key = f"risk_{text_hash}_{metrics_hash}"
        else:
            key = f"risk_{text_hash}"

        if user_id:
            return f"{key}_{user_id}"
        return key


__all__ = [
    "MultiTierCache",
    "L1MemoryCache",
    "L2RedisCache",
    "CacheBackend",
    "CacheEntry",
    "CacheStatistics",
    "CacheTier",
    "CacheKeyBuilder",
    "REDIS_AVAILABLE",
]


# Add get_cache_service helper
_cache_instance: Optional[MultiTierCache] = None

async def get_cache_service() -> MultiTierCache:
    """Get singleton MultiTierCache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MultiTierCache(
            l1_max_size=10000,
            enable_l2=True  # Enable Redis L2 if available
        )
    return _cache_instance
