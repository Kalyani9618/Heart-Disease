"""
Redis-backed rate limiter for distributed environments.

Uses Redis sorted sets for efficient sliding window rate limiting
that works across multiple workers and containers.
"""


import time
import logging
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    logger.warning("redis not installed. Run: pip install redis")


class RedisRateLimiter:
    """
    Distributed rate limiter using Redis sorted sets.
    
    Algorithm: Sliding window log using ZSET
    - Each request timestamp is stored as score
    - Window cleanup happens on each check
    - Atomic operations ensure consistency
    
    Usage:
        limiter = RedisRateLimiter()
        await limiter.connect()
        allowed, reason = await limiter.check_rate_limit("user123", "/api/chat")
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        requests_per_minute: int = 100,
        requests_per_hour: int = 5000,
        key_prefix: str = "heartguard:ratelimit"
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.key_prefix = key_prefix
        self._redis: Optional[object] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Establish Redis connection."""
        if not REDIS_AVAILABLE:
            logger.error("Redis package not available")
            return False
        
        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str = "global"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request should be allowed.
        
        Args:
            user_id: Unique user identifier
            endpoint: API endpoint for granular limits
            
        Returns:
            (is_allowed, reason_if_blocked)
        """
        if not self._connected or not self._redis:
            # Fallback: allow if Redis unavailable (fail open)
            logger.warning("Redis unavailable, allowing request (fail-open)")
            return True, None
        
        now = time.time()
        now_ms = int(now * 1000)
        
        # Keys for minute and hour windows
        minute_key = f"{self.key_prefix}:min:{user_id}"
        hour_key = f"{self.key_prefix}:hour:{user_id}"
        
        try:
            pipe = self._redis.pipeline()
            
            # === Minute Window ===
            minute_ago_ms = int((now - 60) * 1000)
            
            # Remove old entries
            pipe.zremrangebyscore(minute_key, 0, minute_ago_ms)
            # Count current entries
            pipe.zcard(minute_key)
            # Add new entry
            pipe.zadd(minute_key, {f"{now_ms}": now_ms})
            # Set expiry (cleanup)
            pipe.expire(minute_key, 120)
            
            # === Hour Window ===
            hour_ago_ms = int((now - 3600) * 1000)
            
            pipe.zremrangebyscore(hour_key, 0, hour_ago_ms)
            pipe.zcard(hour_key)
            pipe.zadd(hour_key, {f"{now_ms}": now_ms})
            pipe.expire(hour_key, 7200)
            
            results = await pipe.execute()
            
            minute_count = results[1]  # zcard result for minute
            hour_count = results[5]    # zcard result for hour
            
            # Check limits
            if minute_count >= self.requests_per_minute:
                return (
                    False,
                    f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
                )
            
            if hour_count >= self.requests_per_hour:
                return (
                    False,
                    f"Rate limit exceeded: {self.requests_per_hour} requests per hour"
                )
            
            return True, None
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis has issues
            return True, None
    
    async def get_stats(self, user_id: str) -> dict:
        """Get current rate limit stats for user."""
        if not self._connected or not self._redis:
            return {"error": "Redis unavailable"}
        
        now = time.time()
        minute_key = f"{self.key_prefix}:min:{user_id}"
        hour_key = f"{self.key_prefix}:hour:{user_id}"
        
        try:
            pipe = self._redis.pipeline()
            pipe.zcard(minute_key)
            pipe.zcard(hour_key)
            results = await pipe.execute()
            
            minute_count, hour_count = results
            
            return {
                "requests_this_minute": minute_count,
                "limit_per_minute": self.requests_per_minute,
                "remaining_this_minute": max(0, self.requests_per_minute - minute_count),
                "requests_this_hour": hour_count,
                "limit_per_hour": self.requests_per_hour,
                "remaining_this_hour": max(0, self.requests_per_hour - hour_count),
            }
        except Exception as e:
            logger.error(f"Failed to get rate limit stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("Redis rate limiter connection closed")


# === Integration with existing security.py ===
# Update check_rate_limit_dependency to use Redis

_redis_limiter: Optional[RedisRateLimiter] = None

async def get_redis_rate_limiter() -> RedisRateLimiter:
    """Get singleton Redis rate limiter."""
    global _redis_limiter
    if _redis_limiter is None:
        _redis_limiter = RedisRateLimiter()
        await _redis_limiter.connect()
    return _redis_limiter