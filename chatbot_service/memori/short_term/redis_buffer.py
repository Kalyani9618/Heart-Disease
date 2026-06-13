"""
Redis Session Buffer for Short-term Memory

Provides fast, ephemeral storage for active session data.
Supports automatic TTL-based expiration and seamless fallback to in-memory storage.

Key Features:
- Redis-backed session storage with TTL
- In-memory fallback when Redis is unavailable
- Session-scoped message buffering
- Async and sync operation support
- Thread-safe operations
- Automatic memory cleanup
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SessionMessage:
    """A message in the session buffer."""
    
    id: str = ""
    role: str = "user"  # user, assistant, system
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionData:
    """Data for a session."""
    
    session_id: str = ""
    user_id: Optional[str] = None
    messages: List[SessionMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: int = 3600  # 1 hour default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ttl_seconds": self.ttl_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id"),
            messages=[SessionMessage.from_dict(m) for m in data.get("messages", [])],
            context=data.get("context", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            ttl_seconds=data.get("ttl_seconds", 3600),
        )


# ============================================================================
# Base Session Buffer Interface
# ============================================================================


class SessionBufferInterface(ABC):
    """Abstract interface for session buffers."""
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        pass
    
    @abstractmethod
    def set_session(self, session_data: SessionData) -> bool:
        """Set/update session data."""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        pass
    
    @abstractmethod
    def add_message(
        self,
        session_id: str,
        message: SessionMessage,
        create_if_missing: bool = True,
    ) -> bool:
        """Add a message to a session."""
        pass
    
    @abstractmethod
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[SessionMessage]:
        """Get messages from a session."""
        pass
    
    @abstractmethod
    def clear_session(self, session_id: str) -> bool:
        """Clear all messages from a session."""
        pass
    
    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass


# ============================================================================
# In-Memory Session Buffer (Fallback)
# ============================================================================


class InMemorySessionBuffer(SessionBufferInterface):
    """
    In-memory session buffer for when Redis is not available.
    
    Uses LRU eviction and automatic TTL-based cleanup.
    """
    
    def __init__(
        self,
        max_sessions: int = 1000,
        default_ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300,
    ):
        """
        Initialize in-memory buffer.
        
        Args:
            max_sessions: Maximum sessions to store
            default_ttl_seconds: Default TTL for sessions
            cleanup_interval_seconds: Interval for TTL cleanup
        """
        self.max_sessions = max_sessions
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        self._sessions: OrderedDict[str, Tuple[SessionData, float]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="SessionBufferCleanup",
            daemon=True,
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup for expired sessions."""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            
            self._stop_cleanup.wait(timeout=self.cleanup_interval_seconds)
    
    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = time.time()
        expired = []
        
        with self._lock:
            for session_id, (session_data, created_time) in self._sessions.items():
                if now - created_time > session_data.ttl_seconds:
                    expired.append(session_id)
            
            for session_id in expired:
                del self._sessions[session_id]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired sessions")
    
    def _evict_lru(self) -> None:
        """Evict least recently used sessions if at capacity."""
        with self._lock:
            while len(self._sessions) >= self.max_sessions:
                # Pop the oldest item (first in OrderedDict)
                session_id, _ = self._sessions.popitem(last=False)
                logger.debug(f"Evicted LRU session: {session_id}")
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        with self._lock:
            if session_id in self._sessions:
                session_data, created_time = self._sessions[session_id]
                # Check TTL
                if time.time() - created_time <= session_data.ttl_seconds:
                    # Move to end (most recently used)
                    self._sessions.move_to_end(session_id)
                    return session_data
                else:
                    # Expired
                    del self._sessions[session_id]
            return None
    
    def set_session(self, session_data: SessionData) -> bool:
        """Set/update session data."""
        with self._lock:
            if session_data.session_id not in self._sessions:
                self._evict_lru()
            
            session_data.updated_at = datetime.now().isoformat()
            self._sessions[session_data.session_id] = (session_data, time.time())
            self._sessions.move_to_end(session_data.session_id)
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def add_message(
        self,
        session_id: str,
        message: SessionMessage,
        create_if_missing: bool = True,
    ) -> bool:
        """Add a message to a session."""
        with self._lock:
            session_data = self.get_session(session_id)
            
            if session_data is None:
                if not create_if_missing:
                    return False
                session_data = SessionData(
                    session_id=session_id,
                    ttl_seconds=self.default_ttl_seconds,
                )
            
            session_data.messages.append(message)
            session_data.updated_at = datetime.now().isoformat()
            return self.set_session(session_data)
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[SessionMessage]:
        """Get messages from a session."""
        session_data = self.get_session(session_id)
        if session_data is None:
            return []
        
        messages = session_data.messages
        if limit is not None:
            messages = messages[-limit:]
        return messages
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all messages from a session."""
        with self._lock:
            session_data = self.get_session(session_id)
            if session_data is None:
                return False
            
            session_data.messages = []
            session_data.updated_at = datetime.now().isoformat()
            return self.set_session(session_data)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self.get_session(session_id) is not None
    
    def shutdown(self) -> None:
        """Shutdown the cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "type": "in_memory",
                "session_count": len(self._sessions),
                "max_sessions": self.max_sessions,
            }

    def get_promotable_sessions(
        self,
        min_messages: int = 3,
        min_age_seconds: float = 300,
    ) -> List[SessionData]:
        """
        Get sessions eligible for promotion to long-term memory.
        
        Sessions qualify when they have enough context (messages) and
        have been active long enough to contain meaningful data.
        
        Args:
            min_messages: Minimum number of messages for promotion eligibility
            min_age_seconds: Minimum age of session in seconds
            
        Returns:
            List of SessionData objects eligible for promotion
        """
        promotable = []
        now = time.time()
        
        with self._lock:
            for session_id, (session_data, created_time) in self._sessions.items():
                age = now - created_time
                msg_count = len(session_data.messages)
                
                if msg_count >= min_messages and age >= min_age_seconds:
                    promotable.append(session_data)
        
        logger.debug(
            f"Found {len(promotable)} promotable sessions "
            f"(min_messages={min_messages}, min_age={min_age_seconds}s)"
        )
        return promotable

    def analyze_session_importance(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze a session to determine its importance for promotion.
        
        Scores the session based on:
        - Message count and density
        - Presence of health-related keywords
        - User engagement level
        - Content diversity
        
        Args:
            session_id: Session to analyze
            
        Returns:
            Dict with importance metrics
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return {"importance_score": 0.0, "reason": "session_not_found"}
        
        messages = session_data.messages
        if not messages:
            return {"importance_score": 0.0, "reason": "no_messages"}
        
        # Scoring factors
        msg_count = len(messages)
        msg_score = min(msg_count / 10.0, 1.0)  # Normalize to 0-1
        
        # Check for medical/health keywords
        health_keywords = {
            "pain", "symptom", "medication", "doctor", "diagnosis",
            "blood pressure", "heart rate", "allergy", "prescription",
            "treatment", "condition", "surgery", "lab", "test",
        }
        all_content = " ".join(m.content.lower() for m in messages)
        health_matches = sum(1 for kw in health_keywords if kw in all_content)
        health_score = min(health_matches / 5.0, 1.0)
        
        # Engagement: ratio of user messages vs assistant
        user_msgs = sum(1 for m in messages if m.role == "user")
        engagement_score = min(user_msgs / max(msg_count, 1), 1.0)
        
        # Content diversity (unique words ratio)
        words = all_content.split()
        diversity_score = len(set(words)) / max(len(words), 1)
        
        # Weighted importance
        importance_score = (
            msg_score * 0.2 +
            health_score * 0.4 +
            engagement_score * 0.2 +
            diversity_score * 0.2
        )
        
        return {
            "importance_score": round(importance_score, 3),
            "message_count": msg_count,
            "health_relevance": round(health_score, 3),
            "engagement": round(engagement_score, 3),
            "content_diversity": round(diversity_score, 3),
            "promotable": importance_score >= 0.4,
        }


# ============================================================================
# Redis Session Buffer
# ============================================================================


class RedisSessionBuffer(SessionBufferInterface):
    """
    Redis-backed session buffer for production use.
    
    Features:
    - Automatic TTL-based expiration
    - Connection pooling
    - Automatic fallback to in-memory if Redis unavailable
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        key_prefix: str = "memori:session:",
        default_ttl_seconds: int = 3600,
        max_messages_per_session: int = 100,
        use_fallback: bool = True,
    ):
        """
        Initialize Redis session buffer.
        
        Args:
            redis_url: Redis connection URL (overrides host/port)
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            redis_password: Redis password
            key_prefix: Prefix for Redis keys
            default_ttl_seconds: Default TTL for sessions
            max_messages_per_session: Max messages to keep per session
            use_fallback: Use in-memory fallback if Redis unavailable
        """
        self.redis_url = redis_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.key_prefix = key_prefix
        self.default_ttl_seconds = default_ttl_seconds
        self.max_messages_per_session = max_messages_per_session
        self.use_fallback = use_fallback
        
        self._redis_client = None
        self._fallback_buffer: Optional[InMemorySessionBuffer] = None
        self._using_fallback = False
        self._lock = threading.Lock()
        
        self._initialize_redis()
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis
            
            if self.redis_url:
                self._redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                )
            else:
                self._redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    decode_responses=True,
                )
            
            # Test connection
            self._redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            self._using_fallback = False
            
        except ImportError:
            logger.warning("redis-py not installed, using in-memory fallback")
            self._init_fallback()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
            self._init_fallback()
    
    def _init_fallback(self) -> None:
        """Initialize in-memory fallback."""
        if self.use_fallback:
            self._fallback_buffer = InMemorySessionBuffer(
                default_ttl_seconds=self.default_ttl_seconds,
            )
            self._using_fallback = True
            logger.info("Using in-memory session buffer (fallback mode)")
        else:
            raise RuntimeError("Redis unavailable and fallback disabled")
    
    def _get_key(self, session_id: str) -> str:
        """Get Redis key for a session."""
        return f"{self.key_prefix}{session_id}"
    
    def _is_available(self) -> bool:
        """Check if Redis is available."""
        if self._redis_client is None:
            return False
        try:
            self._redis_client.ping()
            return True
        except Exception:
            return False
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        if self._using_fallback:
            return self._fallback_buffer.get_session(session_id)
        
        try:
            key = self._get_key(session_id)
            data = self._redis_client.get(key)
            if data:
                return SessionData.from_dict(json.loads(data))
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            if self._fallback_buffer:
                return self._fallback_buffer.get_session(session_id)
            return None
    
    def set_session(self, session_data: SessionData) -> bool:
        """Set/update session data."""
        if self._using_fallback:
            return self._fallback_buffer.set_session(session_data)
        
        try:
            key = self._get_key(session_data.session_id)
            session_data.updated_at = datetime.now().isoformat()
            
            # Trim messages if needed
            if len(session_data.messages) > self.max_messages_per_session:
                session_data.messages = session_data.messages[-self.max_messages_per_session:]
            
            data = json.dumps(session_data.to_dict())
            ttl = session_data.ttl_seconds or self.default_ttl_seconds
            self._redis_client.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            if self._fallback_buffer:
                return self._fallback_buffer.set_session(session_data)
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if self._using_fallback:
            return self._fallback_buffer.delete_session(session_id)
        
        try:
            key = self._get_key(session_id)
            result = self._redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            if self._fallback_buffer:
                return self._fallback_buffer.delete_session(session_id)
            return False
    
    def add_message(
        self,
        session_id: str,
        message: SessionMessage,
        create_if_missing: bool = True,
    ) -> bool:
        """Add a message to a session."""
        session_data = self.get_session(session_id)
        
        if session_data is None:
            if not create_if_missing:
                return False
            session_data = SessionData(
                session_id=session_id,
                ttl_seconds=self.default_ttl_seconds,
            )
        
        session_data.messages.append(message)
        return self.set_session(session_data)
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[SessionMessage]:
        """Get messages from a session."""
        session_data = self.get_session(session_id)
        if session_data is None:
            return []
        
        messages = session_data.messages
        if limit is not None:
            messages = messages[-limit:]
        return messages
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all messages from a session."""
        session_data = self.get_session(session_id)
        if session_data is None:
            return False
        
        session_data.messages = []
        return self.set_session(session_data)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        if self._using_fallback:
            return self._fallback_buffer.session_exists(session_id)
        
        try:
            key = self._get_key(session_id)
            return self._redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            if self._fallback_buffer:
                return self._fallback_buffer.session_exists(session_id)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if self._using_fallback:
            return self._fallback_buffer.get_stats()
        
        try:
            info = self._redis_client.info("keyspace")
            db_info = info.get(f"db{self.redis_db}", {})
            return {
                "type": "redis",
                "host": self.redis_host,
                "port": self.redis_port,
                "db": self.redis_db,
                "keys": db_info.get("keys", 0),
            }
        except Exception:
            return {"type": "redis", "status": "error"}
    
    def shutdown(self) -> None:
        """Shutdown the buffer."""
        if self._fallback_buffer:
            self._fallback_buffer.shutdown()
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                pass
    
    @property
    def is_using_fallback(self) -> bool:
        """Check if using in-memory fallback."""
        return self._using_fallback

    # ========================================================================
    # Async API (for use in async contexts alongside the rest of the system)
    # ========================================================================

    async def async_get_session(self, session_id: str) -> Optional[SessionData]:
        """Async wrapper for get_session."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_session, session_id)

    async def async_set_session(self, session_data: SessionData) -> bool:
        """Async wrapper for set_session."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.set_session, session_data)

    async def async_add_message(
        self,
        session_id: str,
        message: SessionMessage,
        create_if_missing: bool = True,
    ) -> bool:
        """Async wrapper for add_message."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.add_message, session_id, message, create_if_missing
        )

    async def async_get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[SessionMessage]:
        """Async wrapper for get_messages."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_messages, session_id, limit)

    # ========================================================================
    # Promotion Support (Short-term → Long-term)
    # ========================================================================

    def get_promotable_sessions(
        self,
        min_messages: int = 3,
        min_age_seconds: float = 300,
    ) -> List[SessionData]:
        """
        Get sessions eligible for promotion to long-term memory.
        
        For Redis-backed storage, scans all sessions matching the key prefix.
        Falls back to in-memory buffer's method if using fallback.
        
        Args:
            min_messages: Minimum messages for eligibility
            min_age_seconds: Minimum session age in seconds
            
        Returns:
            List of promotable SessionData objects
        """
        if self._using_fallback and self._fallback_buffer:
            return self._fallback_buffer.get_promotable_sessions(
                min_messages=min_messages,
                min_age_seconds=min_age_seconds,
            )
        
        promotable = []
        now = datetime.now()
        
        try:
            # Scan Redis for all session keys
            pattern = f"{self.key_prefix}*"
            cursor = 0
            keys = []
            
            while True:
                cursor, batch_keys = self._redis_client.scan(
                    cursor=cursor, match=pattern, count=100
                )
                keys.extend(batch_keys)
                if cursor == 0:
                    break
            
            # Use Redis pipeline for efficient batch get
            if keys:
                pipe = self._redis_client.pipeline()
                for key in keys:
                    pipe.get(key)
                values = pipe.execute()
                
                for key, data in zip(keys, values):
                    if not data:
                        continue
                    try:
                        session_data = SessionData.from_dict(json.loads(data))
                        msg_count = len(session_data.messages)
                        
                        # Check age
                        try:
                            created_dt = datetime.fromisoformat(session_data.created_at)
                            age_seconds = (now - created_dt).total_seconds()
                        except (ValueError, TypeError):
                            age_seconds = min_age_seconds + 1  # Assume old enough
                        
                        if msg_count >= min_messages and age_seconds >= min_age_seconds:
                            promotable.append(session_data)
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to parse session {key}: {e}")
            
        except Exception as e:
            logger.error(f"Error scanning promotable sessions: {e}")
        
        logger.debug(f"Found {len(promotable)} promotable sessions in Redis")
        return promotable

    def analyze_session_importance(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze a session to determine its importance for promotion.
        Delegates to the fallback buffer's implementation for consistency.
        """
        if self._using_fallback and self._fallback_buffer:
            return self._fallback_buffer.analyze_session_importance(session_id)
        
        session_data = self.get_session(session_id)
        if not session_data:
            return {"importance_score": 0.0, "reason": "session_not_found"}
        
        messages = session_data.messages
        if not messages:
            return {"importance_score": 0.0, "reason": "no_messages"}
        
        # Same scoring logic as InMemorySessionBuffer
        msg_count = len(messages)
        msg_score = min(msg_count / 10.0, 1.0)
        
        health_keywords = {
            "pain", "symptom", "medication", "doctor", "diagnosis",
            "blood pressure", "heart rate", "allergy", "prescription",
            "treatment", "condition", "surgery", "lab", "test",
        }
        all_content = " ".join(m.content.lower() for m in messages)
        health_matches = sum(1 for kw in health_keywords if kw in all_content)
        health_score = min(health_matches / 5.0, 1.0)
        
        user_msgs = sum(1 for m in messages if m.role == "user")
        engagement_score = min(user_msgs / max(msg_count, 1), 1.0)
        
        words = all_content.split()
        diversity_score = len(set(words)) / max(len(words), 1)
        
        importance_score = (
            msg_score * 0.2 +
            health_score * 0.4 +
            engagement_score * 0.2 +
            diversity_score * 0.2
        )
        
        return {
            "importance_score": round(importance_score, 3),
            "message_count": msg_count,
            "health_relevance": round(health_score, 3),
            "engagement": round(engagement_score, 3),
            "content_diversity": round(diversity_score, 3),
            "promotable": importance_score >= 0.4,
        }


# ============================================================================
# Singleton Instance and Factory Functions
# ============================================================================


# Global instance (lazy initialization)
_session_buffer: Optional[RedisSessionBuffer] = None
_lock = threading.Lock()


def get_session_buffer(
    redis_url: Optional[str] = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    default_ttl_seconds: int = 3600,
    **kwargs,
) -> RedisSessionBuffer:
    """
    Get or create the singleton RedisSessionBuffer instance.
    
    Args:
        redis_url: Optional Redis URL
        redis_host: Redis host (default: localhost)
        redis_port: Redis port (default: 6379)
        default_ttl_seconds: Default session TTL
        **kwargs: Additional arguments for RedisSessionBuffer
        
    Returns:
        RedisSessionBuffer instance (with in-memory fallback if Redis unavailable)
    """
    global _session_buffer
    
    with _lock:
        if _session_buffer is None:
            _session_buffer = RedisSessionBuffer(
                redis_url=redis_url,
                redis_host=redis_host,
                redis_port=redis_port,
                default_ttl_seconds=default_ttl_seconds,
                use_fallback=True,  # Always use fallback
                **kwargs,
            )
            
            mode = "in-memory fallback" if _session_buffer.is_using_fallback else "Redis"
            logger.info(f"Created singleton RedisSessionBuffer instance ({mode})")
        
        return _session_buffer


def shutdown_session_buffer() -> None:
    """Shutdown the global session buffer."""
    global _session_buffer
    
    with _lock:
        if _session_buffer is not None:
            _session_buffer.shutdown()
            _session_buffer = None
            logger.info("Shutdown session buffer")
