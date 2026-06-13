# Chat history management for the NLP service.
"""Chat history management for the NLP service.

Provides a per-session store with:
- In-memory LRU cache for fast access
- Optional database persistence (SQLite/PostgreSQL)
- Session timeout policies
- Automatic cleanup of stale sessions
- Optional query optimization with batch inserts and tiered caching
"""


import json
import os
import time
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from threading import RLock
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Optional dependencies
try:
    from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Import query optimization modules (optional)
try:
    from core.database.query_optimizer import (
        TieredCache,
        BatchInsertManager,
        OptimizedChatHistoryQueries,
        QueryOptimizationConfig,
    )
    from core.database.query_monitor import (
        get_slow_query_logger,
        get_performance_monitor,
        QueryMetrics,
    )
    QUERY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    QUERY_OPTIMIZATION_AVAILABLE = False
    logger.info("Query optimization modules not available - using standard queries")

# Import encryption service for PHI protection
try:
    from core.services.encryption_service import get_encryption_service

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logger.warning(
        "Encryption service not available - chat metadata will not be encrypted"
    )

# SQLAlchemy model for persistent chat history
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class ChatMessage(Base):
        """SQLAlchemy model for chat messages."""

        __tablename__ = "chat_messages"

        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(255), index=True, nullable=False)
        user_id = Column(String(255), index=True, default="default")
        role = Column(String(50), nullable=False)  # "user", "assistant", "system"
        content = Column(Text, nullable=False)
        metadata_json = Column(Text, default="{}")  # JSON serialized metadata
        created_at = Column(DateTime, default=datetime.utcnow, index=True)

        def to_dict(self, encryption_service=None) -> Dict[str, Any]:
            """Convert to dictionary with optional metadata decryption."""
            # Parse metadata, decrypting if necessary
            metadata = {}
            if self.metadata_json:
                try:
                    if encryption_service and self.metadata_json.startswith("ENC:"):
                        # Decrypt encrypted metadata
                        decrypted_metadata = encryption_service.decrypt_field(
                            self.metadata_json
                        )
                        metadata = json.loads(decrypted_metadata)
                    else:
                        # Legacy unencrypted metadata
                        metadata = json.loads(self.metadata_json)
                except Exception as e:
                    logger.error(f"Failed to parse/decrypt metadata: {e}")
                    # Fall back to raw metadata
                    try:
                        metadata = json.loads(self.metadata_json)
                    except Exception as e:
                        logger.warning(f"Failed to parse metadata JSON: {e}")
                        metadata = {}

            return {
                "id": self.id,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "role": self.role,
                "content": self.content,
                "metadata": metadata,
                "created_at": self.created_at.isoformat() if self.created_at else None,
            }

    class ChatSession(Base):
        """SQLAlchemy model for chat sessions."""

        __tablename__ = "chat_sessions"

        session_id = Column(String(255), primary_key=True)
        user_id = Column(String(255), index=True, default="default")
        created_at = Column(DateTime, default=datetime.utcnow)
        last_activity = Column(
            DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
        )
        message_count = Column(Integer, default=0)
        metadata_json = Column(Text, default="{}")

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_activity": (
                    self.last_activity.isoformat() if self.last_activity else None
                ),
                "message_count": self.message_count,
                "metadata": (
                    json.loads(self.metadata_json) if self.metadata_json else {}
                ),
            }


class SessionTimeoutPolicy:
    """
    Session timeout policy configuration.

    Defines when sessions should be considered expired and cleaned up.
    """

    def __init__(
        self,
        idle_timeout: int = 3600,  # 1 hour default
        absolute_timeout: int = 86400,  # 24 hours default
        max_messages_per_session: int = 1000,
        cleanup_interval: int = 300,  # 5 minutes
        auto_cleanup: bool = True,
    ):
        """
        Initialize session timeout policy.

        Args:
            idle_timeout: Seconds of inactivity before session expires
            absolute_timeout: Maximum session lifetime in seconds
            max_messages_per_session: Maximum messages before session archived
            cleanup_interval: Seconds between cleanup runs
            auto_cleanup: Enable automatic background cleanup
        """
        self.idle_timeout = idle_timeout
        self.absolute_timeout = absolute_timeout
        self.max_messages_per_session = max_messages_per_session
        self.cleanup_interval = cleanup_interval
        self.auto_cleanup = auto_cleanup

    def is_expired(
        self, last_activity: datetime, created_at: datetime, message_count: int = 0
    ) -> bool:
        """
        Check if a session has expired based on policy.

        Args:
            last_activity: Timestamp of last session activity
            created_at: Session creation timestamp
            message_count: Number of messages in session

        Returns:
            True if session is expired
        """
        now = datetime.utcnow()

        # Check idle timeout
        if (now - last_activity).total_seconds() > self.idle_timeout:
            return True

        # Check absolute timeout
        if (now - created_at).total_seconds() > self.absolute_timeout:
            return True

        # Check message limit
        if message_count >= self.max_messages_per_session:
            return True

        return False


class PersistentChatHistory:
    """
    Database-backed chat history with session timeout policies.

    Provides persistent storage with automatic session cleanup.
    Optionally uses query optimization with batch inserts and tiered caching.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        timeout_policy: Optional[SessionTimeoutPolicy] = None,
        in_memory_cache_size: int = 100,
        use_optimized_queries: bool = True,
        enable_batch_inserts: bool = False,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize persistent chat history.

        Args:
            database_url: SQLAlchemy database URL. Defaults to PostgreSQL from AppConfig
            timeout_policy: Session timeout configuration
            in_memory_cache_size: Size of LRU cache for recent sessions
            use_optimized_queries: Use optimized query layer with tiered caching
            enable_batch_inserts: Enable batch insert mode for high throughput
            redis_url: Redis URL for L2 cache (optional, uses env var if not provided)
        """
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy not available. Install with: pip install sqlalchemy"
            )

        # Default to PostgreSQL from AppConfig
        if database_url is None:
            database_url = os.environ.get("CHAT_HISTORY_DB_URL")
            
            if not database_url:
                # Build PostgreSQL URL from AppConfig
                try:
                    from core.config.app_config import get_app_config
                    config = get_app_config()
                    db = config.database
                    database_url = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
                except Exception as e:
                    logger.warning(f"Could not load AppConfig for chat history: {e}")
                    # Fallback to environment variables
                    host = os.environ.get("POSTGRES_HOST", "localhost")
                    port = os.environ.get("POSTGRES_PORT", "5432")
                    user = os.environ.get("POSTGRES_USER", "postgres")
                    password = os.environ.get("POSTGRES_PASSWORD", "")
                    db_name = os.environ.get("POSTGRES_DB", "heartguard")
                    database_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        
        self.database_url = database_url
        self.timeout_policy = timeout_policy or SessionTimeoutPolicy()
        self.in_memory_cache_size = in_memory_cache_size
        self.use_optimized_queries = use_optimized_queries and QUERY_OPTIMIZATION_AVAILABLE
        self.enable_batch_inserts = enable_batch_inserts

        # Initialize encryption service if available
        self._encryption = None
        if ENCRYPTION_AVAILABLE:
            try:
                self._encryption = get_encryption_service()
                logger.info("Chat history encryption service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize encryption service: {e}")

        # Initialize database
        self.engine = create_engine(
            self.database_url, pool_size=5, max_overflow=10, pool_pre_ping=True
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize optimized query components if available
        self._tiered_cache: Optional[Any] = None
        self._batch_manager: Optional[Any] = None
        
        if self.use_optimized_queries:
            try:
                # Initialize tiered cache (L1 in-memory + L2 Redis)
                redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
                self._tiered_cache = TieredCache(
                    l1_max_size=in_memory_cache_size,
                    l1_ttl_seconds=60,
                    redis_url=redis_url,
                    l2_ttl_seconds=300,
                )
                logger.info("Tiered cache (L1+L2) initialized for chat history")
                
                if enable_batch_inserts:
                    self._batch_manager = BatchInsertManager(
                        table_name="chat_messages",
                        batch_size=100,
                        flush_interval_seconds=1.0,
                    )
                    logger.info("Batch insert manager initialized for chat history")
            except Exception as e:
                logger.warning(f"Failed to initialize query optimizations: {e}")
                self.use_optimized_queries = False

        # In-memory LRU cache for recent sessions (fallback)
        self._cache: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
        self._cache_timestamps: Dict[str, float] = {}
        self._lock = RLock()

        # Background cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stop = threading.Event()

        if self.timeout_policy.auto_cleanup:
            self._start_cleanup_thread()

        logger.info(
            f"PersistentChatHistory initialized with database: {self.database_url}"
        )

    @contextmanager
    def _get_db(self):
        """Get database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        def cleanup_loop():
            while not self._cleanup_stop.wait(self.timeout_policy.cleanup_interval):
                try:
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Chat history cleanup thread started")

    def _stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        if self._cleanup_thread:
            self._cleanup_stop.set()
            self._cleanup_thread.join(timeout=5)
            self._cleanup_thread = None

    def _update_cache(self, session_id: str, history: List[Dict[str, str]]) -> None:
        """Update cache with LRU eviction. Uses tiered cache if available."""
        cache_key = f"chat_history:{session_id}"
        
        # Use tiered cache if available (L1 + L2 Redis)
        if self._tiered_cache:
            try:
                self._tiered_cache.set(cache_key, history)
                return
            except Exception as e:
                logger.warning(f"Tiered cache set failed, falling back to local cache: {e}")
        
        # Fallback to local LRU cache
        with self._lock:
            if session_id in self._cache:
                self._cache.move_to_end(session_id)
            else:
                if len(self._cache) >= self.in_memory_cache_size:
                    self._cache.popitem(last=False)
                    oldest_key = next(iter(self._cache_timestamps), None)
                    if oldest_key:
                        self._cache_timestamps.pop(oldest_key, None)

            self._cache[session_id] = history
            self._cache_timestamps[session_id] = time.time()

    def _get_from_cache(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """Get session from cache if available and not stale. Uses tiered cache if available."""
        cache_key = f"chat_history:{session_id}"
        
        # Use tiered cache if available (L1 + L2 Redis)
        if self._tiered_cache:
            try:
                cached = self._tiered_cache.get(cache_key)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Tiered cache get failed, falling back to local cache: {e}")
        
        # Fallback to local LRU cache
        with self._lock:
            if session_id not in self._cache:
                return None

            # Check if cache entry is stale (older than 60 seconds)
            timestamp = self._cache_timestamps.get(session_id, 0)
            if time.time() - timestamp > 60:
                return None

            self._cache.move_to_end(session_id)
            return list(self._cache[session_id])
    
    def _invalidate_cache(self, session_id: str) -> None:
        """Invalidate cache entry for a session."""
        cache_key = f"chat_history:{session_id}"
        
        # Invalidate tiered cache if available
        if self._tiered_cache:
            try:
                self._tiered_cache.delete(cache_key)
            except Exception as e:
                logger.warning(f"Tiered cache delete failed: {e}")
        
        # Invalidate local cache
        with self._lock:
            self._cache.pop(session_id, None)
            self._cache_timestamps.pop(session_id, None)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a message to the chat history.

        Args:
            session_id: Session identifier
            role: Message role ("user", "assistant", "system")
            content: Message content
            user_id: User identifier for multi-tenant support
            metadata: Optional message metadata

        Returns:
            Message ID
        """
        # Encrypt metadata if encryption service is available
        encrypted_metadata = None
        if metadata and self._encryption:
            try:
                metadata_json = json.dumps(metadata)
                encrypted_metadata = self._encryption.encrypt_field(metadata_json)
            except Exception as e:
                logger.error(f"Failed to encrypt metadata: {e}")
                # Fall back to unencrypted metadata
                encrypted_metadata = json.dumps(metadata)
        else:
            encrypted_metadata = json.dumps(metadata or {})

        with self._get_db() as db:
            # Ensure session exists
            chat_session = (
                db.query(ChatSession).filter_by(session_id=session_id).first()
            )
            if not chat_session:
                chat_session = ChatSession(session_id=session_id, user_id=user_id)
                db.add(chat_session)
            else:
                chat_session.last_activity = datetime.utcnow()
                chat_session.message_count = (chat_session.message_count or 0) + 1

            # Encrypt content if encryption service is available
            encrypted_content = content
            if self._encryption:
                try:
                    encrypted_content = self._encryption.encrypt_field(content)
                except Exception as e:
                    logger.error(f"Failed to encrypt content: {e}")
                    # Fall back to unencrypted content

            # Add message with encrypted content and metadata
            message = ChatMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=encrypted_content,
                metadata_json=encrypted_metadata,
            )
            db.add(message)
            db.flush()
            message_id = message.id

        # Invalidate cache using unified method
        self._invalidate_cache(session_id)

        return message_id

    def get_history(
        self, session_id: str, limit: int = 50, include_metadata: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get chat history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum messages to return
            include_metadata: Include message metadata

        Returns:
            List of message dictionaries
        """
        # Check cache first
        cached = self._get_from_cache(session_id)
        if cached is not None:
            return cached[-limit:] if len(cached) > limit else cached

        with self._get_db() as db:
            messages = (
                db.query(ChatMessage)
                .filter_by(session_id=session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
                .all()
            )

            # Reverse to chronological order
            messages = list(reversed(messages))

            if include_metadata:
                # Decrypt metadata if encryption service is available
                history = []
                for m in messages:
                    message_dict = {
                        "id": m.id,
                        "session_id": m.session_id,
                        "user_id": m.user_id,
                        "role": m.role,
                        "content": m.content,
                        "created_at": (
                            m.created_at.isoformat() if m.created_at else None
                        ),
                    }

                    # Decrypt content if it's encrypted
                    if m.content and self._encryption:
                        try:
                            if m.content.startswith("ENC:"):
                                # Decrypt encrypted content
                                message_dict["content"] = (
                                    self._encryption.decrypt_field(m.content)
                                )
                        except Exception as e:
                            logger.error(f"Failed to decrypt content: {e}")
                            # Fall back to raw content

                    # Decrypt metadata if it's encrypted
                    if m.metadata_json and self._encryption:
                        try:
                            if m.metadata_json.startswith("ENC:"):
                                # Decrypt encrypted metadata
                                decrypted_metadata = self._encryption.decrypt_field(
                                    m.metadata_json
                                )
                                message_dict["metadata"] = json.loads(
                                    decrypted_metadata
                                )
                            else:
                                # Legacy unencrypted metadata
                                message_dict["metadata"] = json.loads(m.metadata_json)
                        except Exception as e:
                            logger.error(f"Failed to decrypt metadata: {e}")
                            # Fall back to raw metadata
                            try:
                                message_dict["metadata"] = json.loads(m.metadata_json)
                            except Exception as e:
                                logger.warning(f"Failed to parse metadata JSON fallback: {e}")
                                message_dict["metadata"] = {}
                    else:
                        # No encryption or no metadata
                        try:
                            message_dict["metadata"] = json.loads(
                                m.metadata_json or "{}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to parse metadata JSON: {e}")
                            message_dict["metadata"] = {}

                    history.append(message_dict)
            else:
                history = []
            for m in messages:
                content = m.content
                # Decrypt content if it's encrypted
                if content and self._encryption:
                    try:
                        if content.startswith("ENC:"):
                            # Decrypt encrypted content
                            content = self._encryption.decrypt_field(content)
                    except Exception as e:
                        logger.error(f"Failed to decrypt content: {e}")
                        # Fall back to raw content
                history.append({"role": m.role, "content": content})

        # Update cache
        self._update_cache(session_id, history)

        return history

    def clear(self, session_id: str) -> None:
        """Clear all messages for a session."""
        with self._get_db() as db:
            db.query(ChatMessage).filter_by(session_id=session_id).delete()
            db.query(ChatSession).filter_by(session_id=session_id).delete()

        # Invalidate cache using unified method
        self._invalidate_cache(session_id)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata and statistics."""
        with self._get_db() as db:
            session = db.query(ChatSession).filter_by(session_id=session_id).first()
            if not session:
                return None

            message_count = (
                db.query(ChatMessage).filter_by(session_id=session_id).count()
            )

            return {
                **session.to_dict(),
                "message_count": message_count,
                "is_expired": self.timeout_policy.is_expired(
                    session.last_activity or session.created_at,
                    session.created_at,
                    message_count,
                ),
            }

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        include_expired: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List available chat sessions.

        Args:
            user_id: Filter by user (optional)
            include_expired: Include expired sessions
            limit: Maximum sessions to return
        """
        with self._get_db() as db:
            query = db.query(ChatSession)

            if user_id:
                query = query.filter_by(user_id=user_id)

            if not include_expired:
                # Filter out expired sessions
                cutoff = datetime.utcnow() - timedelta(
                    seconds=self.timeout_policy.idle_timeout
                )
                query = query.filter(ChatSession.last_activity >= cutoff)

            sessions = (
                query.order_by(ChatSession.last_activity.desc()).limit(limit).all()
            )

            return [s.to_dict() for s in sessions]

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions based on timeout policy.

        Returns:
            Number of sessions cleaned up
        """
        with self._get_db() as db:
            now = datetime.utcnow()
            idle_cutoff = now - timedelta(seconds=self.timeout_policy.idle_timeout)
            absolute_cutoff = now - timedelta(
                seconds=self.timeout_policy.absolute_timeout
            )

            # Find expired sessions
            expired = (
                db.query(ChatSession)
                .filter(
                    (ChatSession.last_activity < idle_cutoff)
                    | (ChatSession.created_at < absolute_cutoff)
                )
                .all()
            )

            count = 0
            for session in expired:
                session_id = session.session_id
                db.query(ChatMessage).filter_by(session_id=session_id).delete()
                db.delete(session)
                count += 1

                # Clear from cache
                with self._lock:
                    self._cache.pop(session_id, None)
                    self._cache_timestamps.pop(session_id, None)

            if count > 0:
                logger.info(f"Cleaned up {count} expired chat sessions")

            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get chat history statistics."""
        with self._get_db() as db:
            total_sessions = db.query(ChatSession).count()
            total_messages = db.query(ChatMessage).count()

            # Active sessions (within idle timeout)
            cutoff = datetime.utcnow() - timedelta(
                seconds=self.timeout_policy.idle_timeout
            )
            active_sessions = (
                db.query(ChatSession)
                .filter(ChatSession.last_activity >= cutoff)
                .count()
            )

            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "cache_size": len(self._cache),
                "cache_max_size": self.in_memory_cache_size,
                "timeout_policy": {
                    "idle_timeout": self.timeout_policy.idle_timeout,
                    "absolute_timeout": self.timeout_policy.absolute_timeout,
                    "max_messages_per_session": self.timeout_policy.max_messages_per_session,
                    "auto_cleanup": self.timeout_policy.auto_cleanup,
                },
            }

    def close(self) -> None:
        """Close resources and stop background threads."""
        self._stop_cleanup_thread()
        self.engine.dispose()
        logger.info("PersistentChatHistory closed")


class ChatHistory:
    """
    Hybrid chat history manager with optional database persistence.

    Provides backward-compatible in-memory storage with optional
    database persistence for durability.
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        max_messages: int = 50,
        persistence_enabled: bool = False,
        database_url: Optional[str] = None,
        timeout_policy: Optional[SessionTimeoutPolicy] = None,
    ):
        """
        Initialize chat history manager.

        Args:
            max_sessions: Maximum in-memory sessions
            max_messages: Maximum messages per session
            persistence_enabled: Enable database persistence
            database_url: Database URL for persistence
            timeout_policy: Session timeout configuration
        """
        self._store: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_sessions = max_sessions
        self.max_messages = max_messages
        self._lock = RLock()

        # Optional persistent backend
        self._persistent: Optional[PersistentChatHistory] = None
        self._persistence_enabled = persistence_enabled

        if persistence_enabled and SQLALCHEMY_AVAILABLE:
            try:
                self._persistent = PersistentChatHistory(
                    database_url=database_url, timeout_policy=timeout_policy
                )
                logger.info("Chat history persistence enabled")
            except Exception as e:
                logger.error(f"Failed to initialize persistence: {e}")
                self._persistence_enabled = False

    def _ensure_session(self, session_id: str) -> List[Dict[str, str]]:
        if session_id not in self._store:
            if len(self._store) >= self.max_sessions:
                self._store.popitem(last=False)
            self._store[session_id] = []
            self._session_metadata[session_id] = {
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
            }
        self._store.move_to_end(session_id)
        return self._store[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to the chat history."""
        with self._lock:
            history = self._ensure_session(session_id)
            history.append({"role": role, "content": content})

            # Update metadata
            if session_id in self._session_metadata:
                self._session_metadata[session_id]["last_activity"] = datetime.utcnow()

            if len(history) > self.max_messages:
                history.pop(0)

        # Persist to database if enabled
        if self._persistent:
            try:
                self._persistent.add_message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    user_id=user_id,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Failed to persist message: {e}")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """Get chat history for a session."""
        # Try in-memory first
        with self._lock:
            if session_id in self._store:
                history = list(self._store[session_id])
                return history[-limit:] if len(history) > limit else history

        # Fall back to persistent storage
        if self._persistent:
            try:
                return self._persistent.get_history(session_id, limit=limit)
            except Exception as e:
                logger.error(f"Failed to load history from persistence: {e}")

        return []

    def clear(self, session_id: str) -> None:
        """Clear chat history for a session."""
        with self._lock:
            self._store.pop(session_id, None)
            self._session_metadata.pop(session_id, None)

        if self._persistent:
            try:
                self._persistent.clear(session_id)
            except Exception as e:
                logger.error(f"Failed to clear persistent history: {e}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata."""
        with self._lock:
            if session_id in self._session_metadata:
                meta = self._session_metadata[session_id]
                return {
                    "session_id": session_id,
                    "created_at": meta.get("created_at"),
                    "last_activity": meta.get("last_activity"),
                    "message_count": len(self._store.get(session_id, [])),
                }

        if self._persistent:
            return self._persistent.get_session_info(session_id)

        return None

    def list_active_sessions(self, limit: int = 100) -> List[str]:
        """List active session IDs."""
        with self._lock:
            return list(self._store.keys())[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get chat history statistics."""
        stats = {
            "in_memory_sessions": len(self._store),
            "max_sessions": self.max_sessions,
            "max_messages_per_session": self.max_messages,
            "persistence_enabled": self._persistence_enabled,
        }

        if self._persistent:
            stats["persistent"] = self._persistent.get_stats()

        return stats

    def close(self) -> None:
        """Close resources."""
        if self._persistent:
            self._persistent.close()
            self._persistent = None


# Singleton instance for the service
# Configure via environment variables:
#   CHAT_HISTORY_PERSISTENCE=true
#   CHAT_HISTORY_DB_URL=sqlite:///chat_history.db
#   CHAT_SESSION_IDLE_TIMEOUT=3600
#   CHAT_SESSION_ABSOLUTE_TIMEOUT=86400
_persistence_enabled = os.environ.get("CHAT_HISTORY_PERSISTENCE", "").lower() in (
    "true",
    "1",
    "yes",
)
_timeout_policy = (
    SessionTimeoutPolicy(
        idle_timeout=int(os.environ.get("CHAT_SESSION_IDLE_TIMEOUT", "3600")),
        absolute_timeout=int(os.environ.get("CHAT_SESSION_ABSOLUTE_TIMEOUT", "86400")),
        auto_cleanup=_persistence_enabled,
    )
    if _persistence_enabled
    else None
)

chat_history_manager = ChatHistory(
    persistence_enabled=_persistence_enabled,
    database_url=os.environ.get("CHAT_HISTORY_DB_URL"),
    timeout_policy=_timeout_policy,
)
