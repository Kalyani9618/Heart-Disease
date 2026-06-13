"""
MemoryManager - Modular memory management system for Memori

This is a working implementation that coordinates interceptors and provides
a clean interface for memory management operations.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from loguru import logger

# Redis for distributed caching (optional)
# aioredis is deprecated and has compatibility issues with Python 3.11+
# Use redis.asyncio instead (part of redis-py >= 4.2)
try:
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.debug("redis.asyncio not available - Redis persistence disabled")
except Exception as e:
    REDIS_AVAILABLE = False
    logger.debug(f"Redis initialization failed: {e} - Redis persistence disabled")

# Interceptor system removed - using LiteLLM native callbacks only


class MemoryPersistenceBackend:
    """
    Abstract base for memory persistence backends.
    Supports multiple storage options: Redis, File, or custom backends.
    """
    
    def save_snapshot(self, user_id: str, data: dict[str, Any]) -> bool:
        """Save a memory snapshot."""
        raise NotImplementedError
    
    def load_snapshot(self, user_id: str) -> Optional[dict[str, Any]]:
        """Load a memory snapshot."""
        raise NotImplementedError
    
    def delete_snapshot(self, user_id: str) -> bool:
        """Delete a memory snapshot."""
        raise NotImplementedError
    
    def list_snapshots(self) -> list[str]:
        """List all available snapshot user IDs."""
        raise NotImplementedError


class FileSystemPersistence(MemoryPersistenceBackend):
    """File-based memory persistence backend."""
    
    def __init__(self, storage_dir: str = "./memory_snapshots"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.debug(f"FileSystemPersistence initialized at {storage_dir}")
    
    def _get_filepath(self, user_id: str) -> str:
        """Get the file path for a user's snapshot."""
        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.storage_dir, f"{safe_user_id}_snapshot.json")
    
    def save_snapshot(self, user_id: str, data: dict[str, Any]) -> bool:
        """Save a memory snapshot to file."""
        try:
            filepath = self._get_filepath(user_id)
            snapshot = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
            logger.info(f"Memory snapshot saved for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save snapshot for {user_id}: {e}")
            return False
    
    def load_snapshot(self, user_id: str) -> Optional[dict[str, Any]]:
        """Load a memory snapshot from file."""
        try:
            filepath = self._get_filepath(user_id)
            if not os.path.exists(filepath):
                logger.debug(f"No snapshot found for user {user_id}")
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            logger.info(f"Memory snapshot loaded for user {user_id}")
            return snapshot.get("data")
        except Exception as e:
            logger.error(f"Failed to load snapshot for {user_id}: {e}")
            return None
    
    def delete_snapshot(self, user_id: str) -> bool:
        """Delete a memory snapshot file."""
        try:
            filepath = self._get_filepath(user_id)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Memory snapshot deleted for user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete snapshot for {user_id}: {e}")
            return False
    
    def list_snapshots(self) -> list[str]:
        """List all available snapshot user IDs."""
        try:
            files = os.listdir(self.storage_dir)
            user_ids = [
                f.replace("_snapshot.json", "")
                for f in files
                if f.endswith("_snapshot.json")
            ]
            return user_ids
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []


class RedisPersistence(MemoryPersistenceBackend):
    """Redis-based memory persistence backend with async support."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "memori:snapshot:",
        ttl: int = 86400 * 30  # 30 days default
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl = ttl
        self._redis: Optional[aioredis.Redis] = None
        self._sync_mode = True  # Will use sync wrappers for compatibility
        logger.debug(f"RedisPersistence initialized with prefix {key_prefix}")
    
    async def _get_redis(self) -> "aioredis.Redis":
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis
    
    def _get_key(self, user_id: str) -> str:
        """Generate Redis key for a user."""
        return f"{self.key_prefix}{user_id}"
    
    def save_snapshot(self, user_id: str, data: dict[str, Any]) -> bool:
        """Save a memory snapshot to Redis (sync wrapper)."""
        import asyncio
        
        async def _async_save():
            try:
                redis = await self._get_redis()
                key = self._get_key(user_id)
                snapshot = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }
                await redis.set(key, json.dumps(snapshot, default=str), ex=self.ttl)
                logger.info(f"Memory snapshot saved to Redis for user {user_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to save snapshot to Redis for {user_id}: {e}")
                return False
        
        try:
            loop = asyncio.get_running_loop()
            # If already in async context, create task
            future = asyncio.ensure_future(_async_save())
            return True  # Optimistic return, actual save is async
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_async_save())
    
    def load_snapshot(self, user_id: str) -> Optional[dict[str, Any]]:
        """Load a memory snapshot from Redis (sync wrapper)."""
        import asyncio
        
        async def _async_load():
            try:
                redis = await self._get_redis()
                key = self._get_key(user_id)
                data = await redis.get(key)
                if data is None:
                    logger.debug(f"No snapshot found in Redis for user {user_id}")
                    return None
                snapshot = json.loads(data)
                logger.info(f"Memory snapshot loaded from Redis for user {user_id}")
                return snapshot.get("data")
            except Exception as e:
                logger.error(f"Failed to load snapshot from Redis for {user_id}: {e}")
                return None
        
        try:
            asyncio.get_running_loop()
            # Cannot run sync in async context, return None
            logger.warning("Cannot load Redis snapshot synchronously in async context")
            return None
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_async_load())
    
    def delete_snapshot(self, user_id: str) -> bool:
        """Delete a memory snapshot from Redis (sync wrapper)."""
        import asyncio
        
        async def _async_delete():
            try:
                redis = await self._get_redis()
                key = self._get_key(user_id)
                result = await redis.delete(key)
                if result:
                    logger.info(f"Memory snapshot deleted from Redis for user {user_id}")
                return bool(result)
            except Exception as e:
                logger.error(f"Failed to delete snapshot from Redis for {user_id}: {e}")
                return False
        
        try:
            asyncio.get_running_loop()
            future = asyncio.ensure_future(_async_delete())
            return True
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_async_delete())
    
    def list_snapshots(self) -> list[str]:
        """List all available snapshot user IDs from Redis."""
        import asyncio
        
        async def _async_list():
            try:
                redis = await self._get_redis()
                keys = []
                async for key in redis.scan_iter(f"{self.key_prefix}*"):
                    user_id = key.replace(self.key_prefix, "")
                    keys.append(user_id)
                return keys
            except Exception as e:
                logger.error(f"Failed to list snapshots from Redis: {e}")
                return []
        
        try:
            asyncio.get_running_loop()
            # Cannot list synchronously in async context
            return []
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_async_list())
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


class MemoryCache:
    """
    In-memory LRU cache for frequently accessed memories.
    Provides fast access layer before hitting persistence backend.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_order: list[str] = []
        import threading
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            import time
            if time.time() - timestamp > self.ttl:
                # Expired
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # Update access order (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with LRU eviction."""
        import time
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
            
            self._cache[key] = (value, time.time())
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0
            }


class MemoryManager:
    """
    Modular memory management system that coordinates interceptors,
    memory processing, and context injection.

    This class provides a clean interface for memory operations while
    maintaining backward compatibility with the existing Memori system.
    
    Supports multiple persistence backends:
    - FileSystem: Local file-based persistence (default)
    - Redis: Distributed caching with TTL support
    - Custom: Implement MemoryPersistenceBackend interface
    """

    def __init__(
        self,
        database_connect: str | None = None,  # PostgreSQL from AppConfig if None
        template: str = "basic",
        mem_prompt: str | None = None,
        conscious_ingest: bool = False,
        auto_ingest: bool = False,
        namespace: str | None = None,
        shared_memory: bool = False,
        memory_filters: list[str] | None = None,
        user_id: str | None = None,
        verbose: bool = False,
        provider_config: Any | None = None,
        # Persistence configuration
        persistence_backend: str = "filesystem",  # "filesystem", "redis", or "none"
        persistence_config: dict[str, Any] | None = None,
        enable_cache: bool = True,
        cache_max_size: int = 1000,
        cache_ttl: int = 300,
        # Additional parameters for compatibility
        openai_api_key: str | None = None,
        api_key: str | None = None,
        api_type: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        azure_ad_token: str | None = None,
        organization: str | None = None,
        **kwargs,
    ):
        """
        Initialize the MemoryManager.

        Args:
            database_connect: Database connection string
            template: Memory template to use
            mem_prompt: Optional memory prompt
            conscious_ingest: Enable conscious memory ingestion
            auto_ingest: Enable automatic memory ingestion
            namespace: Optional namespace for memory isolation
            shared_memory: Enable shared memory across agents
            memory_filters: Optional memory filters
            user_id: Optional user identifier
            verbose: Enable verbose logging
            provider_config: Provider configuration
            persistence_backend: Backend type ("filesystem", "redis", "none")
            persistence_config: Backend-specific configuration dict
            enable_cache: Enable in-memory LRU cache
            cache_max_size: Maximum cache entries
            cache_ttl: Cache TTL in seconds
            **kwargs: Additional parameters for forward compatibility
        """
        self.database_connect = database_connect
        self.template = template
        self.mem_prompt = mem_prompt
        self.conscious_ingest = conscious_ingest
        self.auto_ingest = auto_ingest
        self.user_id = (
            user_id or namespace or "default"
        )  # Support both params for backward compat
        self.shared_memory = shared_memory
        self.memory_filters = memory_filters or []
        self.verbose = verbose
        self.provider_config = provider_config

        # Store additional configuration
        self.openai_api_key = openai_api_key
        self.api_key = api_key
        self.api_type = api_type
        self.base_url = base_url
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.azure_ad_token = azure_ad_token
        self.organization = organization
        self.kwargs = kwargs

        self._session_id = str(uuid.uuid4())
        self._enabled = False

        # LiteLLM native callback manager
        self.litellm_callback_manager = None
        
        # Initialize persistence backend
        self._persistence_backend: Optional[MemoryPersistenceBackend] = None
        self._init_persistence_backend(persistence_backend, persistence_config or {})
        
        # Initialize in-memory cache
        self._memory_cache: Optional[MemoryCache] = None
        if enable_cache:
            self._memory_cache = MemoryCache(max_size=cache_max_size, ttl=cache_ttl)
            logger.debug(f"Memory cache enabled (max_size={cache_max_size}, ttl={cache_ttl})")

        logger.info(f"MemoryManager initialized with session: {self._session_id}")
    
    def _init_persistence_backend(
        self,
        backend_type: str,
        config: dict[str, Any]
    ) -> None:
        """
        Initialize the persistence backend based on type.
        
        Args:
            backend_type: "filesystem", "redis", or "none"
            config: Backend-specific configuration
        """
        if backend_type == "none":
            self._persistence_backend = None
            logger.debug("Memory persistence disabled")
            return
        
        if backend_type == "redis":
            if not REDIS_AVAILABLE:
                logger.warning("Redis requested but aioredis not available, falling back to filesystem")
                backend_type = "filesystem"
            else:
                try:
                    redis_url = config.get("redis_url", os.environ.get("REDIS_URL", "redis://localhost:6379"))
                    key_prefix = config.get("key_prefix", "memori:snapshot:")
                    ttl = config.get("ttl", 86400 * 30)
                    self._persistence_backend = RedisPersistence(
                        redis_url=redis_url,
                        key_prefix=key_prefix,
                        ttl=ttl
                    )
                    logger.info(f"Redis persistence backend initialized")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize Redis backend: {e}, falling back to filesystem")
                    backend_type = "filesystem"
        
        if backend_type == "filesystem":
            storage_dir = config.get("storage_dir", "./memory_snapshots")
            self._persistence_backend = FileSystemPersistence(storage_dir=storage_dir)
            logger.info(f"Filesystem persistence backend initialized at {storage_dir}")

    def save_memory_snapshot(self, data: Optional[dict[str, Any]] = None) -> bool:
        """
        Save a snapshot of current memory state for the user.
        
        Args:
            data: Optional explicit data to save. If None, collects current state.
        
        Returns:
            True if successful, False otherwise
        """
        if self._persistence_backend is None:
            logger.warning("No persistence backend configured, snapshot not saved")
            return False
        
        snapshot_data = data or self._collect_memory_state()
        
        # Update cache if enabled
        if self._memory_cache:
            self._memory_cache.set(f"snapshot:{self.user_id}", snapshot_data)
        
        return self._persistence_backend.save_snapshot(self.user_id, snapshot_data)
    
    def load_memory_snapshot(self) -> Optional[dict[str, Any]]:
        """
        Load a previously saved memory snapshot for the user.
        
        Returns:
            Snapshot data dict or None if not found
        """
        # Check cache first
        if self._memory_cache:
            cached = self._memory_cache.get(f"snapshot:{self.user_id}")
            if cached is not None:
                logger.debug(f"Memory snapshot loaded from cache for user {self.user_id}")
                return cached
        
        if self._persistence_backend is None:
            logger.warning("No persistence backend configured, cannot load snapshot")
            return None
        
        data = self._persistence_backend.load_snapshot(self.user_id)
        
        # Populate cache if enabled
        if data and self._memory_cache:
            self._memory_cache.set(f"snapshot:{self.user_id}", data)
        
        return data
    
    def restore_memory_from_snapshot(self, snapshot: Optional[dict[str, Any]] = None) -> bool:
        """
        Restore memory state from a snapshot.
        
        Args:
            snapshot: Explicit snapshot to restore. If None, loads from backend.
        
        Returns:
            True if successful, False otherwise
        """
        if snapshot is None:
            snapshot = self.load_memory_snapshot()
        
        if snapshot is None:
            logger.warning("No snapshot available to restore")
            return False
        
        return self._apply_memory_state(snapshot)
    
    def delete_memory_snapshot(self) -> bool:
        """
        Delete the memory snapshot for the current user.
        
        Returns:
            True if successful, False otherwise
        """
        # Clear from cache
        if self._memory_cache:
            self._memory_cache.delete(f"snapshot:{self.user_id}")
        
        if self._persistence_backend is None:
            return False
        
        return self._persistence_backend.delete_snapshot(self.user_id)
    
    def list_available_snapshots(self) -> list[str]:
        """
        List all available memory snapshots.
        
        Returns:
            List of user IDs with available snapshots
        """
        if self._persistence_backend is None:
            return []
        
        return self._persistence_backend.list_snapshots()
    
    def _collect_memory_state(self) -> dict[str, Any]:
        """
        Collect the current memory state for snapshotting.
        
        Returns:
            Dict containing serializable memory state
        """
        state = {
            "user_id": self.user_id,
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "enabled": self._enabled,
            "config": {
                "database_connect": self.database_connect,
                "template": self.template,
                "conscious_ingest": self.conscious_ingest,
                "auto_ingest": self.auto_ingest,
                "shared_memory": self.shared_memory,
                "memory_filters": self.memory_filters,
            }
        }
        
        # Add memori instance state if available
        if hasattr(self, "memori_instance") and self.memori_instance:
            try:
                # Collect recent conversation history
                if hasattr(self.memori_instance, "get_conversation_history"):
                    state["conversation_history"] = self.memori_instance.get_conversation_history(limit=100)
                
                # Collect recent memories if search is available
                if hasattr(self.memori_instance, "search"):
                    recent_memories = self.memori_instance.search("", limit=50)
                    state["recent_memories"] = recent_memories
            except Exception as e:
                logger.error(f"Error collecting memory state: {e}")
        
        return state
    
    def _apply_memory_state(self, state: dict[str, Any]) -> bool:
        """
        Apply a memory state from a snapshot.
        
        Args:
            state: Memory state dict to apply
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Restoring memory state from snapshot dated {state.get('timestamp', 'unknown')}")
            
            # Restore configuration
            config = state.get("config", {})
            if config:
                self.conscious_ingest = config.get("conscious_ingest", self.conscious_ingest)
                self.auto_ingest = config.get("auto_ingest", self.auto_ingest)
                self.shared_memory = config.get("shared_memory", self.shared_memory)
                self.memory_filters = config.get("memory_filters", self.memory_filters)
            
            # Note: Actual memory/conversation restoration would need
            # integration with the Memori instance database layer
            
            logger.info("Memory state restored from snapshot")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply memory state: {e}")
            return False
    
    def get_cache_stats(self) -> Optional[dict[str, Any]]:
        """
        Get memory cache statistics.
        
        Returns:
            Cache stats dict or None if cache disabled
        """
        if self._memory_cache is None:
            return None
        return self._memory_cache.stats()
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        if self._memory_cache:
            self._memory_cache.clear()
            logger.debug("Memory cache cleared")

    def set_memori_instance(self, memori_instance):
        """Set the parent Memori instance for memory management."""
        self.memori_instance = memori_instance

        # Initialize LiteLLM callback manager
        try:
            from ..integrations.litellm_integration import setup_litellm_callbacks

            self.litellm_callback_manager = setup_litellm_callbacks(memori_instance)
            if self.litellm_callback_manager:
                logger.debug("LiteLLM callback manager initialized")
            else:
                logger.debug("Failed to initialize LiteLLM callback manager")
        except ImportError as e:
            logger.debug(f"Could not initialize LiteLLM callback manager: {e}")

        logger.debug("MemoryManager configured with Memori instance")

    def enable(self, interceptors: list[str] | None = None) -> dict[str, Any]:
        """
        Enable memory recording using LiteLLM native callbacks.

        Args:
            interceptors: Legacy parameter (ignored, using LiteLLM callbacks)

        Returns:
            Dict containing enablement results
        """
        if self._enabled:
            return {
                "success": True,
                "message": "Already enabled",
                "enabled_interceptors": ["litellm_native"],
            }

        if interceptors is None:
            interceptors = ["litellm_native"]  # Only LiteLLM native callbacks supported

        try:
            # Enable LiteLLM native callback system
            if (
                self.litellm_callback_manager
                and not self.litellm_callback_manager.is_registered
            ):
                success = self.litellm_callback_manager.register_callbacks()
                if not success:
                    return {
                        "success": False,
                        "message": "Failed to register LiteLLM callbacks",
                    }
            elif not self.litellm_callback_manager:
                logger.debug("No LiteLLM callback manager available")

            self._enabled = True

            logger.info("MemoryManager enabled with LiteLLM native callbacks")

            return {
                "success": True,
                "message": "Enabled LiteLLM native callback system",
                "enabled_interceptors": ["litellm_native"],
            }
        except Exception as e:
            logger.error(f"Failed to enable MemoryManager: {e}")
            return {"success": False, "message": str(e)}

    def disable(self) -> dict[str, Any]:
        """
        Disable memory recording using LiteLLM native callbacks.

        Returns:
            Dict containing disable results
        """
        if not self._enabled:
            return {"success": True, "message": "Already disabled"}

        try:
            # Disable LiteLLM native callback system
            if (
                self.litellm_callback_manager
                and self.litellm_callback_manager.is_registered
            ):
                success = self.litellm_callback_manager.unregister_callbacks()
                if not success:
                    logger.warning("Failed to unregister LiteLLM callbacks")

            self._enabled = False

            logger.info("MemoryManager disabled")

            return {
                "success": True,
                "message": "MemoryManager disabled successfully (LiteLLM native callbacks)",
            }
        except Exception as e:
            logger.error(f"Failed to disable MemoryManager: {e}")
            return {"success": False, "message": str(e)}

    def get_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of memory recording system.

        Returns:
            Dict containing memory system status information
        """
        callback_status = "inactive"
        if self.litellm_callback_manager:
            if self.litellm_callback_manager.is_registered:
                callback_status = "active"
            else:
                callback_status = "available_but_not_registered"
        else:
            callback_status = "unavailable"

        return {
            "litellm_native": {
                "enabled": self._enabled,
                "status": callback_status,
                "method": "litellm_callbacks",
                "session_id": self._session_id,
                "callback_manager": self.litellm_callback_manager is not None,
            }
        }

    def get_health(self) -> dict[str, Any]:
        """
        Get health check of the memory management system.

        Returns:
            Dict containing health information
        """
        # Determine persistence backend type
        persistence_type = "none"
        if isinstance(self._persistence_backend, RedisPersistence):
            persistence_type = "redis"
        elif isinstance(self._persistence_backend, FileSystemPersistence):
            persistence_type = "filesystem"
        
        health_info = {
            "session_id": self._session_id,
            "enabled": self._enabled,
            "user_id": self.user_id,
            "litellm_callback_manager": self.litellm_callback_manager is not None,
            "litellm_callbacks_registered": (
                self.litellm_callback_manager.is_registered
                if self.litellm_callback_manager
                else False
            ),
            "memory_filters": len(self.memory_filters),
            "conscious_ingest": self.conscious_ingest,
            "auto_ingest": self.auto_ingest,
            "database_connect": self.database_connect,
            "template": self.template,
            "persistence": {
                "backend": persistence_type,
                "enabled": self._persistence_backend is not None,
                "available_snapshots": len(self.list_available_snapshots()) if self._persistence_backend else 0,
            },
            "cache": self.get_cache_stats(),
        }
        
        return health_info

    # === BACKWARD COMPATIBILITY PROPERTIES ===

    @property
    def session_id(self) -> str:
        """Get session ID for backward compatibility."""
        return self._session_id

    @property
    def enabled(self) -> bool:
        """Check if enabled for backward compatibility."""
        return self._enabled

    # === PLACEHOLDER METHODS FOR FUTURE MODULAR COMPONENTS ===

    def record_conversation(
        self,
        user_input: str,
        ai_output: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record a conversation (placeholder for future implementation).

        Returns:
            Placeholder conversation ID
        """
        logger.info(f"Recording conversation (placeholder): {user_input[:50]}...")
        return str(uuid.uuid4())

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: list[str] | None = None,
        categories: list[str] | None = None,
        min_importance: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search memories (placeholder for future implementation).

        Returns:
            Empty list (placeholder)
        """
        logger.info(f"Searching memories (placeholder): {query}")
        return []

    def cleanup(self):
        """Cleanup resources including persistence backends and cache."""
        try:
            if self._enabled:
                self.disable()

            # Clean up callback manager
            if self.litellm_callback_manager:
                self.litellm_callback_manager.unregister_callbacks()
                self.litellm_callback_manager = None
            
            # Clean up cache
            if self._memory_cache:
                self._memory_cache.clear()
                self._memory_cache = None
            
            # Clean up Redis persistence if applicable
            if isinstance(self._persistence_backend, RedisPersistence):
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    if not loop.is_running():
                        loop.run_until_complete(self._persistence_backend.close())
                except Exception as e:
                    logger.debug(f"Error closing Redis connection: {e}")
            
            self._persistence_backend = None

            logger.info("MemoryManager cleanup completed")
        except Exception as e:
            logger.error(f"Error during MemoryManager cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            # Use print as logger might be gone
            print(f"Error in MemoryManager cleanup: {e}")
