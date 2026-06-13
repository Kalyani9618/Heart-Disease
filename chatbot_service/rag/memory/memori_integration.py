"""
Memori Integration - Bridge between RAG and existing Memori memory system.

This module integrates the new RAG (Retrieval-Augmented Generation) system
with the existing Memori memory layer, enabling:

1. Automatic sync of Memori memories to RAG vector store
2. Unified search across both systems
3. Intelligent memory enrichment with embeddings
4. Backward compatibility with existing Memori APIs

Addresses GAPs from MEMORI_VS_RAG_ANALYSIS.md:
- Memori has keyword search → RAG adds semantic/embedding search
- Memori has SQLite storage → RAG adds vector storage for similarity
- Combined: Best of both worlds

Integration:
- Uses AbstractMemoriInterface for type-safe contracts
- Validates Memori implementation before use
- Provides graceful error handling and fallbacks
"""


import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    from .memori_interfaces import AbstractMemoriInterface, MemoriValidator
except ImportError:
    # Handle direct import when run as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag.memory.memori_interfaces import AbstractMemoriInterface, MemoriValidator

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Configuration for Memori-RAG synchronization."""

    sync_on_write: bool = True  # Sync to RAG on every new memory
    sync_interval_seconds: int = 300  # Background sync interval
    batch_size: int = 100  # Number of memories to sync per batch
    include_categories: List[str] = field(
        default_factory=lambda: [
            "health",
            "symptoms",
            "medications",
            "conditions",
            "appointments",
            "measurements",
            "lifestyle",
        ]
    )
    exclude_sensitive: bool = True  # Exclude highly sensitive data


class MemoriRAGBridge:
    """
    Bridge between Memori memory system and RAG vector store.

    This class enables:
    - Dual storage: Memori (structured) + RAG (semantic)
    - Unified search with hybrid ranking
    - Automatic synchronization
    - Memory enrichment with embeddings

    Architecture:
        ┌─────────────────────────────────────────────┐
        │             MemoriRAGBridge                 │
        ├─────────────────┬───────────────────────────┤
        │    Memori       │        RAG                │
        │  (Structured)   │     (Semantic)            │
        │                 │                           │
        │  - PostgreSQL   │  - ChromaDB               │
        │  - Categories   │  - Embeddings             │
        │  - Sessions     │  - Similarity             │
        │  - Timestamps   │  - Vector Search          │
        └────────┬────────┴─────────────┬─────────────┘
                 │                      │
                 └──────────┬───────────┘
                            │
                   ┌────────▼────────┐
                   │  Unified Search │
                   └─────────────────┘

    Example:
        from memori import Memori
        from rag import VectorStore

        memori = Memori()  # Uses PostgreSQL from AppConfig
        vector_store = VectorStore()

        bridge = MemoriRAGBridge(memori, vector_store)

        # Add memory (syncs to both)
        bridge.add_memory("user123", "Patient has history of high blood pressure")

        # Unified search
        results = await bridge.search("blood pressure history", user_id="user123")
    """

    def __init__(
        self,
        memori: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        config: Optional[SyncConfig] = None,
    ):
        """
        Initialize Memori-RAG Bridge.

        Args:
            memori: Memori instance implementing AbstractMemoriInterface (optional)
            vector_store: VectorStore instance (optional, can be set later)
            config: Sync configuration

        Raises:
            TypeError: If memori is provided but doesn't implement required interface
        """
        # Validate memori if provided
        if memori is not None:
            try:
                MemoriValidator.validate(memori)
                logger.info("✅ Memori instance validated successfully")
            except (TypeError, AttributeError) as e:
                logger.error(f"❌ Memori validation failed: {e}")
                raise ValueError(f"Invalid Memori instance: {e}") from e

        self.memori = memori
        self.vector_store = vector_store
        self.config = config or SyncConfig()

        self._sync_task: Optional[asyncio.Task] = None
        self._sync_lock = asyncio.Lock()
        self._last_sync_time: Optional[datetime] = None
        self._sync_stats = {
            "total_synced": 0,
            "last_batch_count": 0,
            "errors": 0,
        }

        logger.info("✅ MemoriRAGBridge initialized")

    def set_memori(self, memori: Any) -> None:
        """Set or update Memori instance."""
        self.memori = memori
        logger.info("Memori instance connected to bridge")

    def set_vector_store(self, vector_store: Any) -> None:
        """Set or update VectorStore instance."""
        self.vector_store = vector_store
        logger.info("VectorStore instance connected to bridge")

    # =========================================================================
    # MEMORY OPERATIONS (Write)
    # =========================================================================

    async def add_memory(
        self,
        user_id: str,
        content: str,
        category: str = "general",
        metadata: Optional[Dict] = None,
        sync_to_rag: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a memory through the bridge (stores in both systems concurrently).

        Stores to Memori and RAG in parallel using asyncio.gather for
        simultaneous data ingestion from the AI pipeline.

        Args:
            user_id: User identifier
            content: Memory content
            category: Memory category
            metadata: Additional metadata
            sync_to_rag: Whether to sync to RAG immediately

        Returns:
            Dict with memory_id and status
        """
        result = {"memori_id": None, "rag_id": None, "status": "pending"}

        async def _store_in_memori():
            """Store in Memori (primary structured storage)."""
            if not self.memori:
                return None
            try:
                memori_result = self.memori.record_conversation(
                    user_input=content,
                    ai_output="Memory recorded",
                    metadata={
                        "category": category,
                        "user_id": user_id,
                        **(metadata or {}),
                    },
                )
                return memori_result.get("memory_id")
            except Exception as e:
                logger.error(f"Failed to store in Memori: {e}")
                return None

        async def _store_in_rag(memori_id=None):
            """Store in RAG (semantic vector storage)."""
            if not self.vector_store or not sync_to_rag:
                return None
            try:
                rag_id = self.vector_store.add_user_memory(
                    user_id=user_id,
                    memory_id=memori_id or f"mem_{datetime.now().timestamp()}",
                    content=content,
                    metadata={
                        "category": category,
                        "timestamp": datetime.now().isoformat(),
                        "source": "memori_bridge",
                        **(metadata or {}),
                    },
                )
                return rag_id
            except Exception as e:
                logger.error(f"Failed to store in RAG: {e}")
                return None

        # Execute both stores concurrently
        tasks = []
        if self.memori:
            tasks.append(_store_in_memori())
        if self.vector_store and sync_to_rag:
            tasks.append(_store_in_rag())

        if tasks:
            store_results = await asyncio.gather(*tasks, return_exceptions=True)

            idx = 0
            if self.memori:
                r = store_results[idx]
                if isinstance(r, Exception):
                    result["memori_error"] = str(r)
                else:
                    result["memori_id"] = r
                idx += 1

            if self.vector_store and sync_to_rag:
                r = store_results[idx]
                if isinstance(r, Exception):
                    result["rag_error"] = str(r)
                else:
                    result["rag_id"] = r

        result["status"] = (
            "success" if result.get("memori_id") or result.get("rag_id") else "failed"
        )
        return result

    # =========================================================================
    # SEARCH OPERATIONS (Read)
    # =========================================================================

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
        use_memori: bool = True,
        use_rag: bool = True,
        hybrid_weight: float = 0.5,  # 0 = all Memori, 1 = all RAG
    ) -> List[Dict]:
        """
        Unified search across Memori and RAG.

        Performs hybrid search combining:
        - Memori: Keyword/metadata search (structured)
        - RAG: Semantic/embedding search (similarity)

        Args:
            query: Search query
            user_id: Optional user filter
            top_k: Number of results
            use_memori: Include Memori results
            use_rag: Include RAG results
            hybrid_weight: Weight for RAG vs Memori (0-1)

        Returns:
            List of search results with scores
        """
        results = []

        # Run both searches in parallel
        tasks = []

        if use_memori and self.memori:
            tasks.append(self._search_memori(query, user_id, top_k))

        if use_rag and self.vector_store:
            tasks.append(self._search_rag(query, user_id, top_k))

        if not tasks:
            return []

        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        memori_results = []
        rag_results = []

        idx = 0
        if use_memori and self.memori:
            if not isinstance(search_results[idx], Exception):
                memori_results = search_results[idx]
            idx += 1

        if use_rag and self.vector_store:
            if not isinstance(search_results[idx], Exception):
                rag_results = search_results[idx]

        # Hybrid ranking
        results = self._hybrid_rank(
            memori_results,
            rag_results,
            hybrid_weight,
            top_k,
        )

        return results

    async def _search_memori(
        self,
        query: str,
        user_id: Optional[str],
        top_k: int,
    ) -> List[Dict]:
        """Search Memori system."""
        try:
            # Use Memori's built-in search
            results = self.memori.search(query, limit=top_k)

            # Normalize results with proper scoring (avoids negative scores)
            normalized = []
            total = len(results) if results else 1
            for i, r in enumerate(results):
                # Use reciprocal rank scoring: 1/(rank+1), always positive
                score = 1.0 / (i + 1)
                normalized.append(
                    {
                        "content": r.get("content", r.get("message", "")),
                        "metadata": r.get("metadata", {}),
                        "score": score,
                        "source": "memori",
                        "id": r.get("id", r.get("memory_id")),
                    }
                )

            return normalized
        except Exception as e:
            logger.error(f"Memori search failed: {e}")
            return []

    async def _search_rag(
        self,
        query: str,
        user_id: Optional[str],
        top_k: int,
    ) -> List[Dict]:
        """Search RAG vector store."""
        try:
            if user_id:
                results = self.vector_store.search_user_memories(
                    user_id, query, top_k=top_k
                )
            else:
                results = self.vector_store.search("memories", query, top_k=top_k)

            # Already in normalized format from VectorStore
            for r in results:
                r["source"] = "rag"

            return results
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    def _hybrid_rank(
        self,
        memori_results: List[Dict],
        rag_results: List[Dict],
        rag_weight: float,
        top_k: int,
    ) -> List[Dict]:
        """
        Hybrid ranking combining Memori and RAG results.

        Uses Reciprocal Rank Fusion (RRF) algorithm.
        """
        # RRF constant
        k = 60

        # Calculate RRF scores
        scores = {}

        # Memori results (weight: 1 - rag_weight)
        memori_weight = 1.0 - rag_weight
        for rank, result in enumerate(memori_results):
            content_key = result.get("content", "")[
                :100
            ]  # Use truncated content as key
            rrf_score = memori_weight / (k + rank + 1)

            if content_key in scores:
                scores[content_key]["score"] += rrf_score
                scores[content_key]["sources"].append("memori")
            else:
                scores[content_key] = {
                    "result": result,
                    "score": rrf_score,
                    "sources": ["memori"],
                }

        # RAG results (weight: rag_weight)
        for rank, result in enumerate(rag_results):
            content_key = result.get("content", "")[:100]
            rrf_score = rag_weight / (k + rank + 1)

            if content_key in scores:
                scores[content_key]["score"] += rrf_score
                scores[content_key]["sources"].append("rag")
            else:
                scores[content_key] = {
                    "result": result,
                    "score": rrf_score,
                    "sources": ["rag"],
                }

        # Sort by combined score
        ranked = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        # Format results
        results = []
        for item in ranked[:top_k]:
            result = item["result"].copy()
            result["hybrid_score"] = item["score"]
            result["found_in"] = item["sources"]
            results.append(result)

        return results

    # =========================================================================
    # SYNCHRONIZATION
    # =========================================================================

    async def sync_memori_to_rag(
        self,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Sync memories from Memori to RAG vector store.

        Args:
            user_id: Optional user filter
            since: Only sync memories since this time

        Returns:
            Sync statistics
        """
        async with self._sync_lock:
            if not self.memori or not self.vector_store:
                return {"error": "Both Memori and VectorStore required"}

            stats = {"synced": 0, "errors": 0, "skipped": 0}

            try:
                # Get memories from Memori
                # Note: This uses Memori's internal API
                memories = self._get_memori_memories(user_id, since)

                # Batch sync to RAG
                batch = []
                for memory in memories:
                    try:
                        # Check if category should be synced
                        category = memory.get("metadata", {}).get("category", "general")
                        if (
                            category not in self.config.include_categories
                            and self.config.include_categories
                        ):
                            stats["skipped"] += 1
                            continue

                        batch.append(
                            {
                                "id": memory.get("id", f"mem_{len(batch)}"),
                                "content": memory.get("content", ""),
                                "metadata": {
                                    **memory.get("metadata", {}),
                                    "synced_from_memori": True,
                                    "sync_time": datetime.now().isoformat(),
                                },
                                "user_id": memory.get("user_id", user_id or "default"),
                            }
                        )

                        if len(batch) >= self.config.batch_size:
                            self._sync_batch(batch)
                            stats["synced"] += len(batch)
                            batch = []

                    except Exception as e:
                        logger.error(f"Error syncing memory: {e}")
                        stats["errors"] += 1

                # Sync remaining batch
                if batch:
                    self._sync_batch(batch)
                    stats["synced"] += len(batch)

                self._last_sync_time = datetime.now()
                self._sync_stats["total_synced"] += stats["synced"]
                self._sync_stats["last_batch_count"] = stats["synced"]

            except Exception as e:
                logger.error(f"Sync failed: {e}")
                stats["error"] = str(e)
                self._sync_stats["errors"] += 1

            return stats

    def _get_memori_memories(
        self,
        user_id: Optional[str],
        since: Optional[datetime],
    ) -> List[Dict]:
        """
        Get memories from Memori for syncing.

        Args:
            user_id: User identifier (optional)
            since: Get memories since this datetime (optional)

        Returns:
            List of memory dictionaries

        Raises:
            RuntimeError: If Memori is not initialized
        """
        if not self.memori:
            logger.warning("Memori not initialized, cannot retrieve memories")
            return []

        try:
            # Validate that Memori has the required method
            MemoriValidator.validate_method(self.memori, "get_all_memories")

            # Use primary method
            memories = self.memori.get_all_memories(
                user_id=user_id,
                limit=self.config.batch_size
            )

            logger.debug(f"Retrieved {len(memories)} memories from Memori")
            return memories

        except AttributeError:
            logger.info(
                "Memori doesn't have get_all_memories, "
                "trying fallback search method..."
            )

            # Fallback: try search method
            try:
                MemoriValidator.validate_method(self.memori, "search")
                memories = self.memori.search("", limit=self.config.batch_size)
                logger.debug(f"Retrieved {len(memories)} memories via fallback search")
                return memories

            except AttributeError as e:
                logger.error(
                    f"❌ Memori doesn't support required retrieval methods: {e}. "
                    f"Must implement get_all_memories() or search()"
                )
                return []

        except Exception as e:
            logger.error(f"Failed to get memories from Memori: {e}")
            self._sync_stats["errors"] += 1
            return []

    def _sync_batch(self, batch: List[Dict]) -> None:
        """
        Sync a batch of memories to RAG with parallel processing.
        
        Uses asyncio.gather for concurrent vector store writes when possible,
        falling back to sequential writes.
        """
        if not batch:
            return

        async def _async_sync():
            """Concurrent batch sync."""
            tasks = []
            for memory in batch:
                tasks.append(self._sync_single_memory(memory))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to sync memory {batch[i].get('id', i)}: {result}")

        async def _sync_single_memory(memory: Dict) -> None:
            """Sync a single memory, wrapped for async gather."""
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self.vector_store.add_user_memory,
                memory["user_id"],
                memory["id"],
                memory["content"],
                memory["metadata"],
            )

        # Try async batch first
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use create_task
            loop.create_task(_async_sync())
        except RuntimeError:
            # No event loop, fall back to sequential
            for memory in batch:
                try:
                    self.vector_store.add_user_memory(
                        user_id=memory["user_id"],
                        memory_id=memory["id"],
                        content=memory["content"],
                        metadata=memory["metadata"],
                    )
                except Exception as e:
                    logger.error(f"Failed to sync memory {memory['id']}: {e}")

    async def start_background_sync(self) -> None:
        """Start background synchronization task."""
        if self._sync_task and not self._sync_task.done():
            logger.warning("Background sync already running")
            return

        self._sync_task = asyncio.create_task(self._background_sync_loop())
        logger.info("Background sync started")

    async def stop_background_sync(self) -> None:
        """Stop background synchronization task."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            logger.info("Background sync stopped")

    async def _background_sync_loop(self) -> None:
        """Background sync loop."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)

                logger.debug("Running background sync...")
                await self.sync_memori_to_rag(since=self._last_sync_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background sync error: {e}")

    # =========================================================================
    # CONTEXT RETRIEVAL (for RAG Pipeline)
    # =========================================================================

    async def get_context_for_query(
        self,
        query: str,
        user_id: str,
        max_memories: int = 5,
        include_metadata: bool = True,
    ) -> str:
        """
        Get formatted context for RAG pipeline.

        Returns a formatted string suitable for LLM prompt augmentation.
        """
        results = await self.search(
            query,
            user_id=user_id,
            top_k=max_memories,
        )

        if not results:
            return ""

        context_parts = ["**Relevant Patient History:**"]

        for i, result in enumerate(results, 1):
            content = result.get("content", "")[:300]

            if include_metadata:
                metadata = result.get("metadata", {})
                category = metadata.get("category", "general")
                timestamp = metadata.get("timestamp", "")
                context_parts.append(f"{i}. [{category}] {content}")
                if timestamp:
                    context_parts.append(f"   (Recorded: {timestamp})")
            else:
                context_parts.append(f"{i}. {content}")

        return "\n".join(context_parts)

    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            **self._sync_stats,
            "last_sync_time": (
                self._last_sync_time.isoformat() if self._last_sync_time else None
            ),
            "sync_running": self._sync_task is not None and not self._sync_task.done(),
            "config": {
                "sync_on_write": self.config.sync_on_write,
                "sync_interval_seconds": self.config.sync_interval_seconds,
                "batch_size": self.config.batch_size,
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "memori_connected": self.memori is not None,
            "vector_store_connected": self.vector_store is not None,
            "sync_stats": self.get_sync_stats(),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_memori_rag_bridge(
    memori: Any = None,
    vector_store: Any = None,
    persist_directory: str = None,
) -> MemoriRAGBridge:
    """
    Factory function to create MemoriRAGBridge.

    Args:
        memori: Existing Memori instance
        vector_store: Existing VectorStore instance
        persist_directory: Directory for vector store if creating new

    Returns:
        Configured MemoriRAGBridge instance
    """
    # Create VectorStore if not provided
    if vector_store is None and persist_directory:
        from rag.store.vector_store import InMemoryVectorStore as VectorStore

        vector_store = VectorStore(persist_directory=persist_directory)

    return MemoriRAGBridge(
        memori=memori,
        vector_store=vector_store,
    )




# =============================================================================
# TESTING NOTE: Tests moved to tests/test_memori_integration.py
# =============================================================================
# Production code should not contain test blocks. All tests are in tests/

