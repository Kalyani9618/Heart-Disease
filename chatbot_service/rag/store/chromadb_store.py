"""
ChromaDB-based Vector Store for HeartGuard AI

This module provides ChromaDB-based vector storage for semantic search
over medical knowledge and user memories.

Features:
- Persistent local ChromaDB storage
- Multi-collection support (medical, drugs, symptoms, memories)
- Multi-tenant support with user_id isolation
- Async and sync query support
- LRU + Redis caching for repeated queries

Performance:
- L1: In-memory LRU cache (100 entries)
- L2: Redis cache with 300s TTL
- Target: <300ms p95 latency for vector search

Collections:
1. medical_knowledge - Medical guidelines/protocols
2. drug_interactions - Medication information
3. symptoms_conditions - Symptom-to-condition mapping
4. user_memories - Per-user memory storage
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Redis cache configuration
VECTOR_CACHE_TTL = int(os.getenv("VECTOR_CACHE_TTL", "300"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Lazy Redis client
_vector_redis_client = None
_vector_redis_available = None


def _get_sync_redis_client():
    """Get synchronous Redis client for vector store caching."""
    global _vector_redis_client, _vector_redis_available

    if _vector_redis_available is False:
        return None

    if _vector_redis_client is not None:
        return _vector_redis_client

    try:
        import redis
        _vector_redis_client = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2.0,
        )
        _vector_redis_client.ping()
        _vector_redis_available = True
        logger.info(f"✅ Vector cache Redis connected (TTL={VECTOR_CACHE_TTL}s)")
        return _vector_redis_client
    except Exception as e:
        logger.warning(f"⚠️ Redis unavailable for vector caching: {e}")
        _vector_redis_available = False
        return None


# Import embedding service (remote-only for inference mode)
try:
    from rag.embedding.remote import RemoteEmbeddingService
except ImportError:
    RemoteEmbeddingService = None  # type: ignore[assignment,misc]


class ChromaDBVectorStore:
    """
    ChromaDB-based vector store for healthcare RAG.

    Uses local persistent ChromaDB for vector storage and semantic search.

    Features:
    - Persistent storage with local SQLite backend
    - HNSW indexing for fast approximate nearest neighbor search
    - Multi-tenant user isolation via user_id metadata
    - Hybrid queries (vector + metadata filters)

    Collections:
    - medical_knowledge: Medical guidelines, protocols
    - drug_interactions: Medication information
    - symptoms_conditions: Symptom-condition mapping
    - user_memories: Per-user memory storage

    Example:
        store = ChromaDBVectorStore()

        # Store medical document
        await store.add_medical_document(
            doc_id="aha_guidelines_2024",
            content="Heart failure treatment guidelines...",
            metadata={"source": "AHA", "year": 2024}
        )

        # Search
        results = await store.search_medical_knowledge(
            "heart failure treatment",
            top_k=5
        )
    """

    # Collection names
    MEDICAL_COLLECTION = "medical_knowledge"
    DRUG_COLLECTION = "drug_interactions"
    SYMPTOMS_COLLECTION = "symptoms_conditions"
    MEMORIES_COLLECTION = "user_memories"

    # Embedding dimension (MedCPT 768-dim via remote Colab)
    EMBEDDING_DIMENSION = 768

    def __init__(
        self,
        persist_directory: str = None,
        embedding_model: str = "MedCPT-Query-Encoder",
        config=None,  # Accept but ignore config for compatibility
        **kwargs,  # Accept any extra kwargs for flexibility
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Path to ChromaDB persistence directory.
                             If None, uses CHROMADB_DIR env or ./Chromadb
            embedding_model: Model name for embeddings (default: MedCPT-Query-Encoder)
            config: Optional AppConfig (accepted for compatibility, not used)
            **kwargs: Additional arguments (ignored)
        """
        import chromadb

        # Determine persistence directory
        self.persist_directory = persist_directory or os.getenv(
            "CHROMADB_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Chromadb")
        )

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(path=self.persist_directory)

        # Initialize embedding service (optional — rag_engines.py provides its own)
        self.embedding_service = None
        use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
        if use_remote and RemoteEmbeddingService is not None:
            try:
                self.embedding_service = RemoteEmbeddingService.get_instance()
                logger.info("✅ Using RemoteEmbeddingService (MedCPT 768-dim)")
            except Exception as e:
                logger.warning(f"Remote embedding unavailable: {e}")
        elif not use_remote:
            logger.info("ℹ️ Remote embeddings disabled (USE_REMOTE_EMBEDDINGS=false)")

        # Query result cache
        self._query_cache: OrderedDict[str, List[Dict]] = OrderedDict()
        self._query_cache_max_size = 100
        self._query_cache_lock = threading.Lock()

        # Pre-create collections
        self._collections: Dict[str, Any] = {}
        for name in [
            self.MEDICAL_COLLECTION,
            self.DRUG_COLLECTION,
            self.SYMPTOMS_COLLECTION,
            self.MEMORIES_COLLECTION,
        ]:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info(
            f"✅ ChromaDBVectorStore initialized (persist_dir={self.persist_directory})"
        )

    def _get_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # =========================================================================
    # CACHING
    # =========================================================================

    def _cache_key(self, query: str, collection: str, top_k: int, filters: dict = None) -> str:
        """Generate cache key for query."""
        key_data = f"{query}:{collection}:{top_k}:{json.dumps(filters or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Check L1 (memory) and L2 (Redis) cache."""
        # L1: Memory cache
        with self._query_cache_lock:
            if cache_key in self._query_cache:
                logger.debug(f"⚡ L1 cache HIT: {cache_key[:8]}")
                return self._query_cache[cache_key]

        # L2: Redis cache
        redis_client = _get_sync_redis_client()
        if redis_client:
            try:
                cached = redis_client.get(f"cvec:{cache_key}")
                if cached:
                    results = json.loads(cached)
                    logger.debug(f"⚡ L2 cache HIT: {cache_key[:8]}")
                    # Promote to L1
                    with self._query_cache_lock:
                        if len(self._query_cache) >= self._query_cache_max_size:
                            self._query_cache.popitem(last=False)
                        self._query_cache[cache_key] = results
                    return results
            except Exception as e:
                logger.debug(f"Redis cache get failed: {e}")

        return None

    def _update_cache(self, cache_key: str, results: List[Dict]):
        """Update L1 and L2 cache."""
        # L1: Memory cache
        with self._query_cache_lock:
            if len(self._query_cache) >= self._query_cache_max_size:
                self._query_cache.popitem(last=False)
            self._query_cache[cache_key] = results

        # L2: Redis cache
        redis_client = _get_sync_redis_client()
        if redis_client:
            try:
                redis_client.setex(
                    f"cvec:{cache_key}",
                    VECTOR_CACHE_TTL,
                    json.dumps(results),
                )
            except Exception as e:
                logger.debug(f"Redis cache set failed: {e}")

    # =========================================================================
    # MEDICAL KNOWLEDGE BASE
    # =========================================================================

    def add_medical_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Add a medical document to the knowledge base.

        Args:
            doc_id: Unique document ID
            content: Document text content
            metadata: Additional metadata
            embedding: Pre-computed embedding (optional)

        Returns:
            Document ID
        """
        collection = self._get_collection(self.MEDICAL_COLLECTION)

        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embedding_service.embed_text(content)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

        # Prepare metadata
        meta = metadata or {}
        meta["added_at"] = datetime.now().isoformat()
        meta["content_length"] = len(content)
        # ChromaDB metadata values must be str, int, float, or bool
        meta = self._sanitize_metadata(meta)

        collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
        )

        logger.debug(f"Added medical document: {doc_id}")
        return doc_id

    async def add_medical_document_async(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Async version of add_medical_document."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.add_medical_document, doc_id, content, metadata
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        limit: int = None,
        collection_name: str = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Generic search method for vector store compatibility.
        
        Delegates to the appropriate collection-specific search method.
        """
        k = limit or top_k
        if collection_name == "drug_interactions":
            return self.search_drug_interactions(query, k)
        elif collection_name == "symptoms_conditions":
            return self.search_symptoms(query, k)
        elif collection_name == "user_memories":
            user_id = kwargs.get("user_id", "default")
            return self.search_user_memories(query, user_id, k)
        else:
            return self.search_medical_knowledge(query, k)

    def search_medical_knowledge(
        self,
        query: str = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Search medical knowledge base using vector similarity.

        Args:
            query: Search query text (optional if query_embedding provided)
            top_k: Number of results to return
            filter_metadata: Metadata filters (applied as ChromaDB where clause)
            query_embedding: Pre-computed embedding vector

        Returns:
            List of matching documents with scores
        """
        # Check cache
        if query:
            cache_key = self._cache_key(query, self.MEDICAL_COLLECTION, top_k, filter_metadata)
            cached = self._check_cache(cache_key)
            if cached:
                return cached

        # Generate embedding if needed
        if query_embedding is None:
            if query:
                query_embedding = self.embedding_service.embed_text(query)
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()
            else:
                raise ValueError("Either query or query_embedding required")

        collection = self._get_collection(self.MEDICAL_COLLECTION)

        # Build query kwargs
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            where = {}
            for key, value in filter_metadata.items():
                where[key] = str(value)
            query_kwargs["where"] = where

        raw = collection.query(**query_kwargs)

        # Format results (ChromaDB returns lists-of-lists)
        results = []
        if raw and raw["ids"] and raw["ids"][0]:
            for i, doc_id in enumerate(raw["ids"][0]):
                # ChromaDB returns cosine distance; similarity = 1 - distance
                distance = raw["distances"][0][i] if raw["distances"] else 0.0
                results.append({
                    "id": doc_id,
                    "content": raw["documents"][0][i] if raw["documents"] else "",
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                    "score": 1.0 - distance,
                })

        # Update cache
        if query:
            self._update_cache(cache_key, results)

        return results

    async def search_medical_knowledge_async(
        self,
        query: str = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Async version of search_medical_knowledge."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search_medical_knowledge(query, top_k, filter_metadata, query_embedding),
        )

    # Alias for compatibility
    async def async_search(
        self,
        query: str,
        collection_name: str = None,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """Async search wrapper for compatibility."""
        if collection_name == "drug_interactions":
            return await self.search_drug_interactions_async(query, top_k)
        elif collection_name == "symptoms_conditions":
            return await self.search_symptoms_async(query, top_k)
        elif collection_name == "user_memories":
            user_id = kwargs.get("user_id", "default")
            return await self.search_user_memories_async(query, user_id, top_k)
        else:
            return await self.search_medical_knowledge_async(query, top_k)

    # =========================================================================
    # DRUG INTERACTIONS
    # =========================================================================

    def add_drug_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Add a drug interaction document."""
        collection = self._get_collection(self.DRUG_COLLECTION)

        embedding = self.embedding_service.embed_text(content)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        meta = metadata or {}
        meta["added_at"] = datetime.now().isoformat()
        meta = self._sanitize_metadata(meta)

        collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
        )

        return doc_id

    def search_drug_interactions(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """Search drug interactions."""
        cache_key = self._cache_key(query, self.DRUG_COLLECTION, top_k)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        embedding = self.embedding_service.embed_text(query)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        collection = self._get_collection(self.DRUG_COLLECTION)

        raw = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        if raw and raw["ids"] and raw["ids"][0]:
            for i, doc_id in enumerate(raw["ids"][0]):
                distance = raw["distances"][0][i] if raw["distances"] else 0.0
                results.append({
                    "id": doc_id,
                    "content": raw["documents"][0][i] if raw["documents"] else "",
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                    "score": 1.0 - distance,
                })

        self._update_cache(cache_key, results)
        return results

    async def search_drug_interactions_async(self, query: str, top_k: int = 5) -> List[Dict]:
        """Async version of search_drug_interactions."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_drug_interactions, query, top_k)

    # =========================================================================
    # SYMPTOMS CONDITIONS
    # =========================================================================

    def search_symptoms(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search symptoms and conditions."""
        cache_key = self._cache_key(query, self.SYMPTOMS_COLLECTION, top_k)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        embedding = self.embedding_service.embed_text(query)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        collection = self._get_collection(self.SYMPTOMS_COLLECTION)

        raw = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        if raw and raw["ids"] and raw["ids"][0]:
            for i, doc_id in enumerate(raw["ids"][0]):
                distance = raw["distances"][0][i] if raw["distances"] else 0.0
                results.append({
                    "id": doc_id,
                    "content": raw["documents"][0][i] if raw["documents"] else "",
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                    "score": 1.0 - distance,
                })

        self._update_cache(cache_key, results)
        return results

    async def search_symptoms_async(self, query: str, top_k: int = 5) -> List[Dict]:
        """Async version of search_symptoms."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_symptoms, query, top_k)

    # =========================================================================
    # USER MEMORIES (Multi-tenant)
    # =========================================================================

    def add_user_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str = "general",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Add a user-specific memory with isolation.

        Args:
            memory_id: Unique memory ID
            user_id: User ID for multi-tenant isolation
            content: Memory content
            memory_type: Type of memory (general, preference, context, etc.)
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        collection = self._get_collection(self.MEMORIES_COLLECTION)

        embedding = self.embedding_service.embed_text(content)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        meta = metadata or {}
        meta["user_id"] = user_id
        meta["memory_type"] = memory_type
        meta["added_at"] = datetime.now().isoformat()
        meta = self._sanitize_metadata(meta)

        collection.upsert(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
        )

        logger.debug(f"Added user memory: {memory_id} for user: {user_id}")
        return memory_id

    def search_user_memories(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search user memories with multi-tenant isolation.

        Args:
            query: Search query
            user_id: User ID for isolation (REQUIRED)
            top_k: Number of results
            memory_type: Optional filter by memory type

        Returns:
            List of matching memories
        """
        cache_key = self._cache_key(
            f"{query}:{user_id}",
            self.MEMORIES_COLLECTION,
            top_k,
            {"memory_type": memory_type} if memory_type else None,
        )
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        embedding = self.embedding_service.embed_text(query)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        collection = self._get_collection(self.MEMORIES_COLLECTION)

        # Build where clause for user isolation
        where: Dict[str, Any] = {"user_id": user_id}
        if memory_type:
            where = {"$and": [{"user_id": user_id}, {"memory_type": memory_type}]}

        raw = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        if raw and raw["ids"] and raw["ids"][0]:
            for i, doc_id in enumerate(raw["ids"][0]):
                distance = raw["distances"][0][i] if raw["distances"] else 0.0
                meta = raw["metadatas"][0][i] if raw["metadatas"] else {}
                results.append({
                    "id": doc_id,
                    "content": raw["documents"][0][i] if raw["documents"] else "",
                    "memory_type": meta.get("memory_type", "general"),
                    "metadata": meta,
                    "score": 1.0 - distance,
                })

        self._update_cache(cache_key, results)
        return results

    async def search_user_memories_async(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict]:
        """Async version of search_user_memories."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search_user_memories(query, user_id, top_k, memory_type),
        )

    def delete_user_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a user memory (with ownership check)."""
        collection = self._get_collection(self.MEMORIES_COLLECTION)

        try:
            # Check ownership first
            result = collection.get(ids=[memory_id], include=["metadatas"])
            if result and result["ids"] and result["metadatas"]:
                meta = result["metadatas"][0]
                if meta.get("user_id") != user_id:
                    logger.warning(f"Ownership mismatch for memory {memory_id}")
                    return False

            collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")
            return False

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def add_documents_batch(
        self,
        documents: List[Dict],
        collection_name: str = None,
        batch_size: int = 100,
    ) -> Dict[str, int]:
        """
        Batch insert documents.

        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
            collection_name: Target collection (default: medical_knowledge)
            batch_size: Insert batch size

        Returns:
            Dict with 'added' and 'errors' counts
        """
        collection_name = collection_name or self.MEDICAL_COLLECTION
        collection = self._get_collection(collection_name)
        added = 0
        errors = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            batch_ids = []
            batch_docs = []
            batch_embeddings = []
            batch_metas = []

            for doc in batch:
                try:
                    content = doc["content"]
                    embedding = self.embedding_service.embed_text(content)
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    meta = doc.get("metadata", {})
                    meta["added_at"] = datetime.now().isoformat()
                    meta = self._sanitize_metadata(meta)

                    batch_ids.append(doc["id"])
                    batch_docs.append(content)
                    batch_embeddings.append(embedding)
                    batch_metas.append(meta)
                    added += 1
                except Exception as e:
                    errors += 1
                    logger.warning(f"Failed to prepare document {doc.get('id')}: {e}")

            if batch_ids:
                try:
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metas,
                    )
                except Exception as e:
                    logger.warning(f"Batch upsert failed: {e}")
                    errors += len(batch_ids)
                    added -= len(batch_ids)

        logger.info(f"Batch insert complete: {added} added, {errors} errors")
        return {"added": added, "errors": errors}

    # =========================================================================
    # STATS & UTILITIES
    # =========================================================================

    def get_collection_stats(self) -> Dict[str, int]:
        """Get document counts for all collections."""
        stats = {}
        for name in [
            self.MEDICAL_COLLECTION,
            self.DRUG_COLLECTION,
            self.SYMPTOMS_COLLECTION,
            self.MEMORIES_COLLECTION,
        ]:
            try:
                collection = self._get_collection(name)
                stats[name] = collection.count()
            except Exception as e:
                logger.warning(f"Failed to get stats for {name}: {e}")
                stats[name] = -1

        return stats

    def clear_cache(self):
        """Clear query cache."""
        with self._query_cache_lock:
            self._query_cache.clear()

        redis_client = _get_sync_redis_client()
        if redis_client:
            try:
                keys = redis_client.keys("cvec:*")
                if keys:
                    redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")

        logger.info("Vector store cache cleared")

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self._client.delete_collection(name)
            self._collections.pop(name, None)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection {name}: {e}")
            return False

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _sanitize_metadata(meta: Dict) -> Dict:
        """Sanitize metadata values for ChromaDB (must be str, int, float, or bool)."""
        sanitized = {}
        for key, value in meta.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = str(value)
        return sanitized


# ============================================================================
# Singleton instance
# ============================================================================

_chromadb_store_instance = None
_chromadb_store_lock = threading.Lock()


def get_chromadb_store(**kwargs) -> ChromaDBVectorStore:
    """Get singleton ChromaDBVectorStore instance."""
    global _chromadb_store_instance

    if _chromadb_store_instance is None:
        with _chromadb_store_lock:
            if _chromadb_store_instance is None:
                _chromadb_store_instance = ChromaDBVectorStore(**kwargs)

    return _chromadb_store_instance


# ============================================================================
# Compatibility aliases
# ============================================================================

VectorStore = ChromaDBVectorStore
get_vector_store = get_chromadb_store
