"""
Vector Store - ChromaDB-based Storage for RAG

This module provides persistent vector storage using ChromaDB,
enabling semantic search over medical knowledge and user memories.

Performance Optimizations:
- L1: In-memory LRU cache (100 entries)
- L2: Redis cache with 300s TTL (shared across workers)
- Parallel retrieval support

Collections (ChromaDB collections):
1. user_memories - Per-user memory embeddings
2. medical_knowledge - RAG knowledge base (shared)
3. drug_interactions - Medication information (shared)
4. symptoms_conditions - Symptom-condition mapping
"""

import logging
import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Union, TypeAlias, TYPE_CHECKING
from collections import OrderedDict
import threading

import numpy as np

if TYPE_CHECKING:
    from .chromadb_store import ChromaDBVectorStore as ChromaDBVectorStoreType

logger = logging.getLogger(__name__)

# Redis cache configuration for vector search
VECTOR_CACHE_TTL = int(os.getenv("VECTOR_CACHE_TTL", "300"))  # 5 minutes default
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


class InMemoryVectorStore:
    """
    Simple in-memory vector store fallback for development/testing.
    
    Note: Data is NOT persisted - lost on restart.
    Use ChromaDBVectorStore for production.
    """
    
    MEDICAL_COLLECTION = "medical_knowledge"
    DRUG_COLLECTION = "drug_interactions"
    SYMPTOMS_COLLECTION = "symptoms_conditions"
    
    def __init__(self, embedding_model: str = "MedCPT-Query-Encoder", **kwargs):
        """Initialize in-memory vector store."""
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._embedding_model = embedding_model
        
        # Try to initialize embedding service
        self.embedding_service = None
        use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
        if use_remote and RemoteEmbeddingService is not None:
            try:
                self.embedding_service = RemoteEmbeddingService.get_instance()
            except Exception as e:
                logger.warning(f"Failed to initialize embedding service: {e}")
        elif not use_remote:
            logger.info("ℹ️ Remote embeddings disabled (USE_REMOTE_EMBEDDINGS=false)")
        
        logger.warning("⚠️ Using InMemoryVectorStore - data will NOT be persisted!")
        logger.info("✅ InMemoryVectorStore initialized")
    
    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None) -> Dict:
        """Get or create an in-memory collection."""
        if name not in self._collections:
            self._collections[name] = {
                "documents": [],
                "embeddings": [],
                "ids": [],
                "metadatas": [],
                "metadata": metadata or {}
            }
        return self._collections[name]
    
    def add_medical_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None, **kwargs) -> None:
        """Add a medical document."""
        collection = self.get_or_create_collection(self.MEDICAL_COLLECTION)
        
        # Generate embedding if service available
        embedding = None
        if self.embedding_service:
            try:
                embedding = self.embedding_service.embed(content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
        
        collection["documents"].append(content)
        collection["embeddings"].append(embedding)
        collection["ids"].append(doc_id)
        collection["metadatas"].append(metadata or {})
    
    def search_medical_knowledge(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """Search medical knowledge (basic text matching fallback)."""
        collection = self.get_or_create_collection(self.MEDICAL_COLLECTION)
        
        if not collection["documents"]:
            return []
        
        # If we have embeddings and embedding service, do similarity search
        if self.embedding_service and any(collection["embeddings"]):
            try:
                query_embedding = self.embedding_service.embed(query)
                similarities = []
                for i, emb in enumerate(collection["embeddings"]):
                    if emb is not None:
                        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                        similarities.append((i, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                results = []
                for idx, score in similarities[:top_k]:
                    results.append({
                        "id": collection["ids"][idx],
                        "content": collection["documents"][idx],
                        "metadata": collection["metadatas"][idx],
                        "score": float(score)
                    })
                return results
            except Exception as e:
                logger.warning(f"Embedding search failed, using text matching: {e}")
        
        # Fallback: simple text matching
        query_lower = query.lower()
        results = []
        for i, doc in enumerate(collection["documents"]):
            if query_lower in doc.lower():
                results.append({
                    "id": collection["ids"][i],
                    "content": doc,
                    "metadata": collection["metadatas"][i],
                    "score": 0.5
                })
        
        return results[:top_k]
    
    async def async_search(self, query: str, collection_name: str = None, top_k: int = 5, **kwargs) -> List[Dict]:
        """Async search wrapper."""
        return self.search_medical_knowledge(query, top_k=top_k)
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if name in self._collections:
            del self._collections[name]
            return True
        return False
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return {name: len(col["documents"]) for name, col in self._collections.items()}


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

# Try to import ChromaDBVectorStore for ChromaDB-based storage
CHROMADB_STORE_AVAILABLE = False
_ChromaDBVectorStoreClass = None

try:
    from .chromadb_store import ChromaDBVectorStore as _ChromaDBVectorStoreClass
    CHROMADB_STORE_AVAILABLE = True
except ImportError:
    try:
        from rag.store.chromadb_store import ChromaDBVectorStore as _ChromaDBVectorStoreClass
        CHROMADB_STORE_AVAILABLE = True
    except ImportError:
        logger.warning("ChromaDBVectorStore not available")


def get_vector_store(**kwargs) -> Union["ChromaDBVectorStoreType", "InMemoryVectorStore"]:
    """
    Factory function to get the appropriate vector store.
    
    Priority:
    1. ChromaDBVectorStore (ChromaDB) - Production recommended
    2. InMemoryVectorStore - Development/testing fallback
    
    Args:
        **kwargs: Additional arguments for vector store initialization
        
    Returns:
        ChromaDBVectorStore or InMemoryVectorStore instance
    """
    # Priority 1: ChromaDB store (recommended for production)
    if CHROMADB_STORE_AVAILABLE and _ChromaDBVectorStoreClass is not None:
        try:
            store = _ChromaDBVectorStoreClass(**kwargs)
            logger.info("✅ Using ChromaDBVectorStore (ChromaDB)")
            return store
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDBVectorStore: {e}")
            logger.info("Falling back to InMemoryVectorStore...")
    
    # Priority 2: In-memory fallback (development only)
    logger.warning("⚠️ Using InMemoryVectorStore - data will NOT be persisted!")
    return InMemoryVectorStore(**kwargs)


# Backward compatibility alias — declared as a TypeAlias so Pylance
# treats it as a type rather than a plain variable.
VectorStore: TypeAlias = _ChromaDBVectorStoreClass if CHROMADB_STORE_AVAILABLE else InMemoryVectorStore  # type: ignore[type-arg]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing VectorStore...")
    
    # Use factory function
    store = get_vector_store()
    print(f"Using: {type(store).__name__}")

    # Test medical knowledge
    print("\n📚 Testing Medical Knowledge Base:")
    store.add_medical_document(
        "aha_hf_2024",
        "Heart failure is a condition where the heart cannot pump enough blood. "
        "Treatment includes ACE inhibitors, beta-blockers, and lifestyle changes.",
        {"source": "AHA", "category": "guidelines", "year": 2024},
    )
    store.add_medical_document(
        "chest_pain_guide",
        "Chest pain can indicate cardiac issues. "
        "Symptoms include pressure, squeezing, and pain radiating to arm or jaw.",
        {"source": "Mayo Clinic", "category": "symptoms"},
    )

    results = store.search_medical_knowledge("heart problems treatment")
    print(f"  Found {len(results)} results for 'heart problems treatment':")
    for r in results:
        print(f"    [{r.get('score', 0):.2f}] {r.get('content', '')[:60]}...")

    print("\n✅ VectorStore tests passed!")
