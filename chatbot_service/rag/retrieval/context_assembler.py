"""
Context Assembler for RAG System

Handles parallel retrieval from multiple sources (vector, graph, memory)
and combines results into unified context.

Performance Optimizations:
- Redis caching with 300s TTL for assembled context
- Parallel retrieval from all sources
- Document compression for token efficiency
"""


import logging
import asyncio
import hashlib
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Redis cache configuration
CONTEXT_CACHE_TTL = int(os.getenv("CONTEXT_CACHE_TTL", "300"))  # 5 minutes default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Lazy Redis import
_redis_client = None
_redis_available = None

async def _get_redis_client():
    """Get async Redis client, lazily initialized."""
    global _redis_client, _redis_available
    
    if _redis_available is False:
        return None
    
    if _redis_client is not None:
        return _redis_client
    
    try:
        from redis import asyncio as aioredis
        _redis_client = await aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2.0,
        )
        # Test connection
        await _redis_client.ping()
        _redis_available = True
        logger.info(f"✅ Context cache Redis connected (TTL={CONTEXT_CACHE_TTL}s)")
        return _redis_client
    except Exception as e:
        logger.warning(f"⚠️ Redis unavailable for context caching: {e}")
        _redis_available = False
        return None


class CompressionStrategy(Enum):
    """Supported compression strategies for document assembly."""
    EXTRACTIVE = "extractive"      # Paragraph-level relevance scoring
    SENTENCE = "sentence"          # Sentence-level filtering
    LIST_AWARE = "list_aware"      # Specialized for lists and tables
    HYBRID = "hybrid"              # Auto-select based on content type
    NONE = "none"                  # No compression


@dataclass
class RetrievalResult:
    """Container for retrieval results from a single source."""
    source: str  # "vector", "graph", "memory"
    documents: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    """Combined context from all retrieval sources."""
    vector_results: List[Dict[str, Any]] = field(default_factory=list)
    graph_results: List[Dict[str, Any]] = field(default_factory=list)
    memory_results: List[Dict[str, Any]] = field(default_factory=list)
    combined_ranked: List[Dict[str, Any]] = field(default_factory=list)
    total_documents: int = 0
    retrieval_time_ms: float = 0.0


class ContextAssembler:
    """
    Assembles context from multiple retrieval sources.
    
    Implements parallel retrieval and intelligent result combination.
    """
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        memory_bridge: Optional[Any] = None,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        memory_weight: Optional[float] = None,
        llm_gateway: Optional[Any] = None,
        enable_compression: bool = True,
        compression_strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        compression_ratio: float = 0.5,
    ):
        """
        Initialize context assembler with retrieval sources.
        
        Args:
            vector_store: Vector store for semantic search
            memory_bridge: Memory service for user context
            vector_weight: Weight for vector results (default: 0.5, from AppConfig)
            graph_weight: Weight for graph results (default: 0.35, from AppConfig)
            memory_weight: Weight for memory results (default: 0.15, from AppConfig)
            llm_gateway: Optional LLM for intelligent document compression
            enable_compression: Enable document compression after assembly
            compression_strategy: Strategy for compression (HYBRID auto-selects)
            compression_ratio: Target compression ratio (0.5 = 50% of original)
        """
        self.vector_store = vector_store
        self.memory_bridge = memory_bridge
        
        # Load weights from config if not provided
        if vector_weight is None or graph_weight is None or memory_weight is None:
            try:
                from core.config.app_config import get_app_config
                config = get_app_config()
                vector_weight = vector_weight or config.rag.vector_weight
                graph_weight = graph_weight or config.rag.graph_weight
                memory_weight = memory_weight or config.rag.memory_weight
                logger.info(
                    f"Loaded retrieval weights from AppConfig: "
                    f"vector={vector_weight}, graph={graph_weight}, memory={memory_weight}"
                )
            except Exception as e:
                logger.warning(f"Failed to load weights from AppConfig: {e}. Using defaults.")
                vector_weight = vector_weight or 0.5
                graph_weight = graph_weight or 0.35
                memory_weight = memory_weight or 0.15
        
        # Validate weights sum to 1.0
        total_weight = vector_weight + graph_weight + memory_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Retrieval weights do not sum to 1.0 (sum={total_weight}). "
                f"Normalizing..."
            )
            self.vector_weight = vector_weight / total_weight
            self.graph_weight = graph_weight / total_weight
            self.memory_weight = memory_weight / total_weight
        else:
            self.vector_weight = vector_weight
            self.graph_weight = graph_weight
            self.memory_weight = memory_weight
        
        # P3.3: Initialize document compressor
        self.enable_compression = enable_compression
        self.compression_strategy = compression_strategy
        self.compression_ratio = compression_ratio
        self.compressor = None
        
        if enable_compression:
            try:
                from rag.retrieval.unified_compressor import UnifiedDocumentCompressor
                self.compressor = UnifiedDocumentCompressor(
                    llm_gateway=llm_gateway,
                    target_ratio=compression_ratio,
                    default_strategy=compression_strategy
                )
                logger.info(f"✅ Document compression enabled (strategy={compression_strategy.value}, ratio={compression_ratio})")
            except ImportError:
                logger.warning("UnifiedDocumentCompressor not available, compression disabled")
                self.enable_compression = False
        
        # Cache settings
        self.cache_ttl = CONTEXT_CACHE_TTL
        self._cache_enabled = True  # Can be disabled via env var
        
        logger.info(
            f"ContextAssembler initialized: "
            f"vector={bool(vector_store)}, "
            f"memory={bool(memory_bridge)}, "
            f"weights=(vector={self.vector_weight:.2f}, graph={self.graph_weight:.2f}, memory={self.memory_weight:.2f}), "
            f"cache_ttl={self.cache_ttl}s"
        )
    
    def _generate_cache_key(self, query: str, user_id: Optional[str], top_k: int) -> str:
        """
        Generate deterministic cache key from query parameters.
        
        Uses SHA256 hash for consistent key length and collision resistance.
        """
        key_data = f"{query}:{user_id or 'anon'}:{top_k}:{self.vector_weight}:{self.graph_weight}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"ctx:asm:{key_hash}"
    
    async def _get_cached_context(self, cache_key: str) -> Optional[AssembledContext]:
        """Retrieve cached context from Redis."""
        if not self._cache_enabled:
            return None
        
        try:
            redis = await _get_redis_client()
            if redis is None:
                return None
            
            cached = await redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                logger.debug(f"✅ Cache hit for context: {cache_key}")
                return AssembledContext(
                    vector_results=data.get("vector_results", []),
                    graph_results=data.get("graph_results", []),
                    memory_results=data.get("memory_results", []),
                    combined_ranked=data.get("combined_ranked", []),
                    total_documents=data.get("total_documents", 0),
                    retrieval_time_ms=data.get("retrieval_time_ms", 0.0),
                )
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    async def _set_cached_context(self, cache_key: str, context: AssembledContext):
        """Store context in Redis cache."""
        if not self._cache_enabled:
            return
        
        try:
            redis = await _get_redis_client()
            if redis is None:
                return
            
            data = {
                "vector_results": context.vector_results,
                "graph_results": context.graph_results,
                "memory_results": context.memory_results,
                "combined_ranked": context.combined_ranked,
                "total_documents": context.total_documents,
                "retrieval_time_ms": context.retrieval_time_ms,
            }
            await redis.setex(cache_key, self.cache_ttl, json.dumps(data))
            logger.debug(f"✅ Cached context: {cache_key} (TTL={self.cache_ttl}s)")
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    async def assemble(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        skip_cache: bool = False,
    ) -> AssembledContext:
        """
        Assemble context from all sources.
        
        Performs parallel retrieval from vector store, knowledge graph,
        and user memory (if available).
        
        Performance: Uses Redis cache with 300s TTL for repeated queries.
        
        Args:
            query: The search query
            user_id: Optional user ID for memory context
            top_k: Number of results per source
            skip_cache: Force fresh retrieval (bypass cache)
        """
        import time
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, user_id, top_k)
        
        # Check cache first (unless skip_cache is True)
        if not skip_cache:
            cached = await self._get_cached_context(cache_key)
            if cached:
                # Update retrieval time to reflect cache hit
                elapsed = (time.time() - start_time) * 1000
                cached.retrieval_time_ms = elapsed
                logger.info(f"⚡ Context cache hit ({elapsed:.1f}ms vs original {cached.retrieval_time_ms:.1f}ms)")
                return cached
        
        try:
            # Create parallel retrieval tasks
            tasks = []
            
            if self.vector_store:
                tasks.append(self._vector_search(query, top_k))
            else:
                tasks.append(self._empty_search("vector"))
            
            # Graph search slot (no backend configured)
            tasks.append(self._empty_search("graph"))
            
            if self.memory_bridge and user_id:
                tasks.append(self._memory_search(query, user_id, top_k))
            else:
                tasks.append(self._empty_search("memory"))
            
            # Execute in parallel with timeout to prevent slow sources from blocking
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0  # Max 2 seconds for all retrievals
                )
            except asyncio.TimeoutError:
                logger.warning("Context assembly timed out after 2s, using partial results")
                results = [[], [], []]  # Return empty results on timeout
            
            vector_results = results[0] if not isinstance(results[0], Exception) else []
            graph_results = results[1] if not isinstance(results[1], Exception) else []
            memory_results = results[2] if not isinstance(results[2], Exception) else []
            
            # Combine results
            combined = self._combine_and_rank(
                vector_results, graph_results, memory_results
            )
            
            # P3.3: Apply compression if enabled
            if self.enable_compression and combined and self.compressor:
                try:
                    logger.debug(f"Compressing {len(combined)} documents using {self.compression_strategy.value} strategy")
                    # Convert dict results to LangChain Document format for compressor
                    from langchain_core.documents import Document
                    docs = [
                        Document(
                            page_content=doc.get("content", "") or doc.get("text", ""),
                            metadata=doc.get("metadata", {})
                        )
                        for doc in combined
                    ]
                    
                    compressed = await self.compressor.compress(
                        query=query,
                        documents=docs,
                        strategy=self.compression_strategy
                    )
                    
                    # Safety check: ensure compressed is not None and has valid results
                    if compressed is None or not isinstance(compressed, list):
                        logger.warning(f"Compression returned invalid result: {type(compressed)}, using uncompressed")
                        raise ValueError("Compression returned None or invalid type")
                    
                    # Convert compressed docs back to dict format
                    combined = [
                        {
                            **{k: v for k, v in doc_dict.items() if k not in ['compressed_content', 'original_content']},
                            "content": comp_doc.compressed_content if comp_doc else doc_dict.get("content", ""),
                            "compressed": True if comp_doc else False,
                            "compression_ratio": comp_doc.compression_ratio if comp_doc else 1.0,
                        }
                        for doc_dict, comp_doc in zip(combined, compressed)
                        if comp_doc is not None  # Filter out None results
                    ]
                    
                    if compressed:
                        avg_ratio = sum(c.compression_ratio for c in compressed if c) / len([c for c in compressed if c])
                        logger.debug(f"Compression complete: avg ratio={avg_ratio:.2f}")
                    else:
                        logger.warning("No valid compression results, using uncompressed")
                except Exception as e:
                    logger.warning(f"Compression failed, using uncompressed results: {e}")
            
            elapsed = (time.time() - start_time) * 1000
            
            context = AssembledContext(
                vector_results=vector_results,
                graph_results=graph_results,
                memory_results=memory_results,
                combined_ranked=combined,
                total_documents=len(vector_results) + len(graph_results) + len(memory_results),
                retrieval_time_ms=elapsed,
            )
            
            # Cache the result for future requests
            await self._set_cached_context(cache_key, context)
            
            logger.info(
                f"Context assembled in {elapsed:.1f}ms: "
                f"{len(vector_results)} vector + {len(graph_results)} graph + {len(memory_results)} memory"
            )
            
            return context
        
        except Exception as e:
            logger.error(f"Error assembling context: {e}", exc_info=True)
            elapsed = (time.time() - start_time) * 1000
            return AssembledContext(retrieval_time_ms=elapsed)
    
    async def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Search vector store."""
        try:
            if hasattr(self.vector_store, 'search_medical_knowledge'):
                results = self.vector_store.search_medical_knowledge(query, top_k=top_k)
            elif hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(query, top_k=top_k)
            else:
                results = []
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
    
    async def _memory_search(
        self,
        query: str,
        user_id: str,
        top_k: int,
    ) -> List[Dict]:
        """Search user memory."""
        try:
            if hasattr(self.memory_bridge, 'search'):
                result = self.memory_bridge.search(query, user_id=user_id, top_k=top_k)
                # Handle both async and sync search methods
                if asyncio.iscoroutine(result):
                    results = await result
                else:
                    results = result
            else:
                results = []
            
            # Ensure results is a list
            if results is None:
                results = []
            
            logger.debug(f"Memory search returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []
    
    async def _empty_search(self, source: str) -> List[Dict]:
        """Return empty results for disabled source."""
        logger.debug(f"{source} source not available")
        return []
    
    def _combine_and_rank(
        self,
        vector: List[Dict],
        graph: List[Dict],
        memory: List[Dict],
    ) -> List[Dict]:
        """
        Combine and rank results from all sources.
        
        Uses weighted scoring (configurable via AppConfig):
        - Vector: {self.vector_weight} (semantic similarity)
        - Graph: {self.graph_weight} (structured knowledge)
        - Memory: {self.memory_weight} (user context)
        """
        combined = []
        
        # Add vector results with weight
        for doc in vector:
            combined.append({
                **doc,
                "source": "vector",
                "combined_score": doc.get("score", 0.0) * self.vector_weight,
            })
        
        # Add graph results with weight
        for doc in graph:
            combined.append({
                **doc,
                "source": "graph",
                "combined_score": doc.get("score", 0.0) * self.graph_weight,
            })
        
        # Add memory results with weight
        for doc in memory:
            combined.append({
                **doc,
                "source": "memory",
                "combined_score": doc.get("score", 0.0) * self.memory_weight,
            })
        
        # Sort by combined score
        combined.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        logger.info(
            f"Combined context: {len(vector)} vector + {len(graph)} graph + {len(memory)} memory "
            f"= {len(combined)} total documents "
            f"(weights: vector={self.vector_weight:.2f}, graph={self.graph_weight:.2f}, memory={self.memory_weight:.2f})"
        )
        
        return combined
