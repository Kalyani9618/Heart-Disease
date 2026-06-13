"""
Memory Search Engine - Intelligent memory retrieval using Pydantic models

Updated: Added embedding-based semantic search support
"""

import asyncio
import json
import os
import threading
import time
import numpy as np
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, List, Dict

import openai
from loguru import logger

# Import PromptRegistry for centralized prompts
from core.prompts.registry import get_prompt

# Optional embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Optional remote embedding service
try:
    from rag.embedding.remote import RemoteEmbeddingService
    REMOTE_EMBEDDING_AVAILABLE = True
except ImportError:
    REMOTE_EMBEDDING_AVAILABLE = False
    RemoteEmbeddingService = None

if TYPE_CHECKING:
    from ..core.providers import ProviderConfig

from ..utils.pydantic_models import MemorySearchQuery


# ============================================================================
# EMBEDDING-BASED SEMANTIC SEARCH
# ============================================================================


class EmbeddingSearchEngine:
    """
    Embedding-based semantic search for memory retrieval.
    
    Uses sentence-transformers for local embeddings or OpenAI API for remote.
    Supports cosine similarity search with configurable thresholds.
    
    Features:
    - Local embedding with sentence-transformers (fast, no API cost)
    - OpenAI embedding API fallback (higher quality)
    - Embedding cache for performance
    - Result ranking by similarity score
    """
    
    # Default embedding model
    DEFAULT_LOCAL_MODEL = "MedCPT-Query-Encoder"  # Remote MedCPT, 768 dimensions
    DEFAULT_OPENAI_MODEL = "text-embedding-3-small"  # 1536 dimensions
    
    def __init__(
        self,
        use_local: bool = True,
        local_model: str = None,
        openai_client: Any = None,
        openai_model: str = None,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize Embedding Search Engine.
        
        Args:
            use_local: Use local sentence-transformers model
            local_model: Local model name (default: MedCPT-Query-Encoder)
            openai_client: OpenAI client for API embeddings
            openai_model: OpenAI embedding model name
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.use_local = use_local and SENTENCE_TRANSFORMERS_AVAILABLE
        self.similarity_threshold = similarity_threshold
        self.local_model = None
        self.remote_service = None
        self.openai_client = openai_client
        self.openai_model = openai_model or self.DEFAULT_OPENAI_MODEL
        
        # Embedding cache: {text_hash: embedding_vector}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 10000
        
        # Try remote embedding service first (if not using local)
        use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
        if not self.use_local and REMOTE_EMBEDDING_AVAILABLE and use_remote:
            try:
                self.remote_service = RemoteEmbeddingService.get_instance()
                logger.info("EmbeddingSearchEngine: Using RemoteEmbeddingService (MedCPT)")
            except Exception as e:
                logger.warning(f"Remote embedding service unavailable: {e}")
        
        # Initialize local model if available and requested
        if self.use_local:
            try:
                model_name = local_model or self.DEFAULT_LOCAL_MODEL
                self.local_model = SentenceTransformer(model_name)
                logger.info(f"EmbeddingSearchEngine: Loaded local model '{model_name}'")
            except Exception as e:
                logger.warning(f"Failed to load local embedding model: {e}")
                self.use_local = False
        
        if not self.use_local and not self.remote_service and not self.openai_client:
            logger.warning(
                "EmbeddingSearchEngine: No embedding method available. "
                "Set COLAB_API_URL for remote embeddings, install sentence-transformers, or provide OpenAI client."
            )
    
    def _hash_text(self, text: str) -> str:
        """Create hash for text caching."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embedding or None
        """
        if not text:
            return None
        
        # Check cache
        text_hash = self._hash_text(text)
        with self._cache_lock:
            if text_hash in self._embedding_cache:
                return self._embedding_cache[text_hash]
        
        embedding = None
        
        # Try local model first
        if self.use_local and self.local_model:
            try:
                embedding = self.local_model.encode(text, convert_to_numpy=True)
            except Exception as e:
                logger.warning(f"Local embedding failed: {e}")
        
        # Try remote embedding service
        if embedding is None and self.remote_service:
            try:
                embedding_list = self.remote_service.embed_text(text)
                if embedding_list:
                    embedding = np.array(embedding_list)
            except Exception as e:
                logger.warning(f"Remote embedding failed: {e}")
        
        # Fall back to OpenAI
        if embedding is None and self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.openai_model
                )
                embedding = np.array(response.data[0].embedding)
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}")
        
        # Cache result
        if embedding is not None:
            with self._cache_lock:
                # Evict old entries if cache is full
                if len(self._embedding_cache) >= self._max_cache_size:
                    # Remove oldest 10%
                    keys_to_remove = list(self._embedding_cache.keys())[:self._max_cache_size // 10]
                    for key in keys_to_remove:
                        del self._embedding_cache[key]
                self._embedding_cache[text_hash] = embedding
        
        return embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding arrays (None for failures)
        """
        if not texts:
            return []
        
        # Check cache for all
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        with self._cache_lock:
            for i, text in enumerate(texts):
                if text:
                    text_hash = self._hash_text(text)
                    if text_hash in self._embedding_cache:
                        results[i] = self._embedding_cache[text_hash]
                    else:
                        uncached_indices.append(i)
                        uncached_texts.append(text)
        
        if not uncached_texts:
            return results
        
        # Batch embed uncached texts
        embeddings = []
        
        if self.use_local and self.local_model:
            try:
                embeddings = self.local_model.encode(uncached_texts, convert_to_numpy=True)
            except Exception as e:
                logger.warning(f"Batch local embedding failed: {e}")
                embeddings = [None] * len(uncached_texts)
        elif self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=uncached_texts,
                    model=self.openai_model
                )
                embeddings = [np.array(item.embedding) for item in response.data]
            except Exception as e:
                logger.warning(f"Batch OpenAI embedding failed: {e}")
                embeddings = [None] * len(uncached_texts)
        else:
            embeddings = [None] * len(uncached_texts)
        
        # Update results and cache
        with self._cache_lock:
            for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
                results[idx] = emb
                if emb is not None:
                    self._embedding_cache[self._hash_text(text)] = emb
        
        return results
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def search(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        content_field: str = "content",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories using embedding similarity.
        
        Args:
            query: Search query
            memories: List of memory dictionaries
            content_field: Field containing text content
            limit: Maximum results to return
            
        Returns:
            List of memories sorted by similarity (includes similarity_score)
        """
        if not query or not memories:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            logger.warning("Could not generate query embedding")
            return []
        
        # Get embeddings for all memories
        memory_texts = [
            str(m.get(content_field, "") or m.get("summary", "") or "")
            for m in memories
        ]
        memory_embeddings = self.get_batch_embeddings(memory_texts)
        
        # Calculate similarities
        results = []
        for memory, embedding in zip(memories, memory_embeddings):
            if embedding is not None:
                similarity = self.cosine_similarity(query_embedding, embedding)
                if similarity >= self.similarity_threshold:
                    memory_copy = memory.copy()
                    memory_copy["similarity_score"] = similarity
                    memory_copy["search_strategy"] = "semantic_embedding"
                    results.append(memory_copy)
        
        # Sort by similarity
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        logger.debug(f"Embedding search found {len(results)} results above threshold {self.similarity_threshold}")
        return results[:limit]
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with self._cache_lock:
            self._embedding_cache.clear()
        logger.debug("Embedding cache cleared")


class MemorySearchEngine:
    """
    Pydantic-based search engine for intelligent memory retrieval.
    Uses OpenAI Structured Outputs to understand queries and plan searches.
    Prompts are now centrally managed via PromptRegistry.
    """

    @property
    def SYSTEM_PROMPT(self) -> str:
        """Get system prompt from PromptRegistry for memory search agent."""
        return get_prompt("memori", "search_agent")

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider_config: Optional["ProviderConfig"] = None,
    ):
        """
        Initialize Memory Search Engine with LLM provider configuration

        Args:
            api_key: API key (deprecated, use provider_config)
            model: Model to use for query understanding (defaults to 'gpt-4o' if not specified)
            provider_config: Provider configuration for LLM client
        """
        if provider_config:
            # Use provider configuration to create client
            self.client = provider_config.create_client()
            # Use provided model, fallback to provider config model, then default to gpt-4o
            self.model = model or provider_config.model or "gpt-4o"
            logger.debug(f"Search engine initialized with model: {self.model}")
            self.provider_config = provider_config
        else:
            # Backward compatibility: try to detect provider from environment first
            try:
                from ..core.providers import detect_provider_from_env
                
                self.provider_config = detect_provider_from_env()
                self.client = self.provider_config.create_client()
                self.model = model or self.provider_config.model or "gpt-4o"
                logger.debug(f"Search engine initialized from environment with model: {self.model}")
            except (ImportError, Exception) as e:
                # Final fallback: use api_key directly for OpenAI
                logger.debug(f"Provider detection failed ({e}), using direct API key")
                self.client = openai.OpenAI(api_key=api_key)
                self.model = model or "gpt-4o"
                self.provider_config = None

        # Determine if we're using a local/custom endpoint that might not support structured outputs
        self._supports_structured_outputs = self._detect_structured_output_support()
        self._max_tokens = int(os.getenv("MEMORI_MAX_TOKENS", "2000"))

        # Performance improvements
        self._query_cache = {}  # Cache for search plans
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_lock = threading.Lock()

        # Background processing
        self._background_executor = None

        # Database type detection for unified search
        self._database_type = None

    def _detect_database_type(self, db_manager):
        """Detect database type from db_manager"""
        if self._database_type is None:
            self._database_type = getattr(db_manager, "database_type", "sql")
            logger.debug(
                f"MemorySearchEngine: Detected database type: {self._database_type}"
            )
        return self._database_type

    def plan_search(self, query: str, context: str | None = None) -> MemorySearchQuery:
        """
        Plan search strategy for a user query using OpenAI Structured Outputs with caching

        Args:
            query: User's search query
            context: Optional additional context

        Returns:
            Structured search query plan
        """
        try:
            # Create cache key
            cache_key = f"{query}|{context or ''}"

            # Check cache first
            with self._cache_lock:
                if cache_key in self._query_cache:
                    cached_result, timestamp = self._query_cache[cache_key]
                    if time.time() - timestamp < self._cache_ttl:
                        logger.debug(f"Using cached search plan for: {query}")
                        return cached_result

            # Prepare the prompt with internal marker to prevent recording
            prompt = f"[INTERNAL_MEMORI_SEARCH]\nUser query: {query}"
            if context:
                prompt += f"\nAdditional context: {context}"

            # Try structured outputs first, fall back to manual parsing
            search_query = None

            if self._supports_structured_outputs:
                try:
                    # Call OpenAI Structured Outputs
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        response_format=MemorySearchQuery,
                        temperature=0.1,
                        max_tokens=self._max_tokens,
                    )

                    # Handle potential refusal
                    if completion.choices[0].message.refusal:
                        logger.warning(
                            f"Search planning refused: {completion.choices[0].message.refusal}"
                        )
                        return self._create_fallback_query(query)

                    search_query = completion.choices[0].message.parsed

                except Exception as e:
                    logger.warning(
                        f"Structured outputs failed for search planning, falling back to manual parsing: {e}"
                    )
                    self._supports_structured_outputs = (
                        False  # Disable for future calls
                    )
                    search_query = None

            # Fallback to manual parsing if structured outputs failed or not supported
            if search_query is None:
                search_query = self._plan_search_with_fallback_parsing(query)

            # Cache the result
            with self._cache_lock:
                self._query_cache[cache_key] = (search_query, time.time())
                # Clean old cache entries
                self._cleanup_cache()

            logger.debug(
                f"Planned search for query '{query}': intent='{search_query.intent}', strategies={search_query.search_strategy}"
            )
            return search_query

        except Exception as e:
            logger.error(f"Search planning failed: {e}")
            return self._create_fallback_query(query)

    def execute_search(
        self, query: str, db_manager, user_id: str = "default", limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Execute intelligent search using planned strategies

        Args:
            query: User's search query
            db_manager: Database manager instance (SQL or MongoDB)
            user_id: User identifier for multi-tenant isolation
            limit: Maximum results to return

        Returns:
            List of relevant memory items with search metadata
        """
        try:
            # Detect database type for optimal search strategy
            db_type = self._detect_database_type(db_manager)

            # Plan the search
            search_plan = self.plan_search(query)
            logger.debug(
                f"Search plan for '{query}': strategies={search_plan.search_strategy}, entities={search_plan.entity_filters}, db_type={db_type}"
            )

            all_results = []
            seen_memory_ids = set()

            # For MongoDB and SQL, use SearchService directly to avoid recursion
            # This ensures we use the database's native search capabilities without triggering context injection
            logger.debug(f"Executing direct SearchService search using {db_type}")
            try:
                from ..database.search_service import SearchService

                with db_manager.SessionLocal() as session:
                    search_service = SearchService(session, db_type)
                    primary_results = search_service.search_memories(
                        query=search_plan.query_text or query,
                        user_id=user_id,
                        limit=limit,
                    )
                logger.debug(
                    f"Direct SearchService returned {len(primary_results)} results"
                )
            except Exception as e:
                logger.error(f"SearchService direct access failed: {e}")
                primary_results = []

            # Process primary results and add search metadata
            for result in primary_results:
                if (
                    isinstance(result, dict)
                    and result.get("memory_id") not in seen_memory_ids
                ):
                    seen_memory_ids.add(result["memory_id"])
                    result["search_strategy"] = f"{db_type}_unified_search"
                    result["search_reasoning"] = f"Direct {db_type} database search"
                    all_results.append(result)

            # If we have room for more results and specific entity filters, try keyword search
            if len(all_results) < limit and search_plan.entity_filters:
                logger.debug(
                    f"Adding targeted keyword search for: {search_plan.entity_filters}"
                )
                keyword_results = self._execute_keyword_search(
                    search_plan, db_manager, user_id, limit - len(all_results)
                )

                for result in keyword_results:
                    if (
                        isinstance(result, dict)
                        and result.get("memory_id") not in seen_memory_ids
                    ):
                        seen_memory_ids.add(result["memory_id"])
                        result["search_strategy"] = "keyword_search"
                        result["search_reasoning"] = (
                            f"Keyword match for: {', '.join(search_plan.entity_filters)}"
                        )
                        all_results.append(result)

            # If we have room for more results, try category-based search
            if len(all_results) < limit and (
                search_plan.category_filters
                or "category_filter" in search_plan.search_strategy
            ):
                logger.debug(
                    f"Adding category search for: {[c.value for c in search_plan.category_filters]}"
                )
                category_results = self._execute_category_search(
                    search_plan, db_manager, user_id, limit - len(all_results)
                )

                for result in category_results:
                    if (
                        isinstance(result, dict)
                        and result.get("memory_id") not in seen_memory_ids
                    ):
                        seen_memory_ids.add(result["memory_id"])
                        result["search_strategy"] = "category_filter"
                        result["search_reasoning"] = (
                            f"Category match: {', '.join([c.value for c in search_plan.category_filters])}"
                        )
                        all_results.append(result)

            # If we have room for more results, try importance-based search
            if len(all_results) < limit and (
                search_plan.min_importance > 0.0
                or "importance_filter" in search_plan.search_strategy
            ):
                logger.debug(
                    f"Adding importance search with min_importance: {search_plan.min_importance}"
                )
                importance_results = self._execute_importance_search(
                    search_plan, db_manager, user_id, limit - len(all_results)
                )

                for result in importance_results:
                    if (
                        isinstance(result, dict)
                        and result.get("memory_id") not in seen_memory_ids
                    ):
                        seen_memory_ids.add(result["memory_id"])
                        result["search_strategy"] = "importance_filter"
                        result["search_reasoning"] = (
                            f"High importance (≥{search_plan.min_importance})"
                        )
                        all_results.append(result)

            # If we have room for more results and semantic search is recommended, use embeddings
            if len(all_results) < limit and "semantic_search" in search_plan.search_strategy:
                logger.debug(
                    f"Adding semantic search for query: {query}"
                )
                semantic_results = self._execute_semantic_search(
                    query, db_manager, user_id, limit - len(all_results)
                )

                for result in semantic_results:
                    if (
                        isinstance(result, dict)
                        and result.get("memory_id") not in seen_memory_ids
                    ):
                        seen_memory_ids.add(result["memory_id"])
                        result["search_strategy"] = "semantic_search"
                        result["search_reasoning"] = (
                            f"Semantic similarity match (score: {result.get('similarity_score', 'N/A')})"
                        )
                        all_results.append(result)

            # Filter out any non-dictionary results before processing
            valid_results = []
            for result in all_results:
                if isinstance(result, dict):
                    valid_results.append(result)
                else:
                    logger.warning(
                        f"Filtering out non-dict search result: {type(result)}"
                    )

            all_results = valid_results

            # Sort by relevance (importance score + recency)
            if all_results:

                def safe_created_at_parse(created_at_value):
                    """Safely parse created_at value to datetime"""
                    try:
                        if created_at_value is None:
                            return datetime.fromisoformat("2000-01-01")
                        if isinstance(created_at_value, str):
                            return datetime.fromisoformat(created_at_value)
                        if hasattr(created_at_value, "isoformat"):  # datetime object
                            return created_at_value
                        # Fallback for any other type
                        return datetime.fromisoformat("2000-01-01")
                    except (ValueError, TypeError):
                        return datetime.fromisoformat("2000-01-01")

                all_results.sort(
                    key=lambda x: (
                        x.get("importance_score", 0) * 0.7  # Importance weight
                        + (
                            datetime.now().replace(tzinfo=None)  # Ensure timezone-naive
                            - safe_created_at_parse(x.get("created_at")).replace(
                                tzinfo=None
                            )
                        ).days
                        * -0.001  # Recency weight
                    ),
                    reverse=True,
                )

                # Add search metadata
                for result in all_results:
                    result["search_metadata"] = {
                        "original_query": query,
                        "interpreted_intent": search_plan.intent,
                        "search_timestamp": datetime.now().isoformat(),
                    }

            logger.debug(
                f"Search executed for '{query}': {len(all_results)} results found"
            )
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []

    def _execute_keyword_search(
        self, search_plan: MemorySearchQuery, db_manager, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Execute keyword-based search"""
        keywords = search_plan.entity_filters
        if not keywords:
            # Extract keywords from query text
            keywords = [
                word.strip()
                for word in search_plan.query_text.split()
                if len(word.strip()) > 2
            ]

        search_terms = " ".join(keywords)
        try:
            # Use SearchService directly to avoid recursion
            from ..database.search_service import SearchService

            db_type = self._detect_database_type(db_manager)

            with db_manager.SessionLocal() as session:
                search_service = SearchService(session, db_type)
                results = search_service.search_memories(
                    query=search_terms, user_id=user_id, limit=limit
                )

            # Ensure results is a list of dictionaries
            if not isinstance(results, list):
                logger.warning(f"Search returned non-list result: {type(results)}")
                return []

            # Filter out any non-dictionary items
            valid_results = []
            for result in results:
                if isinstance(result, dict):
                    valid_results.append(result)
                else:
                    logger.warning(f"Search returned non-dict item: {type(result)}")

            return valid_results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _execute_category_search(
        self, search_plan: MemorySearchQuery, db_manager, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Execute category-based search"""
        categories = (
            [cat.value for cat in search_plan.category_filters]
            if search_plan.category_filters
            else []
        )

        if not categories:
            return []

        # Use SearchService directly to avoid recursion
        # Get all memories and filter by category
        logger.debug(
            f"Searching memories by categories: {categories} for user_id: {user_id}"
        )
        try:
            from ..database.search_service import SearchService

            db_type = self._detect_database_type(db_manager)

            with db_manager.SessionLocal() as session:
                search_service = SearchService(session, db_type)
                all_results = search_service.search_memories(
                    query="", user_id=user_id, limit=limit * 3
                )
        except Exception as e:
            logger.error(f"Category search failed: {e}")
            all_results = []

        logger.debug(
            f"Retrieved {len(all_results)} total results for category filtering"
        )

        filtered_results = []
        for i, result in enumerate(all_results):
            logger.debug(f"Processing result {i+1}/{len(all_results)}: {type(result)}")

            # Extract category from processed_data if it's stored as JSON
            try:
                memory_category = None

                # Check processed_data field first
                if "processed_data" in result and result["processed_data"]:
                    processed_data = result["processed_data"]
                    logger.debug(
                        f"Found processed_data: {type(processed_data)} - {str(processed_data)[:100]}..."
                    )

                    # Handle both dict and JSON string formats
                    if isinstance(processed_data, str):
                        try:
                            processed_data = json.loads(processed_data)
                        except json.JSONDecodeError as je:
                            logger.debug(f"JSON decode error for processed_data: {je}")
                            continue

                    if isinstance(processed_data, dict):
                        # Try multiple possible category locations
                        category_paths = [
                            ["category", "primary_category"],
                            ["category"],
                            ["primary_category"],
                            ["metadata", "category"],
                            ["classification", "category"],
                        ]

                        for path in category_paths:
                            temp_data = processed_data
                            try:
                                for key in path:
                                    temp_data = temp_data.get(key, {})
                                if isinstance(temp_data, str) and temp_data:
                                    memory_category = temp_data
                                    logger.debug(
                                        f"Found category via path {path}: {memory_category}"
                                    )
                                    break
                            except (AttributeError, TypeError):
                                continue
                    else:
                        logger.debug(
                            f"processed_data is not a dict after parsing: {type(processed_data)}"
                        )
                        continue

                # Fallback: check direct category field
                if not memory_category and "category" in result and result["category"]:
                    memory_category = result["category"]
                    logger.debug(f"Found category via direct field: {memory_category}")

                # Check if the found category matches any of our target categories
                if memory_category:
                    logger.debug(
                        f"Comparing memory category '{memory_category}' against target categories {categories}"
                    )
                    if memory_category in categories:
                        filtered_results.append(result)
                        logger.debug(f"✓ Category match found: {memory_category}")
                    else:
                        logger.debug(
                            f"✗ Category mismatch: {memory_category} not in {categories}"
                        )
                else:
                    logger.debug("No category found in result")

            except Exception as e:
                logger.debug(f"Error processing result {i+1}: {e}")
                continue

        logger.debug(
            f"Category filtering complete: {len(filtered_results)} results match categories {categories}"
        )
        return filtered_results[:limit]

    def _detect_structured_output_support(self) -> bool:
        """
        Detect if the current provider/endpoint supports OpenAI structured outputs

        Returns:
            True if structured outputs are likely supported, False otherwise
        """
        try:
            # Check if we have a provider config with api_type
            if self.provider_config and hasattr(self.provider_config, "api_type"):
                api_type = self.provider_config.api_type
                
                if api_type == "openrouter":
                    # OpenRouter supports structured outputs for most OpenAI models
                    model = getattr(self.provider_config, "model", "")
                    # OpenAI models on OpenRouter support structured outputs
                    if model and model.startswith("openai/"):
                        logger.debug(
                            f"Detected OpenRouter with OpenAI model ({model}), enabling structured outputs"
                        )
                        return True
                    else:
                        # Other models on OpenRouter may not support structured outputs
                        logger.debug(
                            f"Detected OpenRouter with non-OpenAI model ({model}), disabling structured outputs"
                        )
                        return False
                
                elif api_type == "llama_local":
                    # Local Llama servers typically don't support beta features
                    logger.debug(
                        "Detected Llama Local endpoint, disabling structured outputs"
                    )
                    return False
                
                elif api_type == "azure":
                    return self._test_azure_structured_outputs_support()
                
                elif api_type in ["custom", "openai_compatible"]:
                    logger.debug(
                        f"Detected {api_type} endpoint, disabling structured outputs"
                    )
                    return False
                
                elif api_type == "openai":
                    logger.debug("Detected OpenAI endpoint, enabling structured outputs")
                    return True

            # Check if we have a provider config with custom base_url
            if self.provider_config and hasattr(self.provider_config, "base_url"):
                base_url = self.provider_config.base_url
                if base_url:
                    # Local/custom endpoints typically don't support beta features
                    if "localhost" in base_url or "127.0.0.1" in base_url:
                        logger.debug(
                            f"Detected local endpoint ({base_url}), disabling structured outputs"
                        )
                        return False
                    # OpenRouter endpoint
                    if "openrouter.ai" in base_url:
                        logger.debug(
                            f"Detected OpenRouter endpoint ({base_url}), enabling structured outputs"
                        )
                        return True
                    # Custom endpoints that aren't OpenAI
                    if "api.openai.com" not in base_url:
                        logger.debug(
                            f"Detected custom endpoint ({base_url}), disabling structured outputs"
                        )
                        return False

            # Default: assume OpenAI endpoint supports structured outputs
            logger.debug("Assuming OpenAI endpoint, enabling structured outputs")
            return True

        except Exception as e:
            logger.debug(
                f"Error detecting structured output support: {e}, defaulting to enabled"
            )
            return True

    def _test_azure_structured_outputs_support(self) -> bool:
        """
        Test if Azure OpenAI supports structured outputs by making a test call

        Returns:
            True if structured outputs are supported, False otherwise
        """
        try:
            from pydantic import BaseModel

            # Simple test model
            class TestModel(BaseModel):
                test_field: str

            # Try to make a structured output call
            test_response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": "Say hello"}],
                response_format=TestModel,
                max_tokens=10,
                temperature=0,
            )

            if (
                test_response
                and hasattr(test_response, "choices")
                and test_response.choices
            ):
                logger.debug(
                    "Azure endpoint supports structured outputs - test successful"
                )
                return True
            else:
                logger.debug(
                    "Azure endpoint structured outputs test failed - response invalid"
                )
                return False

        except Exception as e:
            # If structured outputs fail, log the error and fall back to regular completions
            logger.debug(f"Azure endpoint doesn't support structured outputs: {e}")
            return False

    def _plan_search_with_fallback_parsing(self, query: str) -> MemorySearchQuery:
        """
        Plan search strategy using regular chat completions with manual JSON parsing

        This method works with any OpenAI-compatible API that supports chat completions
        but doesn't support structured outputs (like Ollama, local models, etc.)
        """
        try:
            # Prepare the prompt from raw query with internal marker
            prompt = f"[INTERNAL_MEMORI_SEARCH]\nUser query: {query}"

            # Enhanced system prompt for JSON output
            json_system_prompt = (
                self.SYSTEM_PROMPT
                + "\n\nIMPORTANT: You MUST respond with a valid JSON object that matches this exact schema:\n"
            )
            json_system_prompt += self._get_search_query_json_schema()
            json_system_prompt += "\n\nRespond ONLY with the JSON object, no additional text or formatting."

            # Call regular chat completions
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": json_system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=1000,  # Ensure enough tokens for full response
            )

            # Extract and parse JSON response
            response_text = completion.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from model")

            # Clean up response (remove markdown formatting if present)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            try:
                parsed_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for search planning: {e}")
                logger.debug(f"Raw response: {response_text}")
                return self._create_fallback_query(query)

            # Convert to MemorySearchQuery object with validation and defaults
            search_query = self._create_search_query_from_dict(parsed_data, query)

            logger.debug("Successfully parsed search query using fallback method")
            return search_query

        except Exception as e:
            logger.error(f"Fallback search planning failed: {e}")
            return self._create_fallback_query(query)

    def _get_search_query_json_schema(self) -> str:
        """
        Get JSON schema description for manual search query parsing
        """
        return """{
  "query_text": "string - Original query text",
  "intent": "string - Interpreted intent of the query",
  "entity_filters": ["array of strings - Specific entities to search for"],
  "category_filters": ["array of strings - Memory categories: fact, preference, skill, context, rule"],
  "time_range": "string or null - Time range for search (e.g., last_week)",
  "min_importance": "number - Minimum importance score (0.0-1.0)",
  "search_strategy": ["array of strings - Recommended search strategies"],
  "expected_result_types": ["array of strings - Expected types of results"]
}"""

    def _create_search_query_from_dict(
        self, data: dict[str, Any], original_query: str
    ) -> MemorySearchQuery:
        """
        Create MemorySearchQuery from dictionary with proper validation and defaults
        """
        try:
            # Import here to avoid circular imports
            from ..utils.pydantic_models import MemoryCategoryType

            # Validate and convert category filters
            category_filters = []
            raw_categories = data.get("category_filters", [])
            if isinstance(raw_categories, list):
                for cat_str in raw_categories:
                    try:
                        category = MemoryCategoryType(cat_str.lower())
                        category_filters.append(category)
                    except ValueError:
                        logger.debug(f"Invalid category filter '{cat_str}', skipping")

            # Create search query object with proper validation
            search_query = MemorySearchQuery(
                query_text=data.get("query_text", original_query),
                intent=data.get("intent", "General search (fallback)"),
                entity_filters=data.get("entity_filters", []),
                category_filters=category_filters,
                time_range=data.get("time_range"),
                min_importance=max(
                    0.0, min(1.0, float(data.get("min_importance", 0.0)))
                ),
                search_strategy=data.get("search_strategy", ["keyword_search"]),
                expected_result_types=data.get("expected_result_types", ["any"]),
            )

            return search_query

        except Exception as e:
            logger.error(f"Error creating search query from dict: {e}")
            return self._create_fallback_query(original_query)

    def _execute_semantic_search(
        self, query: str, db_manager, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """
        Execute semantic search using embedding similarity.

        Uses the EmbeddingSearchEngine to find semantically similar memories
        even when exact keywords don't match.

        Args:
            query: Search query text
            db_manager: Database manager instance
            user_id: User identifier for multi-tenant isolation
            limit: Maximum results to return

        Returns:
            List of memory dicts with similarity_score field added
        """
        try:
            # Initialize embedding engine lazily
            if not hasattr(self, '_embedding_engine') or self._embedding_engine is None:
                self._embedding_engine = EmbeddingSearchEngine(
                    use_local=True,
                    openai_client=self.client,
                    similarity_threshold=0.4,
                )

            # Fetch candidate memories from database
            try:
                from ..database.search_service import SearchService
                db_type = self._detect_database_type(db_manager)

                with db_manager.SessionLocal() as session:
                    search_service = SearchService(session, db_type)
                    # Get a larger candidate pool for semantic ranking
                    candidates = search_service.search_memories(
                        query="", user_id=user_id, limit=limit * 5
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch candidates for semantic search: {e}")
                candidates = []

            if not candidates:
                return []

            # Use embedding engine to rank by semantic similarity
            ranked_results = self._embedding_engine.search(
                query=query,
                memories=candidates,
                content_field="content",
                limit=limit,
            )

            # Enrich results with similarity metadata
            for result in ranked_results:
                result["search_strategy"] = "semantic_search"
                result["search_reasoning"] = (
                    f"Semantic similarity match (score: {result.get('similarity_score', 'N/A'):.3f})"
                )

            logger.debug(
                f"Semantic search for '{query}': {len(ranked_results)} results "
                f"(from {len(candidates)} candidates)"
            )
            return ranked_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            return []

    def _execute_importance_search(
        self, search_plan: MemorySearchQuery, db_manager, user_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Execute importance-based search with SQL-level filtering."""
        min_importance = max(
            search_plan.min_importance, 0.7
        )  # Default to high importance

        try:
            # Try SQL-level filtering first for performance
            from ..database.search_service import SearchService
            db_type = self._detect_database_type(db_manager)

            with db_manager.SessionLocal() as session:
                search_service = SearchService(session, db_type)
                all_results = search_service.search_memories(
                    query="", user_id=user_id, limit=limit * 2
                )
        except Exception:
            all_results = db_manager.search_memories(
                query="", user_id=user_id, limit=limit * 2
            )

        high_importance_results = [
            result
            for result in all_results
            if isinstance(result, dict) and result.get("importance_score", 0) >= min_importance
        ]

        return high_importance_results[:limit]

    def _create_fallback_query(self, query: str) -> MemorySearchQuery:
        """Create a fallback search query for error cases"""
        return MemorySearchQuery(
            query_text=query,
            intent="General search (fallback)",
            entity_filters=[word for word in query.split() if len(word) > 2],
            search_strategy=["keyword_search", "general_search"],
            expected_result_types=["any"],
        )

    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self._query_cache.items()
            if current_time - timestamp >= self._cache_ttl
        ]
        for key in expired_keys:
            del self._query_cache[key]

    async def execute_search_async(
        self, query: str, db_manager, user_id: str = "default", limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Async version of execute_search for better performance in background processing
        """
        try:
            # Run search planning in background if needed
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            search_plan = await loop.run_in_executor(
                self._background_executor, self.plan_search, query
            )

            # Execute searches concurrently
            search_tasks = []

            # Keyword search task
            if (
                search_plan.entity_filters
                or "keyword_search" in search_plan.search_strategy
            ):
                search_tasks.append(
                    loop.run_in_executor(
                        self._background_executor,
                        self._execute_keyword_search,
                        search_plan,
                        db_manager,
                        user_id,
                        limit,
                    )
                )

            # Category search task
            if (
                search_plan.category_filters
                or "category_filter" in search_plan.search_strategy
            ):
                search_tasks.append(
                    loop.run_in_executor(
                        self._background_executor,
                        self._execute_category_search,
                        search_plan,
                        db_manager,
                        user_id,
                        limit,
                    )
                )

            # Execute all searches concurrently
            if search_tasks:
                results_lists = await asyncio.gather(
                    *search_tasks, return_exceptions=True
                )

                all_results = []
                seen_memory_ids = set()

                for i, results in enumerate(results_lists):
                    if isinstance(results, Exception):
                        logger.warning(f"Search task {i} failed: {results}")
                        continue

                    for result in results:
                        if (
                            isinstance(result, dict)
                            and result.get("memory_id") not in seen_memory_ids
                        ):
                            seen_memory_ids.add(result["memory_id"])
                            all_results.append(result)

                return all_results[:limit]

            # Fallback to sync execution
            return self.execute_search(query, db_manager, user_id, limit)

        except Exception as e:
            logger.error(f"Async search execution failed: {e}")
            return []

    def execute_search_background(
        self,
        query: str,
        db_manager,
        user_id: str = "default",
        limit: int = 10,
        callback=None,
    ):
        """
        Execute search in background thread for non-blocking operation

        Args:
            query: Search query
            db_manager: Database manager
            user_id: User identifier for multi-tenant isolation
            limit: Max results
            callback: Optional callback function to handle results
        """

        def _background_search():
            try:
                results = self.execute_search(query, db_manager, user_id, limit)
                if callback:
                    callback(results)
                return results
            except Exception as e:
                logger.error(f"Background search failed: {e}")
                if callback:
                    callback([])
                return []

        # Start background thread
        thread = threading.Thread(target=_background_search, daemon=True)
        thread.start()
        return thread

    def search_memories(
        self, query: str, max_results: int = 5, user_id: str = "default"
    ) -> list[dict[str, Any]]:
        """
        Simple search interface for compatibility with memory tools

        Args:
            query: Search query
            max_results: Maximum number of results
            user_id: User identifier for multi-tenant isolation

        Returns:
            List of memory search results
        """
        # This is a compatibility method that uses the database manager directly
        # We'll need the database manager to be injected or passed
        # For now, return empty list and log the issue with parameters
        logger.warning(
            f"search_memories called without database manager: query='{query}', "
            f"max_results={max_results}, user_id='{user_id}'"
        )
        return []


def create_retrieval_agent(
    memori_instance=None, api_key: str = None, model: str = "gpt-4o"
) -> MemorySearchEngine:
    """
    Create a retrieval agent instance

    Args:
        memori_instance: Optional Memori instance for direct database access
        api_key: OpenAI API key
        model: Model to use for query planning

    Returns:
        MemorySearchEngine instance
    """
    agent = MemorySearchEngine(api_key=api_key, model=model)
    if memori_instance:
        agent._memori_instance = memori_instance
    return agent


def smart_memory_search(query: str, memori_instance, limit: int = 5) -> str:
    """
    Direct string-based memory search function that uses intelligent retrieval

    Args:
        query: Search query string
        memori_instance: Memori instance with database access
        limit: Maximum number of results

    Returns:
        Formatted string with search results
    """
    try:
        # Create search engine
        search_engine = MemorySearchEngine()

        # Execute intelligent search
        results = search_engine.execute_search(
            query=query,
            db_manager=memori_instance.db_manager,
            user_id=memori_instance.user_id,
            limit=limit,
        )

        if not results:
            return f"No relevant memories found for query: '{query}'"

        # Format results as a readable string
        output = f"Smart Memory Search Results for: '{query}'\n\n"

        for i, result in enumerate(results, 1):
            try:
                # Try to parse processed data for better formatting
                if "processed_data" in result:
                    import json

                    processed_data = result["processed_data"]
                    # Handle both dict and JSON string formats
                    if isinstance(processed_data, str):
                        processed_data = json.loads(processed_data)
                    elif isinstance(processed_data, dict):
                        pass  # Already a dict, use as-is
                    else:
                        # Fallback to basic result fields
                        summary = result.get(
                            "summary",
                            result.get("searchable_content", "")[:100] + "...",
                        )
                        category = result.get("category_primary", "unknown")
                        continue

                    summary = processed_data.get("summary", "")
                    category = processed_data.get("category", {}).get(
                        "primary_category", ""
                    )
                else:
                    summary = result.get(
                        "summary", result.get("searchable_content", "")[:100] + "..."
                    )
                    category = result.get("category_primary", "unknown")

                importance = result.get("importance_score", 0.0)
                created_at = result.get("created_at", "")
                search_strategy = result.get("search_strategy", "unknown")
                search_reasoning = result.get("search_reasoning", "")

                output += f"{i}. [{category.upper()}] {summary}\n"
                output += f"   Importance: {importance:.2f} | Created: {created_at}\n"
                output += f"   Strategy: {search_strategy}\n"

                if search_reasoning:
                    output += f"   Reason: {search_reasoning}\n"

                output += "\n"

            except Exception:
                # Fallback formatting
                content = result.get("searchable_content", "Memory content available")[
                    :100
                ]
                output += f"{i}. {content}...\n\n"

        return output.strip()

    except Exception as e:
        logger.error(f"Smart memory search failed: {e}")
        return f"Error in smart memory search: {str(e)}"
