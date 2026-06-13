"""
Dependency manifest - declares required vs optional dependencies.
Service fails fast on missing required dependencies.
"""

import os
import sys
from loguru import logger
from core.graceful_degradation import with_fallback

REQUIRED_DEPENDENCIES = [
    ("fastapi", "FastAPI web framework"),
    ("pydantic", "Data validation"),
    ("uvicorn", "ASGI server"),
]

OPTIONAL_DEPENDENCIES = {
    "ENABLE_RAG": [("chromadb", "Vector database for RAG")],
    "ENABLE_SPACY": [("spacy", "NLP entity extraction")],
    "ENABLE_TORCH": [("torch", "ML model inference")],
}



def validate_dependencies():
    """Validate all required dependencies are available."""
    missing = []

    for module_name, description in REQUIRED_DEPENDENCIES:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(f"{module_name} ({description})")

    if missing:
        logger.critical(f"Missing required dependencies: {missing}")
        logger.critical("Service cannot start. Install dependencies and retry.")
        sys.exit(1)

    logger.info("All required dependencies validated")


def get_enabled_features() -> dict:
    """Return dict of feature flags based on env vars and available deps."""
    features = {}

    for flag, deps in OPTIONAL_DEPENDENCIES.items():
        enabled = os.getenv(flag, "true").lower() == "true"
        if enabled:
            available = all(_check_import(mod) for mod, _ in deps)
            features[flag] = available
            if enabled and not available:
                logger.warning(f"{flag} enabled but dependencies missing")
        else:
            features[flag] = False

    return features


def _check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception as e:
        logger.warning(f"Failed to import optional dependency {module_name}: {e}")
        return False


def check_optional_dependency(module_name: str) -> bool:
    """Public wrapper for _check_import."""
    return _check_import(module_name)


# ============================================================================
# DEPENDENCY INJECTION CONTAINER (Phase 2 - Task 14)
# ============================================================================
# Provides centralized service wiring and management.
# Replaces scattered singleton get_instance() calls with explicit DI.

import logging
import threading
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.rag_config import RAGConfig
    from rag.embedding import RemoteEmbeddingService
    from rag.store.vector_store import InMemoryVectorStore

logger_di = logging.getLogger(__name__)


class DIContainer:
    """
    Dependency Injection Container for RAG System.
    
    Manages singleton instances of all services and provides factory methods
    for creating fully-wired service combinations.
    
    All services are lazily initialized on first access and cached for
    subsequent use. Thread-safe singleton pattern ensures one instance
    per service globally.
    
    Usage:
        container = DIContainer.get_instance()
        config = container.config
        orchestrator = container.create_unified_orchestrator()
    """
    
    _instance: Optional["DIContainer"] = None
    _lock = threading.RLock()
    
    def __new__(cls) -> "DIContainer":
        """Ensure only one DIContainer instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "DIContainer":
        """Get the global DIContainer singleton instance."""
        instance = cls()
        if not instance._initialized:
            instance.initialize()
        return instance
    
    def __init__(self):
        # Prevent re-initialization of singleton
        if hasattr(self, '_service_cache'):
            return
            
        self._config = None
        self._loader = None
        self._embeddings = None
        self._vector_store = None
        self._chunker = None
        self._drug_dict = None
        self._prompt_builder = None
        self._pii_scrubber = None
        self._feedback_store = None
        self._storage = None
        self._reranker = None
        self._service_cache: Dict[str, Any] = {}
        self._llm_gateway = None
        self._memory_manager = None
        self._interaction_checker = None
        self._sql_tool = None
        self._memori_bridge = None
    
    def initialize(self) -> None:
        """Initialize all service dependencies."""
        if self._initialized:
            return
        
        logger_di.info("Initializing DIContainer...")
        
        try:
            # Service caches initialized in __init__
            self._initialized = True
            logger_di.info("✅ DIContainer initialized successfully")
            
        except Exception as e:
            logger_di.error(f"Failed to initialize DIContainer: {e}")
            self._initialized = False
            raise
    
    # ========== PROPERTY GETTERS (LAZY INITIALIZATION) ==========
    
    @property
    def config(self) -> "RAGConfig":
        """Get RAGConfig singleton."""
        if self._config is None:
            from core.config.rag_config import RAGConfig
            self._config = RAGConfig()
            logger_di.debug("RAGConfig instance created and cached")
        return self._config
    
    @property
    def loader(self):
        """KnowledgeLoader — ARCHIVED (ingestion mode only)."""
        if self._loader is None:
            logger_di.warning(
                "KnowledgeLoaderSingleton is archived (ingestion mode only). "
                "Restore from _archive_ingestion/ if re-ingestion is needed."
            )
        return self._loader
    
    @property
    def embeddings(self):
        """Get EmbeddingService singleton."""
        if self._embeddings is None:
            use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
            if use_remote:
                from rag.embedding.remote import RemoteEmbeddingService
                self._embeddings = RemoteEmbeddingService.get_instance()
                logger_di.debug("RemoteEmbeddingService instance created and cached")
            else:
                logger_di.info("ℹ️ Remote embeddings disabled (USE_REMOTE_EMBEDDINGS=false)")
                self._embeddings = None
        return self._embeddings
    
    @property
    def vector_store(self):
        """Get VectorStore singleton (with fallback to InMemoryVectorStore)."""
        if self._vector_store is None:
            from rag.store.vector_store import get_vector_store
            
            # graceful fallback if vector store fails
            @with_fallback(default=None)
            def _init_store():
                embedding_model = getattr(self.config, 'embedding_model_name', "all-MiniLM-L6-v2")
                return get_vector_store(
                    config=self.config,
                    embedding_model=embedding_model
                )
            
            try:
                self._vector_store = _init_store()
                if self._vector_store is None:
                     # If fallback triggered or init returned None
                     from rag.store.vector_store import InMemoryVectorStore
                     self._vector_store = InMemoryVectorStore()
                     logger_di.warning("VectorStore init failed (fallback triggered), using InMemoryVectorStore")
            except Exception as e:
                logger_di.warning(f"VectorStore initialization failed: {e}, using InMemoryVectorStore")
                from rag.store.vector_store import InMemoryVectorStore
                self._vector_store = InMemoryVectorStore()
            logger_di.debug(f"VectorStore instance created ({type(self._vector_store).__name__}) and cached")
        return self._vector_store
    
    @property
    def chunker(self):
        """UnifiedMedicalChunker — ARCHIVED (ingestion mode only)."""
        if self._chunker is None:
            logger_di.warning(
                "UnifiedMedicalChunker is archived (ingestion mode only). "
                "Restore from _archive_ingestion/ if re-ingestion is needed."
            )
        return self._chunker
    
    @property
    def drug_dict(self):
        """DrugDictionary — ARCHIVED (ingestion mode only)."""
        if self._drug_dict is None:
            logger_di.warning(
                "DrugDictionary is archived (ingestion mode only). "
                "Restore from _archive_ingestion/ if re-ingestion is needed."
            )
        return self._drug_dict
    
    @property
    def prompt_builder(self):
        """Get PromptBuilder singleton."""
        if self._prompt_builder is None:
            try:
                from prompt_builder import PromptBuilder  # type: ignore
                self._prompt_builder = PromptBuilder()
                logger_di.debug("PromptBuilder instance created and cached")
            except ImportError:
                logger_di.warning("PromptBuilder module not found")
                self._prompt_builder = None
        return self._prompt_builder
    
    @property
    def pii_scrubber(self):
        """Get PIIScrubber singleton."""
        if self._pii_scrubber is None:
            try:
                from core.compliance.pii_scrubber_v2 import EnhancedPIIScrubber
                self._pii_scrubber = EnhancedPIIScrubber()
                logger_di.debug("EnhancedPIIScrubber instance created and cached")
            except (ImportError, ModuleNotFoundError) as e:
                logger_di.warning(f"PIIScrubber module not found or initialization failed: {e}")
                self._pii_scrubber = None
        return self._pii_scrubber
    
    @property
    def storage(self):
        """Get FeedbackStorage implementation singleton."""
        if self._storage is None:
            try:
                from core.database.storage_interface import FeedbackStorage
                from core.config.app_config import get_app_config
                
                config = get_app_config()
                db_backend = getattr(getattr(config, 'database', None), 'backend', 'inmemory')
                
                # Create appropriate storage implementation based on config
                if db_backend == "postgres":
                    try:
                        from core.database.postgres_feedback_storage import PostgresFeedbackStorage
                        self._storage = PostgresFeedbackStorage(config=config.database)
                    except ImportError:
                        logger_di.warning("Postgres storage not available, falling back to in-memory")
                        db_backend = "inmemory"
                
                # Fallback to in-memory storage
                if db_backend == "inmemory" or self._storage is None:
                    from core.database.inmemory_feedback_storage import InMemoryFeedbackStorage
                    self._storage = InMemoryFeedbackStorage()
                    logger_di.info("Using InMemoryFeedbackStorage (data will not persist)")
                else:
                    logger_di.info(f"FeedbackStorage instance created ({db_backend}) and cached")
            except Exception as e:
                logger_di.error(f"Failed to initialize FeedbackStorage: {e}")
                # Last resort fallback
                try:
                    from core.database.inmemory_feedback_storage import InMemoryFeedbackStorage
                    self._storage = InMemoryFeedbackStorage()
                    logger_di.warning("Using InMemoryFeedbackStorage as fallback after error")
                except Exception as e2:
                    logger_di.error(f"Could not create fallback storage: {e2}")
                    raise
        
        return self._storage
    
    @property
    def reranker(self):
        """Get MedicalReranker singleton (RAGFlow-enhanced production-ready reranker)."""
        if self._reranker is None:
            try:
                from rag.retrieval.reranker import MedicalReranker
                
                # Initialize with production settings from RAG config
                max_length = getattr(self.config, 'reranker_max_length', 512)
                batch_size = getattr(self.config, 'reranker_batch_size', 32)
                model_name = getattr(self.config, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                
                # Detect GPU availability for device selection
                device = "cpu"  # Default to CPU
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                        logger_di.info("CUDA available - reranker will use GPU")
                except ImportError:
                    logger_di.debug("PyTorch not available - reranker will use CPU")
                
                self._reranker = MedicalReranker(
                    model_name=model_name,
                    max_length=max_length,
                    batch_size=batch_size,
                    device=device
                )
                logger_di.info(
                    f"✓ MedicalReranker instance created and cached\n"
                    f"  - Model: {model_name}\n"
                    f"  - Max Length: {max_length}\n"
                    f"  - Batch Size: {batch_size}\n"
                    f"  - Device: {device}"
                )
            except Exception as e:
                logger_di.warning(f"MedicalReranker initialization failed: {e}. Reranking will be disabled.")
                self._reranker = None
        
        return self._reranker

    @property
    def llm_gateway(self):
        """Get LLMGateway singleton."""
        if not hasattr(self, '_llm_gateway') or self._llm_gateway is None:
            try:
                from core.llm.llm_gateway import LLMGateway
                self._llm_gateway = LLMGateway()
                logger_di.debug("LLMGateway instance created and cached")
            except Exception as e:
                logger_di.warning(f"LLMGateway initialization failed: {e}")
                self._llm_gateway = None
        return self._llm_gateway

    @property
    def memory_manager(self):
        """Get MemoryManager singleton."""
        if not hasattr(self, '_memory_manager') or self._memory_manager is None:
            try:
                from memori.memory_manager import MemoryManager
                self._memory_manager = MemoryManager()
                logger_di.debug("MemoryManager instance created and cached")
            except Exception as e:
                logger_di.warning(f"MemoryManager initialization failed: {e}")
                self._memory_manager = None
        return self._memory_manager

    @property
    def interaction_checker(self):
        """Get GraphInteractionChecker singleton."""
        if not hasattr(self, '_interaction_checker') or self._interaction_checker is None:
            try:
                from rag.knowledge_graph.interaction_checker import GraphInteractionChecker
                self._interaction_checker = GraphInteractionChecker(
                    postgres_db=self.postgres_db  # Pass shared PostgreSQL pool
                )
                logger_di.debug("GraphInteractionChecker instance created and cached")
            except Exception as e:
                logger_di.warning(f"GraphInteractionChecker initialization failed: {e}")
                self._interaction_checker = None
        return self._interaction_checker
    
    async def initialize_interaction_checker(self):
        """Initialize the interaction checker's PostgreSQL fallback (call during startup)."""
        checker = self.interaction_checker
        if checker:
            try:
                await checker.initialize_fallback()
                logger_di.info("✅ GraphInteractionChecker PostgreSQL fallback initialized")
            except Exception as e:
                logger_di.warning(f"GraphInteractionChecker fallback init failed: {e}")

    @property
    def memori_bridge(self):
        """Get MemoriRAGBridge singleton for unified Memori-RAG access."""
        if not hasattr(self, '_memori_bridge') or self._memori_bridge is None:
            try:
                from memori.core.memory import Memori
                from rag.memory.memori_integration import MemoriRAGBridge
                from core.config.app_config import get_app_config
                
                # Build PostgreSQL connection string from AppConfig
                app_config = get_app_config()
                db_cfg = app_config.database
                connection_string = (
                    f"postgresql://{db_cfg.user}:{db_cfg.password}@"
                    f"{db_cfg.host}:{db_cfg.port}/memori"
                )
                
                # Create Memori instance with PostgreSQL
                memori = Memori(
                    database_connect=connection_string,
                    user_id="system",
                    verbose=False
                )
                
                # Enable Memori so it can record conversations
                memori.enable()
                logger_di.info("✅ Memori enabled for conversation recording")
                
                # Create MemoriRAGBridge with both services
                self._memori_bridge = MemoriRAGBridge(
                    memori=memori,
                    vector_store=self.vector_store
                )
                
                logger_di.info(f"✅ MemoriRAGBridge created with PostgreSQL at {db_cfg.host}:{db_cfg.port}")
            except Exception as e:
                logger_di.warning(f"MemoriRAGBridge initialization failed: {e}")
                self._memori_bridge = None
        return self._memori_bridge

    @property
    def sql_tool(self):
        """
        Get TextToSQLTool singleton.
        
        **IMPORTANT**: This is initialized via async_initialize() during app startup,
        NOT in the property getter. The property simply returns the cached instance.
        
        Why: In FastAPI, event loops are always running. Trying to call
        loop.run_until_complete() from a property causes None to be returned,
        breaking the Text-to-SQL tool.
        
        Proper initialization happens in main.py:
            await container.async_initialize()
        """
        if not hasattr(self, '_sql_tool') or self._sql_tool is None:
            logger_di.error(
                "TextToSQLTool not initialized! "
                "Call await container.async_initialize() during app startup in main.py"
            )
        return self._sql_tool
    
    async def async_initialize(self) -> None:
        """
        Async initialization for services that require event loop access.
        
        Call this from FastAPI app startup event:
            @app.on_event("startup")
            async def startup():
                container = DIContainer.get_instance()
                await container.async_initialize()
        """
        try:
            from tools.text_to_sql_tool import TextToSQLTool
            from core.database.postgres_db import get_database
            import asyncio
            
            # Safe async initialization
            db = await get_database()
            self._sql_tool = TextToSQLTool(db=db, llm_gateway=self.llm_gateway)
            logger_di.info("✅ TextToSQLTool initialized via async_initialize()")
            
            # Connection pool warmup: Execute lightweight query to establish connections
            # This reduces first-query latency by ~200-500ms
            try:
                if hasattr(db, 'execute_query'):
                    # PostgresDatabase uses execute_query, check if it works for warmup
                    # It expects a query string.
                    await asyncio.wait_for(
                        db.execute_query("SELECT 1"),
                        timeout=5.0
                    )
                    logger_di.info("✅ Database connection pool warmed up")
            except asyncio.TimeoutError:
                logger_di.warning("⚠️ DB warmup timed out (connections will be lazy)")
            except Exception as warmup_error:
                logger_di.debug(f"DB warmup skipped: {warmup_error}")
            
        except Exception as e:
            logger_di.error(f"Failed to async initialize TextToSQLTool: {e}")
            self._sql_tool = None
    
    def get_feedback_store(self):
        """
        Factory method to create FeedbackStore with StorageInterface injection.
        
        Returns:
            FeedbackStore: Fully wired feedback store with database abstraction
            
        Example:
            store = DIContainer.get_instance().get_feedback_store()
            await store.record_feedback(...)
        """
        from rag.store.feedback_store import FeedbackStore
        
        try:
            storage = self.storage
            store = FeedbackStore(storage=storage)
            logger_di.info("✓ FeedbackStore created with StorageInterface injection")
            return store
        except Exception as e:
            logger_di.error(f"Failed to create FeedbackStore: {e}")
            raise
    
    @property
    def postgres_db(self):
        """Get PostgreSQL database instance (from service cache)."""
        # Return db_manager directly — circuit breaker.call() is async
        # and cannot be used in a synchronous property getter.
        try:
            return self.get_service('db_manager')
        except Exception as e:
            logger_di.error(f"PostgreSQL access failed: {e}")
            return None
    

    @property
    def db(self):
        """Get database instance (generic alias for postgres_db)."""
        return self.get_service('db_manager')

    @property
    def redis_client(self):
        """Get Redis client instance (if registered during startup)."""
        try:
            client = self.get_service('redis_client')
            return client
        except Exception:
            return None

    @property
    def embedding_service(self):
        """Get embedding service (alias for embeddings property)."""
        return self.embeddings
    
    # Backward compatibility class method
    @classmethod
    def get_feedback_store_static(cls):
        """
        Static method for backward compatibility.
        Use get_instance().get_feedback_store() instead.
        """
        return cls.get_instance().get_feedback_store()
    
    # ========== FACTORY METHODS ==========
    

    
    @property
    def memory_optimizer(self):
        """Get the memory query optimizer instance."""
        if not hasattr(self, '_memory_optimizer') or self._memory_optimizer is None:
            try:
                db_manager = self.get_service('db_manager')
                # Need initialized DB with pool
                if db_manager and hasattr(db_manager, 'pool') and db_manager.pool:
                    with self._lock:
                        if not hasattr(self, '_memory_optimizer') or self._memory_optimizer is None:
                            from memori.memory_query_optimizer import OptimizedMemoryQueries, MemoryOptimizationConfig
                            
                            # Create default config
                            config = MemoryOptimizationConfig()
                            
                            self._memory_optimizer = OptimizedMemoryQueries(
                                config=config,
                                db_pool=db_manager.pool,
                                redis_client=self.redis_client
                            )
                            logger_di.info("✅ Memory optimizer initialized")
                else:
                    logger_di.warning("⚠️ Cannot initialize memory optimizer: DB pool not available")
                    return None
            except ImportError as e:
                logger_di.warning(f"⚠️ Memory optimizer import failed: {e}")
                return None
            except Exception as e:
                logger_di.error(f"❌ Memory optimizer initialization failed: {e}")
                return None
                
        return self._memory_optimizer

    @property
    def rag_optimizer(self):
        """Get the RAG query optimizer instance."""
        if not hasattr(self, '_rag_optimizer') or self._rag_optimizer is None:
            try:
                with self._lock:
                    if not hasattr(self, '_rag_optimizer') or self._rag_optimizer is None:
                        from rag.pipeline.query_optimizer import OptimizedRAGQueryExecutor
                        
                        redis_url = None
                        if self.redis_client:
                            try:
                                conn_kwargs = self.redis_client.connection_pool.connection_kwargs
                                host = conn_kwargs.get('host', 'localhost')
                                port = conn_kwargs.get('port', 6379)
                                db = conn_kwargs.get('db', 0)
                                redis_url = f"redis://{host}:{port}/{db}"
                            except Exception:
                                redis_url = "redis://localhost:6379/0"
                        self._rag_optimizer = OptimizedRAGQueryExecutor(
                            vector_store=self.vector_store,
                            embedding_service=self.embedding_service,
                            cache_config={"redis_url": redis_url} if redis_url else None
                        )
                        logger_di.info("✅ RAG optimizer initialized")
            except ImportError as e:
                logger_di.warning(f"⚠️ RAG optimizer import failed: {e}")
                return None
            except Exception as e:
                logger_di.error(f"❌ RAG optimizer initialization failed: {e}")
                return None
                
        return self._rag_optimizer
    
    def reset(self) -> None:
        """Reset all cached services (for testing only)."""
        logger_di.warning("Resetting DIContainer - all cached services cleared")
        self._config = None
        self._loader = None
        self._embeddings = None
        self._vector_store = None
        self._chunker = None
        self._drug_dict = None
        self._prompt_builder = None
        self._pii_scrubber = None
        self._feedback_store = None
        self._storage = None
        self._reranker = None
        self._service_cache.clear()
        self._llm_gateway = None
        self._memory_manager = None
        self._interaction_checker = None
        self._sql_tool = None
        self._memori_bridge = None

    def register_service(self, name: str, service: Any) -> None:
        """Register a service instance dynamically."""
        self._service_cache[name] = service
        logger_di.debug(f"Service registered: {name}")

    def get_service(self, name: str) -> Optional[Any]:
        """Get a registered service instance."""
        return self._service_cache.get(name)
    
    def __repr__(self) -> str:
        """String representation of DIContainer."""
        initialized_services = sum([
            self._config is not None,
            self._loader is not None,
            self._embeddings is not None,
            self._vector_store is not None,
            self._chunker is not None,
            self._drug_dict is not None,
            self._prompt_builder is not None,
            self._pii_scrubber is not None,
            self._reranker is not None,
            self._memori_bridge is not None,
        ])
        return f"DIContainer(initialized_services={initialized_services}/11)"


def reset_di_container() -> None:
    """Reset the global DIContainer singleton (for testing only)."""
    DIContainer._instance = None
    logger_di.debug("DIContainer singleton reset")


def get_di_container() -> DIContainer:
    """Get the global DIContainer singleton instance."""
    return DIContainer.get_instance()
