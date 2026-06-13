"""
RAG Settings Singleton

Provides centralized, environment-aware configuration management for the RAG system.
Replaces ad-hoc default RAGConfig() initialization with enforced dependency injection.

Usage:
    # Get global settings
    settings = RAGSettings.get_instance()
    
    # OR pass config explicitly to RAGOrchestrator
    config = RAGSettings.get_instance().get_config()
    orchestrator = UnifiedRAGOrchestrator(rag_config=config)
"""


import logging
import os
from typing import Optional
from functools import lru_cache
from .rag_config import RAGConfig

logger = logging.getLogger(__name__)


class RAGSettings:
    """
    Singleton for centralized RAG configuration management.

    Ensures:
    - Single source of truth for all RAG settings
    - Environment-based configuration overrides
    - Fail-fast validation of required settings
    - No magic default values scattered across codebase
    """

    _instance: Optional["RAGSettings"] = None
    _config: Optional[RAGConfig] = None
    _initialized: bool = False

    def __new__(cls) -> "RAGSettings":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RAGSettings":
        """
        Get the singleton instance of RAGSettings.

        Returns:
            RAGSettings singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize settings from environment and defaults."""
        if self._initialized:
            return

        try:
            # Load from environment variables or use defaults
            self._config = RAGConfig(
                medical_top_k=int(os.getenv("RAG_MEDICAL_TOP_K", "5")),
                memory_top_k=int(os.getenv("RAG_MEMORY_TOP_K", "5")),
                drug_top_k=int(os.getenv("RAG_DRUG_TOP_K", "3")),
                graph_depth=int(os.getenv("RAG_GRAPH_DEPTH", "2")),
                min_relevance_score=float(os.getenv("RAG_MIN_RELEVANCE", "0.3")),
                embedding_model_name=os.getenv("RAG_EMBEDDING_MODEL", "MedCPT-Query-Encoder"),
                llm_model_name=os.getenv("RAG_LLM_MODEL", "gemma3:1b"),
                rerank_enabled=os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true",
                cache_enabled=os.getenv("RAG_CACHE_ENABLED", "true").lower() == "true",
                embedding_batch_size=int(os.getenv("RAG_BATCH_SIZE", "32")),
                max_context_length=int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "3000")),
            )

            logger.info(
                f"✅ RAGSettings initialized from environment:\n"
                f"  - Embedding Model: {self._config.embedding_model_name}\n"
                f"  - LLM Model: {self._config.llm_model_name}\n"
                f"  - Medical Top-K: {self._config.medical_top_k}\n"
                f"  - Reranking: {'Enabled' if self._config.rerank_enabled else 'Disabled'}\n"
                f"  - Caching: {'Enabled' if self._config.cache_enabled else 'Disabled'}"
            )

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize RAGSettings: {e}")
            raise

    def get_config(self) -> RAGConfig:
        """
        Get the RAG configuration.

        Returns:
            RAGConfig instance with validated settings

        Raises:
            RuntimeError: If settings not initialized
        """
        if not self._initialized or self._config is None:
            self._initialize()
        return self._config

    def update_config(self, **kwargs) -> None:
        """
        Update specific config values.

        Args:
            **kwargs: Config field names and values to update

        Example:
            settings = RAGSettings.get_instance()
            settings.update_config(medical_top_k=10, rerank_enabled=False)
        """
        if not self._initialized or self._config is None:
            self._initialize()

        try:
            # Create new config with updated values
            config_dict = self._config.dict()
            config_dict.update(kwargs)
            self._config = RAGConfig(**config_dict)
            logger.info(f"✅ RAGSettings updated: {kwargs}")
        except Exception as e:
            logger.error(f"Failed to update RAGSettings: {e}")
            raise

    def reset(self) -> None:
        """
        Reset settings to uninitialized state.

        Useful for testing and reinitializing with different environment.
        """
        self._initialized = False
        self._config = None
        logger.info("✅ RAGSettings reset")

    def to_dict(self) -> dict:
        """
        Get settings as dictionary.

        Returns:
            Dictionary representation of current config
        """
        if not self._initialized or self._config is None:
            self._initialize()
        return self._config.dict()

    def __repr__(self) -> str:
        """String representation of settings."""
        if self._config is None:
            return "<RAGSettings: not initialized>"
        return f"<RAGSettings: {self._config}>"


@lru_cache(maxsize=1)
def get_rag_settings() -> RAGSettings:
    """
    Get RAG settings singleton (cached).

    Returns:
        RAGSettings instance

    Example:
        settings = get_rag_settings()
        config = settings.get_config()
    """
    return RAGSettings.get_instance()
