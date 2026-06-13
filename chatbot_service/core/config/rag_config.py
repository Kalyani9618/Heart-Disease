"""
RAG Configuration

Pydantic-based configuration for the RAG system to replace hardcoded magic numbers
and settings with validated configuration objects.

This module now includes path management through PathConfig, providing
centralized access to all filesystem paths used by the RAG system.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .rag_paths import PathConfig



class RAGConfig(BaseModel):
    """
    Configuration for the RAG system with validated fields.
    
    This class now includes a paths property that provides access to centralized
    filesystem path management through PathConfig.
    """
    # Retrieval settings
    medical_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of medical documents to retrieve"
    )
    memory_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of user memories to retrieve"
    )
    drug_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of drug information items to retrieve"
    )
    graph_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum depth for graph traversal"
    )
    
    # Relevance settings
    min_relevance_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for retrieval"
    )
    
    # Model settings
    embedding_model_name: str = Field(
        default="MedCPT-Query-Encoder",
        description="Name of the embedding model to use"
    )
    llm_model_name: str = Field(
        default="gemma3:1b",
        description="Name of the LLM model to use"
    )
    
    # Performance settings
    rerank_enabled: bool = Field(
        default=True,
        description="Whether to enable reranking of results"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable embedding caching"
    )
    
    # Batch processing settings
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embedding operations"
    )
    
    # Token budget settings
    max_context_length: int = Field(
        default=3000,
        ge=500,
        le=8000,
        description="Maximum context length for token budgeting"
    )
    
    model_config = ConfigDict(extra="forbid")
    
    @property
    def paths(self) -> "PathConfig":
        """
        Get the global PathConfig instance for centralized path management.
        
        All filesystem paths for the RAG system go through this property
        to ensure consistency and environment-based overrides.
        
        Returns:
            PathConfig instance with all path properties
            
        Example:
            config = RAGConfig()
            drug_dict = config.paths.get_drug_dictionary_file()
        """
        from .rag_paths import get_path_config
        return get_path_config()


def get_default_rag_config() -> RAGConfig:
    """
    Get a default RAG configuration.
    
    DEPRECATED: Use get_app_config() from core.config.app_config instead.
    This function is kept for backward compatibility.
    
    Returns:
        RAGConfig with default values
    """
    return RAGConfig()


def get_rag_config_from_app() -> RAGConfig:
    """
    Get RAG configuration from the unified AppConfig.
    
    This bridges the old RAGConfig API with the new unified AppConfig system.
    
    Returns:
        RAGConfig instance populated from AppConfig.rag settings
        
    Example:
        # OLD way (deprecated)
        config = RAGConfig()
        
        # NEW way (recommended)
        from core.config.app_config import get_app_config
        app_config = get_app_config()
        rag_settings = app_config.rag
    """
    try:
        from .app_config import get_app_config
        app_config = get_app_config()
        
        # Convert AppConfig.RAGConfig to old RAGConfig format
        return RAGConfig(
            medical_top_k=5,
            memory_top_k=5,
            drug_top_k=3,
            graph_depth=2,
            min_relevance_score=app_config.rag.min_relevance_score,
            embedding_model_name=app_config.rag.embedding_model_name,
            llm_model_name=app_config.llm.model_name,
            rerank_enabled=app_config.rag.rerank_enabled,
            cache_enabled=app_config.rag.cache_enabled,
            embedding_batch_size=32,
            max_context_length=app_config.rag.max_context_tokens,
        )
    except Exception as e:
        import logging
        logging.warning(f"Failed to load RAGConfig from AppConfig: {e}, using defaults")
        return RAGConfig()


def create_rag_config_from_dict(config_dict: dict) -> RAGConfig:
    """
    Create a RAG configuration from a dictionary.
    
    Args:
        config_dict: Dictionary with configuration values
        
    Returns:
        RAGConfig instance
    """
    return RAGConfig(**config_dict)
