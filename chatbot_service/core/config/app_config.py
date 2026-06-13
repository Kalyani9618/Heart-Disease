"""
Unified Application Configuration - Single Source of Truth

Pydantic-based configuration for the entire application:
- LLM Settings (provider, model, temperature, etc.)
- RAG Settings (embedding backend, weights, token budget)
- Database Settings (backend selection, connection details)
- API Settings (host, port, worker configuration)

Environment Variables (highest priority override):
    APP_ENV: 'development' | 'staging' | 'production'
    LLM_PROVIDER: 'ollama' | 'gemini' | 'openai'
    LLM_MODEL_NAME: 'gemma3:1b'
    OLLAMA_API_HOST: 'http://localhost:11434'
    RAG_ENABLED: 'true' | 'false'
    EMBEDDING_MODEL: 'all-MiniLM-L6-v2'
    EMBEDDING_BACKEND: 'onnx' | 'pytorch'
    VECTOR_WEIGHT: '0.5'
    GRAPH_WEIGHT: '0.35'
    MEMORY_WEIGHT: '0.15'
    DB_BACKEND: 'mysql' | 'postgres'
    
    
Usage:
    from core.config.app_config import AppConfig, get_app_config
    
    config = get_app_config()
    print(config.llm.model_name)      # 'gemma3:1b'
    print(config.rag.vector_weight)   # 0.5
    print(config.database.backend)    # 'mysql'
"""

import os
import logging
import threading
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMConfig(BaseModel):
    """MedGemma LLM settings (local medical model only).
    
    This configuration uses a local MedGemma server via OpenAI-compatible API.
    No cloud LLM providers (OpenRouter, Gemini, OpenAI) are used.
    
    MedGemma Variants:
    - medgemma-4b-it: Lightweight, suitable for most tasks
    - medgemma-27b-it: Full-featured, better multimodal support
    """
    
    provider: Literal["medgemma"] = Field(
        default="medgemma",
        description="LLM provider (MedGemma only - local medical model)"
    )
    model_name: str = Field(
        default="medgemma-4b-it",
        description="MedGemma model variant"
    )
    base_url: str = Field(
        default="http://127.0.0.1:8090/v1",
        description="MedGemma server endpoint (OpenAI-compatible API)"
    )
    api_key: str = Field(
        default="sk-no-key-required",
        description="API key (not required for local server)"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0=deterministic, 2=creative)"
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum tokens to generate in response"
    )
    timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Request timeout in seconds"
    )
    
    # Legacy field kept for backward compatibility (ignored)
    api_host: str = Field(
        default="http://127.0.0.1:8090/v1",
        description="Deprecated: Use base_url instead"
    )
    
    model_config = ConfigDict(extra="forbid")


class RAGConfig(BaseModel):
    """RAG system settings."""
    
    enabled: bool = Field(
        default=True,
        description="Enable/disable RAG system"
    )
    
    # Embedding settings
    embedding_model_name: str = Field(
        default="MedCPT-Query-Encoder",
        description="Embedding model to use"
    )
    embedding_backend: Literal["onnx", "pytorch", "huggingface", "remote"] = Field(
        default="remote",
        description="Embedding backend for performance tuning"
    )
    use_remote_embeddings: bool = Field(
        default=True,
        description=(
            "When True the factory will prefer the remote (Colab/ngrok) "
            "backend if COLAB_API_URL is set and reachable, before falling "
            "back to local backends."
        ),
    )
    
    # Chunking settings
    max_chunk_length: int = Field(
        default=512,
        ge=100,
        le=2048,
        description="Maximum chunk length for ingestion"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=256,
        description="Overlap between chunks"
    )
    
    # Retrieval settings
    top_k_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Top K results to retrieve"
    )
    min_relevance_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold"
    )
    
    # Reranking settings
    rerank_enabled: bool = Field(
        default=True,
        description="Enable cross-encoder reranking"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranker model name"
    )
    reranker_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for reranker"
    )
    
    # Weighting for context assembly (NOW FULLY CONFIGURABLE!)
    vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search results (0.0-1.0)"
    )
    graph_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for knowledge graph results (0.0-1.0)"
    )
    memory_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for memory/history results (0.0-1.0)"
    )
    
    # Knowledge graph confidence scoring
    graph_confidence_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Alpha for graph confidence weighting (trustworthiness)"
    )
    graph_confidence_beta: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Beta for graph confidence weighting (novelty)"
    )
    
    # Token budgeting
    max_context_tokens: int = Field(
        default=3000,
        ge=500,
        le=8000,
        description="Maximum context tokens for prompt"
    )
    
    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable embedding cache"
    )
    
    # Vector store paths (for consistency)
    paths: Optional[dict] = Field(
        default=None,
        description="Paths configuration for RAG storage"
    )
    
    model_config = ConfigDict(extra="forbid")


class DatabaseConfig(BaseModel):
    """Database settings."""
    
    backend: Literal["postgres"] = Field(
        default="postgres",
        description="Database backend (PostgreSQL only)"
    )
    host: str = Field(default="localhost", description="DB host")
    port: int = Field(default=5432, ge=1, le=65535, description="DB port")
    user: str = Field(default="root", description="DB user")
    password: str = Field(default="", description="DB password")
    database: str = Field(default="health_data", description="Database name")
    
    model_config = ConfigDict(extra="forbid")


class APIConfig(BaseModel):
    """API server settings."""
    
    host: str = Field(default="0.0.0.0", description="API bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="API bind port")
    reload: bool = Field(default=True, description="Hot reload in development")
    workers: int = Field(default=4, ge=1, le=16, description="Number of workers")
    
    model_config = ConfigDict(extra="forbid")


class SpaCyConfig(BaseModel):
    """SpaCy NLP settings."""
    model_name: str = Field(default="en_core_web_trf", description="SpaCy model name")
    batch_size: int = Field(default=100, description="Batch size for nlp.pipe")
    disable_parser: bool = Field(default=False, description="Disable parser component")
    enable_medical_ruler: bool = Field(default=True, description="Enable medical entity ruler")
    enable_negation_detection: bool = Field(default=True, description="Enable negation detection")
    load_from_disk: bool = Field(default=True, description="Load model from disk")
    use_gpu: bool = Field(default=False, description="Use GPU if available")
    
    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    """Complete application configuration - single source of truth."""
    
    # Environment
    env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        default=True,
        description="Debug logging enabled"
    )
    
    # Subsystems
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    spacy: SpaCyConfig = Field(default_factory=SpaCyConfig)
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("env", mode="before")
    @classmethod
    def validate_env(cls, v):
        if isinstance(v, str):
            return Environment(v)
        return v


# Global instance with thread safety
_global_app_config: Optional[AppConfig] = None
_config_lock = threading.RLock()


def get_app_config() -> AppConfig:
    """
    Get or create the global AppConfig instance.
    
    Loads from environment variables with smart defaults.
    Thread-safe singleton pattern.
    
    Returns:
        AppConfig singleton
        
    Example:
        config = get_app_config()
        print(config.llm.model_name)        # 'medgemma-4b-it'
        print(config.rag.vector_weight)     # 0.5
        print(config.database.backend)      # 'postgres'
    """
    global _global_app_config
    
    if _global_app_config is None:
        with _config_lock:
            if _global_app_config is None:
                # Load .env file if it exists
                load_dotenv()
                
                # Load from environment with defaults
                # MedGemma-only configuration (supports both MEDGEMMA_* and legacy LLAMA_LOCAL_* vars)
                env_dict = {
                    "env": os.environ.get("APP_ENV", "development"),
                    "debug": os.environ.get("APP_DEBUG", "true").lower() == "true",
                    "llm": {
                        "provider": "medgemma",  # Always MedGemma
                        "model_name": os.environ.get(
                            "MEDGEMMA_MODEL", 
                            os.environ.get("LLAMA_LOCAL_MODEL", "medgemma-4b-it")
                        ),
                        "base_url": os.environ.get(
                            "MEDGEMMA_BASE_URL", 
                            os.environ.get("LLAMA_LOCAL_BASE_URL", "http://127.0.0.1:8090/v1")
                        ),
                        "api_host": os.environ.get(
                            "MEDGEMMA_BASE_URL", 
                            os.environ.get("LLAMA_LOCAL_BASE_URL", "http://127.0.0.1:8090/v1")
                        ),
                        "api_key": os.environ.get(
                            "MEDGEMMA_API_KEY", 
                            os.environ.get("LLAMA_LOCAL_API_KEY", "sk-no-key-required")
                        ),
                        "temperature": float(os.environ.get(
                            "MEDGEMMA_TEMPERATURE", 
                            os.environ.get("LLAMA_LOCAL_TEMPERATURE", "0.3")
                        )),
                        "max_tokens": int(os.environ.get(
                            "MEDGEMMA_MAX_TOKENS", 
                            os.environ.get("LLAMA_LOCAL_MAX_TOKENS", "2048")
                        )),
                        "timeout_seconds": int(os.environ.get("LLM_TIMEOUT", "60")),
                    },
                    "rag": {
                        "enabled": os.environ.get("RAG_ENABLED", "true").lower() == "true",
                        "embedding_model_name": os.environ.get("EMBEDDING_MODEL", "MedCPT-Query-Encoder"),
                        "embedding_backend": os.environ.get("EMBEDDING_BACKEND", "remote"),
                        "use_remote_embeddings": os.environ.get("USE_REMOTE_EMBEDDINGS", "true").lower() == "true",
                        "max_chunk_length": int(os.environ.get("MAX_CHUNK_LENGTH", "512")),
                        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "50")),
                        "top_k_results": int(os.environ.get("TOP_K_RESULTS", "5")),
                        "min_relevance_score": float(os.environ.get("MIN_RELEVANCE_SCORE", "0.6")),
                        "rerank_enabled": os.environ.get("RERANK_ENABLED", "true").lower() == "true",
                        "reranker_model": os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                        "reranker_batch_size": int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
                        "vector_weight": float(os.environ.get("VECTOR_WEIGHT", "0.5")),
                        "graph_weight": float(os.environ.get("GRAPH_WEIGHT", "0.35")),
                        "memory_weight": float(os.environ.get("MEMORY_WEIGHT", "0.15")),
                        "graph_confidence_alpha": float(os.environ.get("GRAPH_CONFIDENCE_ALPHA", "0.7")),
                        "graph_confidence_beta": float(os.environ.get("GRAPH_CONFIDENCE_BETA", "0.3")),
                        "max_context_tokens": int(os.environ.get("MAX_CONTEXT_TOKENS", "3000")),
                        "cache_enabled": os.environ.get("CACHE_ENABLED", "true").lower() == "true",
                        "paths": {
                            # Vector storage uses ChromaDB (see rag/chromadb_store.py)
                            "knowledge_graph_dir": os.environ.get("KNOWLEDGE_GRAPH_DIR", "./kg_data"),
                        },
                    },
                    "database": {
                        "backend": os.environ.get("DB_BACKEND", "postgres"),
                        "host": os.environ.get("DB_HOST", "localhost"),
                        "port": int(os.environ.get("DB_PORT", "5432")),
                        "user": os.environ.get("DB_USER", "postgres"),
                        "password": os.environ.get("DB_PASSWORD", ""),
                        "database": os.environ.get("DB_NAME", "heartguard"),
                    },
                    "api": {
                        "host": os.environ.get("API_HOST", "0.0.0.0"),
                        "port": int(os.environ.get("API_PORT", "8000")),
                        "reload": os.environ.get("API_RELOAD", "true").lower() == "true",
                        "workers": int(os.environ.get("API_WORKERS", "4")),
                    },
                    "spacy": {
                        "model_name": os.environ.get("SPACY_MODEL", "en_core_web_trf"),
                        "batch_size": int(os.environ.get("SPACY_BATCH_SIZE", "100")),
                        "disable_parser": os.environ.get("SPACY_DISABLE_PARSER", "false").lower() == "true",
                        "enable_medical_ruler": os.environ.get("SPACY_ENABLE_MEDICAL_RULER", "true").lower() == "true",
                        "enable_negation_detection": os.environ.get("SPACY_ENABLE_NEGATION_DETECTION", "true").lower() == "true",
                        "load_from_disk": os.environ.get("SPACY_LOAD_FROM_DISK", "true").lower() == "true",
                        "use_gpu": os.environ.get("SPACY_USE_GPU", "false").lower() == "true",
                    },
                }
                
                try:
                    _global_app_config = AppConfig(**env_dict)
                    logger.info(
                        f"✅ AppConfig loaded: env={_global_app_config.env}, "
                        f"rag_enabled={_global_app_config.rag.enabled}, "
                        f"llm={_global_app_config.llm.provider}, "
                        f"db={_global_app_config.database.backend}:{_global_app_config.database.port}, "
                        f"weights=(v:{_global_app_config.rag.vector_weight:.2f}, "
                        f"g:{_global_app_config.rag.graph_weight:.2f}, "
                        f"m:{_global_app_config.rag.memory_weight:.2f})"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to load AppConfig: {e}")
                    raise
    
    return _global_app_config


def reset_app_config():
    """Reset global config (for testing)."""
    global _global_app_config
    with _config_lock:
        _global_app_config = None
