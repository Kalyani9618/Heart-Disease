"""
Configuration module for the RAG system.

Provides:
- RAGConfig: Data class for RAG configuration
- RAGSettings: Singleton for centralized settings management
- Environment-based configuration overrides
"""

from .rag_config import RAGConfig, get_default_rag_config
from .rag_settings import RAGSettings, get_rag_settings

__all__ = [
    "RAGConfig",
    "get_default_rag_config",
    "RAGSettings",
    "get_rag_settings",
]
