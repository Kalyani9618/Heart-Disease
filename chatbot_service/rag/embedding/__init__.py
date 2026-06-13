"""
RAG Embedding Service — Remote MedCPT via Colab.

This is the sole embedding backend for inference mode.
Uses RemoteEmbeddingService to connect to a Colab-hosted
MedCPT encoder (768-dim) via ngrok.
"""

import os
from .base import BaseEmbeddingService
from .remote import RemoteEmbeddingService

# Simple factory for backward compatibility
def get_embedding_service(**kwargs):
    """Get or create the singleton RemoteEmbeddingService (if enabled)."""
    use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
    if not use_remote:
        return None
    return RemoteEmbeddingService.get_instance(**kwargs)

# Alias for backward compatibility
EmbeddingService = get_embedding_service

__all__ = [
    "BaseEmbeddingService",
    "RemoteEmbeddingService",
    "get_embedding_service",
    "EmbeddingService",
]
