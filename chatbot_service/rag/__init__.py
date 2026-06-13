"""
RAG Package — Inference Mode

Provides retrieval-augmented generation for the HeartGuard AI chatbot.
All ingestion code has been moved to _archive_ingestion/.

Subpackages:
    embedding/       — Remote embedding service (MedCPT 768-dim)
    store/           — ChromaDB vector store, feedback store
    retrieval/       — Reranker, fusion retriever, context assembler, RAPTOR, explainability
    pipeline/        — Self-RAG, CRAG, query optimizer
    memory/          — Memori conversation memory integration
    knowledge_graph/ — Medical ontology, drug interaction graph
    nlp/             — Medical NLP utilities
    multimodal/      — Image/multimodal query support
    trust/           — Source validation, conflict detection
"""

# Core exports for convenience
from .rag_engines import HeartDiseaseRAG, get_heart_disease_rag

# Embedding
from .embedding import RemoteEmbeddingService, get_embedding_service

# Store
from .store import ChromaDBVectorStore

__all__ = [
    # Core
    "HeartDiseaseRAG",
    "get_heart_disease_rag",
    # Embedding
    "RemoteEmbeddingService",
    "get_embedding_service",
    # Store
    "ChromaDBVectorStore",
]
