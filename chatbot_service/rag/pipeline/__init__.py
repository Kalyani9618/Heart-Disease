"""
RAG Advanced Pipelines — Self-RAG, CRAG, Query Optimizer.
"""

__all__ = []

# Lazy imports to avoid heavy dependencies at package load
def _lazy_import(name):
    """Import pipeline components on demand."""
    if name == "MedicalSelfRAG":
        from .self_rag_medical import MedicalSelfRAG
        return MedicalSelfRAG
    elif name == "CRAGFallback":
        from .crag_fallback import CRAGFallback
        return CRAGFallback
    elif name == "RAGQueryOptimizer":
        from .query_optimizer import RAGQueryOptimizer
        return RAGQueryOptimizer
    raise AttributeError(f"module 'rag.pipeline' has no attribute {name!r}")

def __getattr__(name):
    return _lazy_import(name)
