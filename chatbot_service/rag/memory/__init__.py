"""
RAG Conversation Memory — Memori integration and interfaces.
"""

__all__ = []

def __getattr__(name):
    if name == "MemoriRAGBridge":
        from .memori_integration import MemoriRAGBridge
        return MemoriRAGBridge
    elif name == "create_memori_rag_bridge":
        from .memori_integration import create_memori_rag_bridge
        return create_memori_rag_bridge
    raise AttributeError(f"module 'rag.memory' has no attribute {name!r}")
