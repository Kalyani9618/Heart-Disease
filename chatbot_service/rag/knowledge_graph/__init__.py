"""
Knowledge Graph Package.

Provides graph-based knowledge representation and retrieval
for enhanced context understanding and semantic search.

Components:
- graph_rag: Graph-enhanced RAG pipeline
- medical_ontology: Medical term matching and normalization
- phonetic_matcher: Phonetic drug name matching
"""

from .graph_rag import (
    GraphRAGService,
    GraphContext,
    GraphSearchResult,
)

__all__ = [
    # Graph RAG
    "GraphRAGService",
    "GraphContext",
    "GraphSearchResult",
    # Interaction Checker
    "GraphInteractionChecker",
]

# Lazy import for interaction checker
try:
    from .interaction_checker import GraphInteractionChecker
except ImportError:
    pass
