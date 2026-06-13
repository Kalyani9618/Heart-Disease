"""
Trust & Explainability Layer for Medical RAG

Provides verification, validation, and transparency for RAG outputs:
- Source validation: Assesses credibility of retrieved sources
- Conflict detection: Identifies contradictory medical information
- Explainability: Generates human-readable explanations of retrieval decisions

Components:
- source_validator: Validates source credibility and recency
- conflict_detector: Detects contradictions between sources
- explainability: Adds AI-generated relevance explanations
"""

from .source_validator import MedicalSourceValidator, get_source_validator, ValidationResult, SourceCredibility
from .conflict_detector import MedicalConflictDetector, get_conflict_detector, Conflict, ConflictSeverity
from .explainability import ExplainableRetrieval

__all__ = [
    # Source Validation
    "MedicalSourceValidator",
    "get_source_validator",
    "ValidationResult",
    "SourceCredibility",
    # Conflict Detection
    "MedicalConflictDetector",
    "get_conflict_detector",
    "Conflict",
    "ConflictSeverity",
    # Explainability
    "ExplainableRetrieval",
]
