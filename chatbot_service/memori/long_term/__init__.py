"""
Long-term Memory Module for Memori

Provides persistent fact extraction and memory consolidation for AI agents.
Supports background processing, async memory operations, and parallel
multi-source data extraction.
"""

from .fact_extractor import (
    FactExtractor,
    MemoryExtractionWorker,
    get_fact_extractor,
    get_extraction_worker,
    shutdown_extraction_worker,
    ExtractedFact,
    ExtractionTask,
    FactCategory,
    MedicalPatternMatcher,
)

__all__ = [
    "FactExtractor",
    "MemoryExtractionWorker",
    "get_fact_extractor",
    "get_extraction_worker",
    "shutdown_extraction_worker",
    "ExtractedFact",
    "ExtractionTask",
    "FactCategory",
    "MedicalPatternMatcher",
]
