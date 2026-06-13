"""
Multimodal RAG Module for Cardio AI

Provides multimodal document processing capabilities:
- Table extraction from medical PDFs (lab results, vital signs)
- Image analysis for ECG/medical charts
- Context-aware content processing
- Integration with existing RAG pipeline

NOTE: Ingestion services have been archived to _archive_ingestion/.
This module now provides QUERY-TIME multimodal processing only.
"""

from .config import MultimodalConfig, ContextConfig
from .processors import (
    TableProcessor,
    ImageProcessor,
    EquationProcessor,
    ContextExtractor,
    DocumentParser,
    ParsedContent,
    ProcessedDocument,
    DocStatus,
    ContentType,
    get_processor_for_type,
)
from .prompts import MEDICAL_PROMPTS

# Document Parsers (from RAG-Anything)
from .parser import (
    Parser,
    MineruParser,
    DoclingParser,
    MineruExecutionError,
)

# Batch Processing (from RAG-Anything)
from .batch_parser import (
    BatchParser,
    BatchProcessingResult,
)
from .batch import BatchMixin

# Query Functionality (from RAG-Anything)
from .query import (
    MultimodalQueryMixin,
    MultimodalQueryService as QueryService,
)

# Zero-Shot Medical Classification
from .zero_shot_classifier import ZeroShotMedicalClassifier

# Utilities (from RAG-Anything)
from .utils import (
    RobustJSONParser,
    validate_image_file,
    encode_image_to_base64,
    get_image_mime_type,
    generate_multimodal_cache_key,
    extract_image_paths_from_context,
    build_vlm_messages,
    get_processor_for_type as get_modal_processor,
    compute_content_hash,
)

__all__ = [
    # Configuration
    "MultimodalConfig",
    "ContextConfig",
    # Processors
    "TableProcessor",
    "ImageProcessor",
    "EquationProcessor",
    "ContextExtractor",
    "DocumentParser",
    "ParsedContent",
    "ProcessedDocument",
    "DocStatus",
    "ContentType",
    "get_processor_for_type",
    # Document Parsers
    "Parser",
    "MineruParser",
    "DoclingParser",
    "MineruExecutionError",
    # Batch Processing
    "BatchParser",
    "BatchProcessingResult",
    "BatchMixin",
    # Query Functionality
    "MultimodalQueryMixin",
    "QueryService",
    # Zero-Shot Classification
    "ZeroShotMedicalClassifier",
    # Utilities
    "RobustJSONParser",
    "validate_image_file",
    "encode_image_to_base64",
    "get_image_mime_type",
    "generate_multimodal_cache_key",
    "extract_image_paths_from_context",
    "build_vlm_messages",
    "get_modal_processor",
    "compute_content_hash",
    # Prompts
    "MEDICAL_PROMPTS",
]
