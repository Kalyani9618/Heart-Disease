"""
Lightweight Data Models for Retrieval Layer

Contains only the types needed for inference-time retrieval.
The full models (with ingestion metadata) are archived in
rag/_archive_ingestion/data_sources/models.py.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime


class SourceTier(str, Enum):
    """Priority tier for retrieval routing."""
    TIER_1_STATPEARLS = "tier_1_statpearls"
    TIER_1_TEXTBOOKS = "tier_1_textbooks"
    TIER_2_PUBMED = "tier_2_pubmed"


class DocumentSource(str, Enum):
    """Source identifier."""
    STATPEARLS = "statpearls"
    TEXTBOOKS = "textbooks"
    PUBMED = "pubmed"


class ReviewStatus(str, Enum):
    """Quality/verification status."""
    VERIFIED = "verified"
    PREPRINT = "preprint"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class MedicalDocument:
    """
    Lightweight medical document schema for retrieval results.

    Only includes fields relevant to inference — ingestion-specific
    tracking fields have been removed.
    """
    document_id: str
    title: str
    content: str
    source: DocumentSource
    tier: SourceTier

    source_url: Optional[str] = None
    publication_date: Optional[datetime] = None
    authors: Optional[List[str]] = field(default_factory=list)
    mesh_terms: Optional[List[str]] = field(default_factory=list)

    confidence_score: float = 1.0
    review_status: ReviewStatus = ReviewStatus.VERIFIED

    keywords: Optional[List[str]] = field(default_factory=list)
    related_conditions: Optional[List[str]] = field(default_factory=list)
    clinical_context: Optional[str] = None
