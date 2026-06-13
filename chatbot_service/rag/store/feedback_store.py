"""
Database-backed Feedback Store for RAG Response Quality Tracking (REFACTORED).

PHASE 3: Database Decoupling - Now uses StorageInterface abstraction.

Old Code:
    - Tightly coupled to XAMPP/MySQL via direct get_database() import
    - Can't switch databases without code changes
    - Hard to test without database

New Code:
    - Accepts StorageInterface via dependency injection
    - Database-agnostic (MySQL, PostgreSQL, SQLite, etc.)
    - Easy to mock for testing
    - Fallback to in-memory storage on database failure

Migration Pattern:
    # Old (deprecated)
    store = FeedbackStore()  # Direct database import
    
    # New (recommended)
    from core.dependencies import DIContainer
    store = DIContainer.get_feedback_store()  # Injected
"""


import logging
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import json

from core.database.storage_interface import FeedbackStorage, Feedback, FeedbackType

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Legacy feedback entry model (kept for backward compatibility)."""
    feedback_id: str
    query: str
    response_preview: str
    rating: int  # 1 = thumbs up, -1 = thumbs down, 0 = neutral
    user_id: Optional[str]
    timestamp: str
    citations_count: int
    context_sources: List[str]
    user_comment: Optional[str] = None


class FeedbackStore:
    """
    Database-backed feedback storage with StorageInterface injection.
    
    Key Changes from Previous Version:
    1. Accepts FeedbackStorage interface (not direct database)
    2. Database-agnostic implementation
    3. Proper dependency injection pattern
    4. Fallback to in-memory storage on failure
    
    Example:
        # Get from DI container (recommended)
        store = DIContainer.get_feedback_store()
        
        # Or manually inject
        storage = DIContainer.get_storage()  # Get concrete implementation
        store = FeedbackStore(storage=storage)
    """
    
    def __init__(self, storage: FeedbackStorage):
        """
        Initialize with storage interface.
        
        Args:
            storage: FeedbackStorage implementation (MySQL, Postgres, SQLite, etc.)
            
        Raises:
            TypeError: If storage doesn't implement FeedbackStorage interface
        """
        if not isinstance(storage, FeedbackStorage):
            raise TypeError(
                f"storage must implement FeedbackStorage interface, "
                f"got {type(storage).__name__}"
            )
        
        self._storage = storage
        self._memory_fallback: List[FeedbackEntry] = []
        logger.info(f"FeedbackStore initialized with {type(storage).__name__}")
    
    async def record_feedback(
        self,
        feedback_id: str,
        rating: int,
        query: str,
        response: str,
        citations: List[Dict],
        user_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Record user feedback for a RAG response.
        
        Args:
            feedback_id: UUID from RAGResponse
            rating: 1 (positive), -1 (negative), 0 (neutral)
            query: Original user query
            response: Generated response (first 500 chars stored)
            citations: List of citations used
            user_id: Optional user identifier
            comment: Optional user comment
            
        Returns:
            Success status
        """
        entry = FeedbackEntry(
            feedback_id=feedback_id,
            query=query,
            response_preview=response[:500],
            rating=rating,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            citations_count=len(citations),
            context_sources=[c.get("source", "unknown") for c in citations[:5]],
            user_comment=comment
        )
        
        # Try to persist via storage interface
        try:
            # Convert to standard Feedback model for storage interface
            feedback = Feedback(
                user_id=user_id or "anonymous",
                query=query,
                result_id=feedback_id,
                feedback_type=FeedbackType.HELPFUL if rating == 1 else (
                    FeedbackType.NOT_HELPFUL if rating == -1 else FeedbackType.PARTIALLY_HELPFUL
                ),
                rating=rating,
                comment=comment,
                metadata={
                    "response_preview": response[:500],
                    "citations_count": len(citations),
                    "context_sources": [c.get("source", "unknown") for c in citations[:5]],
                }
            )
            
            stored_id = await self._storage.store_feedback(feedback)
            logger.info(f"Feedback persisted via storage interface: {stored_id} rating={rating}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist feedback via storage: {e}")
            # Fall through to memory fallback
        
        # Memory fallback
        self._memory_fallback.append(entry)
        logger.warning(f"Feedback stored in memory (storage unavailable): {feedback_id}")
        return True
    
    async def get_negative_feedback(self, limit: int = 100) -> List[Dict]:
        """
        Get recent negative feedback for debugging.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of negative feedback entries
        """
        try:
            # Try to get from storage interface
            results = await self._storage.get_feedback_by_type(
                feedback_type=FeedbackType.NOT_HELPFUL,
                limit=limit
            )
            
            return [
                {
                    "feedback_id": f.result_id,
                    "query": f.query,
                    "response_preview": f.metadata.get("response_preview", ""),
                    "rating": f.rating,
                    "user_id": f.user_id,
                    "timestamp": f.created_at.isoformat(),
                    "citations_count": f.metadata.get("citations_count", 0),
                    "context_sources": f.metadata.get("context_sources", []),
                    "user_comment": f.comment
                }
                for f in results
            ]
        except Exception as e:
            logger.error(f"Failed to query negative feedback: {e}")
        
        # Memory fallback
        return [
            asdict(e) for e in self._memory_fallback 
            if e.rating == -1
        ][-limit:]
    
    async def get_feedback_summary(self) -> Dict:
        """
        Get aggregate feedback statistics.
        
        Returns:
            Dictionary with feedback stats (total, positive, negative, neutral, satisfaction_rate)
        """
        try:
            # Try to get stats from storage interface
            stats = await self._storage.get_feedback_stats()
            return {
                **stats,
                "source": "storage_interface"
            }
        except Exception as e:
            logger.error(f"Failed to get feedback stats from storage: {e}")
        
        # Memory fallback
        positive = sum(1 for e in self._memory_fallback if e.rating == 1)
        negative = sum(1 for e in self._memory_fallback if e.rating == -1)
        neutral = sum(1 for e in self._memory_fallback if e.rating == 0)
        total = len(self._memory_fallback)
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "satisfaction_rate": (positive / total) if total > 0 else 0,
            "source": "memory_fallback"
        }
    
    async def bulk_insert_feedback(self, entries: List[FeedbackEntry]) -> int:
        """
        Bulk insert multiple feedback entries.
        
        Args:
            entries: List of FeedbackEntry objects
            
        Returns:
            Number of successfully inserted entries
        """
        success_count = 0
        
        for entry in entries:
            try:
                feedback = Feedback(
                    user_id=entry.user_id or "anonymous",
                    query=entry.query,
                    result_id=entry.feedback_id,
                    feedback_type=FeedbackType.HELPFUL if entry.rating == 1 else (
                        FeedbackType.NOT_HELPFUL if entry.rating == -1 else FeedbackType.PARTIALLY_HELPFUL
                    ),
                    rating=entry.rating,
                    comment=entry.user_comment,
                    metadata={
                        "response_preview": entry.response_preview,
                        "citations_count": entry.citations_count,
                        "context_sources": entry.context_sources,
                    }
                )
                
                await self._storage.store_feedback(feedback)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to insert feedback {entry.feedback_id}: {e}")
        
        logger.info(f"Bulk inserted {success_count}/{len(entries)} feedback entries")
        return success_count
    
    async def flush_in_memory_fallback(self) -> int:
        """
        Flush in-memory fallback to storage.
        
        Called when database becomes available after being down.
        
        Returns:
            Number of entries flushed
        """
        if not self._memory_fallback:
            return 0
        
        logger.info(f"Flushing {len(self._memory_fallback)} in-memory feedback entries to storage")
        
        flushed = 0
        remaining = []
        
        for entry in self._memory_fallback:
            try:
                feedback = Feedback(
                    user_id=entry.user_id or "anonymous",
                    query=entry.query,
                    result_id=entry.feedback_id,
                    feedback_type=FeedbackType.HELPFUL if entry.rating == 1 else (
                        FeedbackType.NOT_HELPFUL if entry.rating == -1 else FeedbackType.PARTIALLY_HELPFUL
                    ),
                    rating=entry.rating,
                    comment=entry.user_comment,
                    metadata={
                        "response_preview": entry.response_preview,
                        "citations_count": entry.citations_count,
                        "context_sources": entry.context_sources,
                    }
                )
                
                await self._storage.store_feedback(feedback)
                flushed += 1
            except Exception as e:
                logger.error(f"Failed to flush feedback {entry.feedback_id}: {e}")
                remaining.append(entry)
        
        self._memory_fallback = remaining
        logger.info(f"Flushed {flushed} entries, {len(remaining)} still in memory")
        
        return flushed


# Legacy function for backward compatibility (DEPRECATED)
# Use DIContainer.get_feedback_store() instead
_feedback_store: Optional[FeedbackStore] = None

async def get_feedback_store() -> FeedbackStore:
    """
    DEPRECATED: Get singleton feedback store instance.
    
    Use DIContainer.get_feedback_store() instead for proper dependency injection.
    """
    import warnings
    warnings.warn(
        "get_feedback_store() is deprecated! Use DIContainer.get_feedback_store()",
        DeprecationWarning,
        stacklevel=2
    )
    
    global _feedback_store
    if _feedback_store is None:
        # Legacy: try to create with database storage
        from core.dependencies import DIContainer
        _feedback_store = DIContainer.get_feedback_store()
    
    return _feedback_store

