"""
In-Memory Feedback Storage Implementation

Provides an in-memory fallback storage for development and testing.
All data is lost when the application restarts.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.database.storage_interface import (
    FeedbackStorage, 
    Feedback, 
    FeedbackType,
    StorageException
)

logger = logging.getLogger(__name__)


class InMemoryFeedbackStorage(FeedbackStorage):
    """
    In-memory implementation of FeedbackStorage.
    
    Uses a dictionary to store feedback records.
    Suitable for development, testing, or when no database is available.
    
    Note: Data is NOT persisted - all data is lost on restart.
    """
    
    def __init__(self, config=None):
        """Initialize in-memory storage."""
        self._feedback: Dict[str, Feedback] = {}
        self._by_user: Dict[str, List[str]] = {}  # user_id -> [feedback_ids]
        self._by_result: Dict[str, List[str]] = {}  # result_id -> [feedback_ids]
        logger.info("InMemoryFeedbackStorage initialized (data will not persist)")
    
    async def store_feedback(self, feedback: Feedback) -> str:
        """Store feedback in memory."""
        feedback_id = str(uuid.uuid4())
        
        # Store the feedback
        self._feedback[feedback_id] = feedback
        
        # Index by user
        if feedback.user_id not in self._by_user:
            self._by_user[feedback.user_id] = []
        self._by_user[feedback.user_id].append(feedback_id)
        
        # Index by result
        if feedback.result_id not in self._by_result:
            self._by_result[feedback.result_id] = []
        self._by_result[feedback.result_id].append(feedback_id)
        
        logger.debug(f"Stored feedback {feedback_id} for user {feedback.user_id}")
        return feedback_id
    
    async def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Retrieve feedback by ID."""
        return self._feedback.get(feedback_id)
    
    async def get_user_feedback(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Feedback]:
        """Get all feedback from a user."""
        ids = self._by_user.get(user_id, [])
        selected_ids = ids[offset:offset + limit]
        return [self._feedback[fid] for fid in selected_ids if fid in self._feedback]
    
    async def get_result_feedback(
        self,
        result_id: str,
        limit: int = 100,
    ) -> List[Feedback]:
        """Get all feedback for a result."""
        ids = self._by_result.get(result_id, [])
        selected_ids = ids[:limit]
        return [self._feedback[fid] for fid in selected_ids if fid in self._feedback]
    
    async def get_feedback_by_type(
        self,
        feedback_type: FeedbackType,
        limit: int = 100,
    ) -> List[Feedback]:
        """Get feedback by type."""
        results = []
        for feedback in self._feedback.values():
            if feedback.feedback_type == feedback_type:
                results.append(feedback)
                if len(results) >= limit:
                    break
        return results
    
    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete feedback by ID."""
        if feedback_id not in self._feedback:
            return False
        
        feedback = self._feedback[feedback_id]
        
        # Remove from indices
        if feedback.user_id in self._by_user:
            if feedback_id in self._by_user[feedback.user_id]:
                self._by_user[feedback.user_id].remove(feedback_id)
        
        if feedback.result_id in self._by_result:
            if feedback_id in self._by_result[feedback.result_id]:
                self._by_result[feedback.result_id].remove(feedback_id)
        
        del self._feedback[feedback_id]
        return True
    
    async def aggregate_feedback(
        self,
        result_id: str,
    ) -> Dict[str, Any]:
        """Aggregate feedback for a result."""
        feedbacks = await self.get_result_feedback(result_id)
        
        total = len(feedbacks)
        helpful = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.HELPFUL)
        not_helpful = sum(1 for f in feedbacks if f.feedback_type == FeedbackType.NOT_HELPFUL)
        
        ratings = [f.rating for f in feedbacks if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        return {
            "total_count": total,
            "helpful_count": helpful,
            "not_helpful_count": not_helpful,
            "avg_rating": avg_rating,
            "sentiment": "positive" if helpful > not_helpful else "negative" if not_helpful > helpful else "neutral"
        }
    
    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        return True  # Always healthy for in-memory
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        helpful = sum(1 for f in self._feedback.values() if f.feedback_type == FeedbackType.HELPFUL)
        not_helpful = sum(1 for f in self._feedback.values() if f.feedback_type == FeedbackType.NOT_HELPFUL)
        neutral = len(self._feedback) - helpful - not_helpful
        
        ratings = [f.rating for f in self._feedback.values() if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        return {
            "total_feedback": len(self._feedback),
            "positive_count": helpful,
            "negative_count": not_helpful,
            "neutral_count": neutral,
            "average_rating": avg_rating,
        }
