import logging
import json
from typing import Dict, Any, List, Optional
from .storage_interface import FeedbackStorage, Feedback, FeedbackType, StorageException

logger = logging.getLogger(__name__)

class PostgresFeedbackStorage(FeedbackStorage):
    """
    PostgreSQL implementation of FeedbackStorage.
    
    Uses the PostgresDatabase integration.
    """
    
    def __init__(self, config=None, db_instance=None):
        """
        Initialize PostgreSQL storage backend.
        
        Args:
            config: Database configuration (from app_config)
            db_instance: PostgresDatabase instance
        """
        self.config = config
        self.db = db_instance
        logger.info("Initialized PostgresFeedbackStorage")
    
    async def _get_db(self):
        """Get database instance (lazy init)."""
        if self.db is None:
            from .postgres_db import get_database
            self.db = await get_database()
        return self.db
    
    async def store_feedback(self, feedback: Feedback) -> str:
        """Store feedback to PostgreSQL database."""
        try:
            db = await self._get_db()
            
            # Generate unique feedback ID if not provided
            feedback_id = feedback.result_id or f"fb_{feedback.user_id}_{int(feedback.created_at.timestamp())}"
            
            query = """
            INSERT INTO feedback (
                feedback_id, user_id, query, result_id, response_preview,
                feedback_type, rating, comment, metadata_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                feedback_id,
                feedback.user_id,
                feedback.query[:500],  # Limit query length
                feedback.result_id,
                feedback.metadata.get("response_preview", "")[:500],
                feedback.feedback_type.value,
                feedback.rating,
                feedback.comment,
                json.dumps(feedback.metadata),
                feedback.created_at,
            )
            
            await db.execute_query(query, params, operation="write")
            logger.info(f"Feedback stored: {feedback_id} for user={feedback.user_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise StorageException(f"Failed to store feedback: {e}") from e
    
    async def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Retrieve feedback from PostgreSQL database."""
        try:
            db = await self._get_db()
            
            query = "SELECT * FROM feedback WHERE feedback_id = %s"
            result = await db.execute_query(
                query, 
                (feedback_id,),
                operation="read",
                fetch_one=True
            )
            
            if not result:
                return None
            
            return self._row_to_feedback(result)
            
        except Exception as e:
            logger.error(f"Failed to retrieve feedback: {e}")
            raise StorageException(f"Failed to retrieve feedback: {e}") from e
    
    async def get_user_feedback(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Feedback]:
        """Retrieve user feedback from PostgreSQL database."""
        try:
            db = await self._get_db()
            
            query = """
            SELECT * FROM feedback 
            WHERE user_id = %s 
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """
            
            results = await db.execute_query(
                query,
                (user_id, limit, offset),
                operation="read",
                fetch_all=True
            )
            
            return [self._row_to_feedback(row) for row in (results or [])]
            
        except Exception as e:
            logger.error(f"Failed to retrieve user feedback: {e}")
            raise StorageException(f"Failed to retrieve user feedback: {e}") from e
    
    async def get_result_feedback(
        self,
        result_id: str,
        limit: int = 100,
    ) -> List[Feedback]:
        """Retrieve result feedback from PostgreSQL database."""
        try:
            db = await self._get_db()
            
            query = """
            SELECT * FROM feedback 
            WHERE result_id = %s 
            ORDER BY created_at DESC
            LIMIT %s
            """
            
            results = await db.execute_query(
                query,
                (result_id, limit),
                operation="read",
                fetch_all=True
            )
            
            return [self._row_to_feedback(row) for row in (results or [])]
            
        except Exception as e:
            logger.error(f"Failed to retrieve result feedback: {e}")
            raise StorageException(f"Failed to retrieve result feedback: {e}") from e
    
    async def get_feedback_by_type(
        self,
        feedback_type: FeedbackType,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Feedback]:
        """Retrieve feedback by type from PostgreSQL database."""
        try:
            db = await self._get_db()
            
            query = """
            SELECT * FROM feedback 
            WHERE feedback_type = %s 
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """
            
            results = await db.execute_query(
                query,
                (feedback_type.value, limit, offset),
                operation="read",
                fetch_all=True
            )
            
            return [self._row_to_feedback(row) for row in (results or [])]
            
        except Exception as e:
            logger.error(f"Failed to retrieve feedback by type: {e}")
            raise StorageException(f"Failed to retrieve feedback by type: {e}") from e
    
    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete feedback from PostgreSQL database."""
        try:
            db = await self._get_db()
            
            query = "DELETE FROM feedback WHERE feedback_id = %s"
            await db.execute_query(query, (feedback_id,), operation="write")
            
            logger.info(f"Feedback deleted: {feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete feedback: {e}")
            raise StorageException(f"Failed to delete feedback: {e}") from e
    
    async def aggregate_feedback(
        self,
        result_id: str,
    ) -> Dict[str, Any]:
        """Aggregate feedback statistics from PostgreSQL database."""
        try:
            db = await self._get_db()
            
            query = """
            SELECT 
                COUNT(*) as total_count,
                SUM(CASE WHEN feedback_type='helpful' THEN 1 ELSE 0 END) as helpful_count,
                SUM(CASE WHEN feedback_type='not_helpful' THEN 1 ELSE 0 END) as not_helpful_count,
                AVG(COALESCE(rating, 0)) as avg_rating
            FROM feedback 
            WHERE result_id = %s
            """
            
            result = await db.execute_query(
                query,
                (result_id,),
                operation="read",
                fetch_one=True
            )
            
            if not result:
                return {
                    "total_count": 0,
                    "helpful_count": 0,
                    "not_helpful_count": 0,
                    "avg_rating": 0.0,
                    "sentiment": "neutral",
                }
            
            return {
                "total_count": result.get("total_count", 0),
                "helpful_count": result.get("helpful_count", 0),
                "not_helpful_count": result.get("not_helpful_count", 0),
                "avg_rating": float(result.get("avg_rating", 0.0)),
                "sentiment": self._sentiment_from_counts(
                    result.get("helpful_count", 0),
                    result.get("not_helpful_count", 0)
                ),
            }
            
        except Exception as e:
            logger.error(f"Failed to aggregate feedback: {e}")
            raise StorageException(f"Failed to aggregate feedback: {e}") from e
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get overall feedback statistics."""
        try:
            db = await self._get_db()
            
            query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN feedback_type='helpful' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN feedback_type='not_helpful' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN feedback_type='partially_helpful' THEN 1 ELSE 0 END) as neutral,
                AVG(COALESCE(rating, 0)) as avg_rating
            FROM feedback
            """
            
            result = await db.execute_query(
                query,
                operation="read",
                fetch_one=True
            )
            
            if not result:
                return {
                    "total": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "satisfaction_rate": 0.0,
                    "avg_rating": 0.0,
                }
            
            total = result.get("total", 0)
            positive = result.get("positive", 0)
            
            return {
                "total": total,
                "positive": result.get("positive", 0),
                "negative": result.get("negative", 0),
                "neutral": result.get("neutral", 0),
                "satisfaction_rate": (positive / total) if total > 0 else 0.0,
                "avg_rating": float(result.get("avg_rating", 0.0)),
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            raise StorageException(f"Failed to get feedback stats: {e}") from e
    
    async def health_check(self) -> bool:
        """Check PostgreSQL database connectivity."""
        try:
            db = await self._get_db()
            result = await db.execute_query(
                "SELECT 1",
                operation="read",
                fetch_one=True
            )
            logger.debug("PostgreSQL health check passed")
            return result is not None
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    @staticmethod
    def _row_to_feedback(row: Dict[str, Any]) -> Feedback:
        """Convert database row to Feedback object."""
        return Feedback(
            user_id=row.get("user_id", ""),
            query=row.get("query", ""),
            result_id=row.get("result_id", ""),
            feedback_type=FeedbackType(row.get("feedback_type", "helpful")),
            rating=row.get("rating"),
            comment=row.get("comment"),
            created_at=row.get("created_at"),
            metadata=json.loads(row.get("metadata_json", "{}")) if isinstance(row.get("metadata_json"), str) else row.get("metadata_json", {}),
        )
    
    @staticmethod
    def _sentiment_from_counts(helpful: int, not_helpful: int) -> str:
        """Determine sentiment from feedback counts."""
        if helpful == 0 and not_helpful == 0:
            return "neutral"
        if helpful > not_helpful:
            return "positive"
        elif not_helpful > helpful:
            return "negative"
        else:
            return "neutral"
