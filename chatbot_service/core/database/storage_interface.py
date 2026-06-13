"""
Database Abstraction Layer - Storage Interface

Provides a backend-agnostic interface for feedback storage, enabling support
for multiple database backends (MySQL, PostgreSQL, DynamoDB, SQLite, etc.)
without changing application code.

Addresses Database Coupling issue from Phase 3 audit:
- Current: XAMPP/MySQL hardcoded throughout codebase
- Solution: Abstract to interface, concrete implementations per backend

Architecture:
    FeedbackStorage (ABC)
        ├── MySQLFeedbackStorage
        ├── PostgresFeedbackStorage (future)
        └── DynamoDBFeedbackStorage (future)
    
    FeedbackStorageFactory: Creates appropriate backend based on config

Environment Variables:
    DB_BACKEND: 'postgres' | 'inmemory'
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_DATABASE
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of user feedback on RAG results."""
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    PARTIALLY_HELPFUL = "partially_helpful"
    IRRELEVANT = "irrelevant"
    INCORRECT = "incorrect"


@dataclass
class Feedback:
    """Feedback record model."""
    user_id: str
    query: str
    result_id: str
    feedback_type: FeedbackType
    rating: Optional[int] = None  # 1-5 stars
    comment: Optional[str] = None
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class FeedbackStorage(ABC):
    """
    Abstract base class for feedback storage backends.
    
    Defines the interface that all storage implementations must provide.
    This allows swapping backends without changing application code.
    """
    
    @abstractmethod
    async def store_feedback(self, feedback: Feedback) -> str:
        """
        Store a feedback record.
        
        Args:
            feedback: Feedback instance to store
            
        Returns:
            Feedback ID (unique identifier in storage backend)
            
        Raises:
            StorageException: If storage fails
        """
        pass
    
    @abstractmethod
    async def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """
        Retrieve a feedback record by ID.
        
        Args:
            feedback_id: ID returned from store_feedback()
            
        Returns:
            Feedback instance or None if not found
            
        Raises:
            StorageException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_user_feedback(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Feedback]:
        """
        Retrieve all feedback from a user.
        
        Args:
            user_id: User ID to query
            limit: Max results to return
            offset: Pagination offset
            
        Returns:
            List of Feedback instances
            
        Raises:
            StorageException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_result_feedback(
        self,
        result_id: str,
        limit: int = 100,
    ) -> List[Feedback]:
        """
        Retrieve all feedback for a specific result.
        
        Args:
            result_id: Result ID to query
            limit: Max results to return
            
        Returns:
            List of Feedback instances
            
        Raises:
            StorageException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def delete_feedback(self, feedback_id: str) -> bool:
        """
        Delete a feedback record.
        
        Args:
            feedback_id: ID to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            StorageException: If deletion fails
        """
        pass
    
    @abstractmethod
    async def aggregate_feedback(
        self,
        result_id: str,
    ) -> Dict[str, Any]:
        """
        Aggregate feedback statistics for a result.
        
        Args:
            result_id: Result ID to aggregate
            
        Returns:
            Dict with:
                - total_count: Total feedback records
                - helpful_count: Helpful votes
                - not_helpful_count: Unhelpful votes
                - avg_rating: Average rating (1-5)
                - sentiment: Overall sentiment estimate
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if storage backend is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
            
        Raises:
            StorageException: On critical failures
        """
        pass


class StorageException(Exception):
    """Base exception for storage errors."""
    pass


# MySQLFeedbackStorage has been moved to its own file: mysql_feedback_storage.py
# to avoid circular imports and follow the project's dependency injection pattern.



class FeedbackStorageFactory:
    """
    Factory for creating appropriate FeedbackStorage backend instances.
    
    Decouples storage backend selection from application code.
    Allows configuration-driven backend selection.
    """
    
    _instance: Optional[FeedbackStorage] = None
    
    @classmethod
    def create(
        cls,
        backend: str,
        connection_config: Dict[str, Any],
    ) -> FeedbackStorage:
        """
        Create a FeedbackStorage instance for the specified backend.
        
        Args:
            backend: Backend type ('mysql', 'postgres', 'dynamodb', 'sqlite')
            connection_config: Backend-specific configuration
            
        Returns:
            FeedbackStorage implementation instance
            
        Raises:
            ValueError: If backend is unknown
        """
        backend_lower = backend.lower().strip()
        
        if backend_lower == "postgres":
            logger.info("Creating PostgresFeedbackStorage instance")
            from .postgres_feedback_storage import PostgresFeedbackStorage
            return PostgresFeedbackStorage(connection_config)
        
        elif backend_lower == "inmemory":
            logger.info("Creating InMemoryFeedbackStorage instance")
            from .inmemory_feedback_storage import InMemoryFeedbackStorage
            return InMemoryFeedbackStorage()
        
        else:
            raise ValueError(f"Unsupported storage backend: {backend}. Only 'postgres' and 'inmemory' are supported.")
    
    @classmethod
    def get_instance(
        cls,
        backend: Optional[str] = None,
        connection_config: Optional[Dict[str, Any]] = None,
    ) -> FeedbackStorage:
        """
        Get or create a singleton FeedbackStorage instance.
        
        Args:
            backend: Backend type (required on first call)
            connection_config: Backend config (required on first call)
            
        Returns:
            FeedbackStorage singleton instance
            
        Raises:
            ValueError: If not initialized and backend not provided
        """
        if cls._instance is None:
            if backend is None or connection_config is None:
                # Try to load from AppConfig
                try:
                    from core.config.app_config import get_app_config
                    config = get_app_config()
                    backend = backend or config.database.backend
                    connection_config = connection_config or {
                        "host": config.database.host,
                        "port": config.database.port,
                        "user": config.database.user,
                        "password": config.database.password,
                        "database": config.database.database,
                    }
                except Exception as e:
                    logger.warning(f"Failed to load config: {e}. Using defaults.")
                    backend = backend or "postgres"
                    connection_config = connection_config or {
                        "host": "localhost",
                        "port": 5432,
                        "user": "postgres",
                        "password": "postgres",
                        "database": "heartguard",
                    }
            
            cls._instance = cls.create(backend, connection_config)
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
