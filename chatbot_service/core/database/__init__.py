"""
Database package for HeartGuard NLP Service.

Provides:
- Base SQLAlchemy models and configuration
- Database abstraction layer (FeedbackStorage interface)
- Backend-specific implementations (MySQL, PostgreSQL, DynamoDB, SQLite)
- XAMPP MySQL integration for existing deployments
- Query optimization with batch inserts, caching, and monitoring
"""

from .storage_interface import (
    FeedbackStorage,
    FeedbackStorageFactory,
    Feedback,
    FeedbackType,
    StorageException,
)
from .postgres_feedback_storage import PostgresFeedbackStorage

# Query optimization modules
from .query_optimizer import (
    QueryOptimizationConfig,
    TieredCache,
    BatchInsertManager,
    OptimizedChatHistoryQueries,
    MaterializedViewManager,
)
from .query_monitor import (
    QueryMetrics,
    SlowQueryLogger,
    QueryPerformanceMonitor,
    ConnectionPoolMonitor,
    DatabaseHealthChecker,
    QueryTimeoutManager,
    initialize_monitoring,
    shutdown_monitoring,
    get_performance_monitor,
    get_slow_query_logger,
    get_health_checker,
    get_timeout_manager,
)

__all__ = [
    # Storage interfaces
    "FeedbackStorage",
    "FeedbackStorageFactory",
    "Feedback",
    "FeedbackType",
    "StorageException",
    "PostgresFeedbackStorage",
    # Query optimization
    "QueryOptimizationConfig",
    "TieredCache",
    "BatchInsertManager",
    "OptimizedChatHistoryQueries",
    "MaterializedViewManager",
    # Query monitoring
    "QueryMetrics",
    "SlowQueryLogger",
    "QueryPerformanceMonitor",
    "ConnectionPoolMonitor",
    "DatabaseHealthChecker",
    "QueryTimeoutManager",
    "initialize_monitoring",
    "shutdown_monitoring",
    "get_performance_monitor",
    "get_slow_query_logger",
    "get_health_checker",
    "get_timeout_manager",
]
