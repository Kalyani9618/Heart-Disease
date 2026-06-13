"""
Database adapters for different database backends
Provides database-specific implementations with proper security measures
"""

from .postgresql_adapter import PostgreSQLSearchAdapter

__all__ = ["PostgreSQLSearchAdapter"]
