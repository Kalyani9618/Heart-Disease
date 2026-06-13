"""
SQLAlchemy-based database manager for Memori v2.0
Replaces the existing database.py with cross-database compatibility
"""

import importlib.util
import json
import uuid
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy import create_engine, func, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from ..config.pool_config import pool_config
from ..utils.exceptions import DatabaseError
from ..utils.pydantic_models import (
    ProcessedLongTermMemory,
)
# auto_creator removed - SQLAlchemy handles database creation via create_all()
# from .auto_creator import DatabaseAutoCreator
from .models import (
    Base,
    ChatHistory,
    LongTermMemory,
    ShortTermMemory,
)
# query_translator removed - using SQLAlchemy ORM directly
# from .query_translator import QueryParameterTranslator
from .search_service import SearchService


class SQLAlchemyDatabaseManager:
    """SQLAlchemy-based database manager with PostgreSQL support"""

    def __init__(
        self,
        database_connect: str,
        template: str = "basic",
        schema_init: bool = True,
        pool_size: int = pool_config.get("pool_size", 5),
        max_overflow: int = pool_config.get("max_overflow", 10),
        pool_timeout: int = pool_config.get("pool_timeout", 30),
        pool_recycle: int = pool_config.get("pool_recycle", 3600),
        pool_pre_ping: bool = pool_config.get("pool_pre_ping", True),
   ):
        self.database_connect = database_connect
        self.template = template
        self.schema_init = schema_init

        # Connection pool settings
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping

        # auto_creator removed - using SQLAlchemy create_all()
        # self.auto_creator = DatabaseAutoCreator(schema_init)

        # Ensure database exists (create if necessary) - handled by SQLAlchemy
        # self.database_connect = self.auto_creator.ensure_database_exists(database_connect)

        # Parse connection string and create engine
        self.engine = self._create_engine(self.database_connect)
        self.database_type = self.engine.dialect.name

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize search service
        self._search_service = None

        # query_translator removed - using SQLAlchemy ORM directly  
        # self.query_translator = QueryParameterTranslator(self.database_type)

        logger.info(
            f"Initialized SQLAlchemy database manager for {self.database_type} "
            f"(pool_size={pool_size}, max_overflow={max_overflow})"
        )

    def _validate_database_dependencies(self, database_connect: str):
        """Validate that required PostgreSQL database drivers are installed"""
        if database_connect.startswith("postgresql:") or database_connect.startswith(
            "postgresql+"
        ):
            # Check for PostgreSQL drivers
            if (
                importlib.util.find_spec("psycopg2") is None
                and importlib.util.find_spec("asyncpg") is None
            ):
                error_msg = (
                    "ERROR: No PostgreSQL driver found. Install one of the following:\n\n"
                    "Option 1 (Recommended): pip install psycopg2-binary\n"
                    "Option 2: pip install memorisdk[postgres]\n\n"
                    "Then use connection string: postgresql://user:pass@host:port/db"
                )
                raise DatabaseError(error_msg)
        else:
            raise DatabaseError(
                f"Unsupported database type. Only PostgreSQL is supported. "
                f"Use: postgresql://user:pass@host:port/db"
            )

    def _create_engine(self, database_connect: str):
        """Create SQLAlchemy engine with PostgreSQL configuration"""
        try:
            # Validate database driver dependencies first
            self._validate_database_dependencies(database_connect)

            # PostgreSQL-specific configuration with connection pool management
            engine = create_engine(
                database_connect,
                json_serializer=json.dumps,
                json_deserializer=json.loads,
                echo=False,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
            )

            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            return engine

        except DatabaseError:
            raise
        except ModuleNotFoundError as e:
            if "psycopg" in str(e).lower() or "postgresql" in str(e).lower():
                error_msg = (
                    "ERROR: PostgreSQL driver not found. Install one of the following:\n\n"
                    "Option 1 (Recommended): pip install psycopg2-binary\n"
                    "Option 2: pip install memorisdk[postgres]\n\n"
                    f"Original error: {e}"
                )
                raise DatabaseError(error_msg)
            else:
                raise DatabaseError(f"Missing required dependency: {e}")
        except SQLAlchemyError as e:
            error_msg = f"Database connection failed: {e}\n\nCheck your connection string and ensure the database server is running."
            raise DatabaseError(error_msg)
        except Exception as e:
            raise DatabaseError(f"Failed to create database engine: {e}")

    def initialize_schema(self):
        """Initialize database schema"""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)

            # Setup database-specific features
            self._setup_database_features()

            logger.info(
                f"Database schema initialized successfully for {self.database_type}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise DatabaseError(f"Failed to initialize schema: {e}")

    def _setup_database_features(self):
        """Setup PostgreSQL full-text search features"""
        try:
            with self.engine.connect() as conn:
                self._setup_postgresql_fts(conn)
                conn.commit()

        except Exception as e:
            logger.warning(f"Failed to setup database-specific features: {e}")

    def _setup_postgresql_fts(self, conn):
        """Setup PostgreSQL full-text search"""
        try:
            # Add tsvector columns
            conn.execute(
                text(
                    "ALTER TABLE short_term_memory ADD COLUMN IF NOT EXISTS search_vector tsvector"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE long_term_memory ADD COLUMN IF NOT EXISTS search_vector tsvector"
                )
            )

            # Create GIN indexes
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_short_term_search_vector ON short_term_memory USING GIN(search_vector)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_long_term_search_vector ON long_term_memory USING GIN(search_vector)"
                )
            )

            # Create update functions and triggers
            conn.execute(
                text(
                    """
                CREATE OR REPLACE FUNCTION update_short_term_search_vector() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector := to_tsvector('english', COALESCE(NEW.searchable_content, '') || ' ' || COALESCE(NEW.summary, ''));
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql;
            """
                )
            )

            conn.execute(
                text(
                    """
                DROP TRIGGER IF EXISTS update_short_term_search_vector_trigger ON short_term_memory;
                CREATE TRIGGER update_short_term_search_vector_trigger
                BEFORE INSERT OR UPDATE ON short_term_memory
                FOR EACH ROW EXECUTE FUNCTION update_short_term_search_vector();
            """
                )
            )

            logger.info("PostgreSQL FTS setup completed")

        except Exception as e:
            logger.warning(f"PostgreSQL FTS setup failed: {e}")

    def _get_search_service(self) -> SearchService:
        """Get search service instance with fresh session and proper error handling"""
        session = None
        try:
            if not hasattr(self, "SessionLocal") or not self.SessionLocal:
                logger.error("SessionLocal not available for search service")
                return None

            # Always create a new session to avoid stale connections
            session = self.SessionLocal()
            if not session:
                logger.error("Failed to create database session")
                return None

            # Verify session is valid
            try:
                # Test the session connection
                session.execute(text("SELECT 1"))
            except Exception as e:
                logger.error(f"Database session is not functional: {e}")
                if session:
                    session.close()
                return None

            search_service = SearchService(session, self.database_type)

            # Verify SearchService was initialized correctly
            if not hasattr(search_service, "session") or search_service.session is None:
                logger.error(
                    "SearchService was not properly initialized with a session"
                )
                if session:
                    session.close()
                return None

            logger.debug(
                f"Created new search service instance for database type: {self.database_type}"
            )
            return search_service

        except Exception as e:
            logger.error(f"Failed to create search service: {e}")
            logger.debug(
                f"Search service creation error: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            if session:
                try:
                    session.close()
                except Exception:
                    # Ignore session close errors during cleanup
                    pass
            return None

    def store_chat_history(
        self,
        chat_id: str,
        user_input: str,
        ai_output: str,
        model: str,
        session_id: str,
        user_id: str = "default",
        assistant_id: str = None,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime = None,
    ):
        """Store chat history with multi-tenant isolation"""
        with self.SessionLocal() as session:
            try:
                # Build ChatHistory kwargs - map timestamp to created_at if provided
                chat_kwargs = {
                    "chat_id": chat_id,
                    "user_input": user_input,
                    "ai_output": ai_output,
                    "model": model,
                    "session_id": session_id,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "tokens_used": tokens_used,
                    "metadata_json": metadata or {},
                }

                # Map timestamp parameter to created_at field for backward compatibility
                if timestamp is not None:
                    chat_kwargs["created_at"] = timestamp

                chat_history = ChatHistory(**chat_kwargs)

                session.merge(chat_history)  # Use merge for INSERT OR REPLACE behavior
                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                raise DatabaseError(f"Failed to store chat history: {e}")

    def get_chat_history(
        self,
        user_id: str = "default",
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get chat history with optional session filtering"""
        with self.SessionLocal() as session:
            try:
                query = session.query(ChatHistory).filter(
                    ChatHistory.user_id == user_id
                )

                if session_id:
                    query = query.filter(ChatHistory.session_id == session_id)

                results = (
                    query.order_by(ChatHistory.timestamp.desc()).limit(limit).all()
                )

                # Convert to dictionaries
                return [
                    {
                        "chat_id": result.chat_id,
                        "user_input": result.user_input,
                        "ai_output": result.ai_output,
                        "model": result.model,
                        "timestamp": result.created_at,
                        "session_id": result.session_id,
                        "user_id": result.user_id,
                        "tokens_used": result.tokens_used,
                        "metadata": result.metadata_json or {},
                    }
                    for result in results
                ]

            except SQLAlchemyError as e:
                raise DatabaseError(f"Failed to get chat history: {e}")

    def store_long_term_memory_enhanced(
        self,
        memory: ProcessedLongTermMemory,
        chat_id: str,
        user_id: str = "default",
        assistant_id: str = None,
        session_id: str = "default",
    ) -> str:
        """Store a ProcessedLongTermMemory with enhanced schema and multi-tenant isolation"""
        memory_id = str(uuid.uuid4())

        with self.SessionLocal() as session:
            try:
                long_term_memory = LongTermMemory(
                    memory_id=memory_id,
                    processed_data=memory.model_dump(mode="json"),
                    importance_score=memory.importance_score,
                    category_primary=memory.classification.value,
                    retention_type="long_term",
                    user_id=user_id,
                    assistant_id=assistant_id,
                    session_id=session_id,
                    created_at=datetime.now(),
                    searchable_content=memory.content,
                    summary=memory.summary,
                    novelty_score=0.5,
                    relevance_score=0.5,
                    actionability_score=0.5,
                    classification=memory.classification.value,
                    memory_importance=memory.importance.value,
                    topic=memory.topic,
                    entities_json=memory.entities,
                    keywords_json=memory.keywords,
                    is_user_context=memory.is_user_context,
                    is_preference=memory.is_preference,
                    is_skill_knowledge=memory.is_skill_knowledge,
                    is_current_project=memory.is_current_project,
                    promotion_eligible=memory.promotion_eligible,
                    duplicate_of=memory.duplicate_of,
                    supersedes_json=memory.supersedes,
                    related_memories_json=memory.related_memories,
                    confidence_score=memory.confidence_score,
                    classification_reason=memory.classification_reason,
                    processed_for_duplicates=False,
                    conscious_processed=False,
                )

                session.add(long_term_memory)
                session.commit()

                logger.debug(f"Stored enhanced long-term memory {memory_id}")
                return memory_id

            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to store enhanced long-term memory: {e}")
                raise DatabaseError(f"Failed to store enhanced long-term memory: {e}")

    def search_memories(
        self,
        query: str,
        user_id: str = "default",
        assistant_id: str | None = None,
        session_id: str | None = None,
        category_filter: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories using the cross-database search service"""
        search_service = None
        try:
            logger.debug(
                f"Starting memory search for query '{query}' in user_id '{user_id}', assistant_id '{assistant_id}', session_id '{session_id}' with category_filter={category_filter}"
            )
            search_service = self._get_search_service()

            if not search_service:
                logger.error("Failed to create search service instance")
                return []

            results = search_service.search_memories(
                query, user_id, assistant_id, session_id, category_filter, limit
            )
            logger.debug(f"Search for '{query}' returned {len(results)} results")

            # Validate results structure
            if not isinstance(results, list):
                logger.warning(
                    f"Search service returned unexpected type: {type(results)}, converting to list"
                )
                results = list(results) if results else []

            return results

        except Exception as e:
            logger.error(
                f"Memory search failed for query '{query}' in user_id '{user_id}': {e}"
            )
            logger.debug(
                f"Search error details: {type(e).__name__}: {str(e)}", exc_info=True
            )
            # Return empty list instead of raising exception to avoid breaking auto_ingest
            return []

        finally:
            # Ensure session is properly closed, even if an exception occurred
            if search_service and hasattr(search_service, "session"):
                try:
                    if search_service.session:
                        logger.debug("Closing search service session")
                        search_service.session.close()
                except Exception as session_e:
                    logger.warning(f"Error closing search service session: {session_e}")

    def get_memory_stats(self, user_id: str = "default") -> dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self.SessionLocal() as session:
            try:
                stats = {}

                # Basic counts
                stats["chat_history_count"] = (
                    session.query(ChatHistory)
                    .filter(ChatHistory.user_id == user_id)
                    .count()
                )

                stats["short_term_count"] = (
                    session.query(ShortTermMemory)
                    .filter(ShortTermMemory.user_id == user_id)
                    .count()
                )

                stats["long_term_count"] = (
                    session.query(LongTermMemory)
                    .filter(LongTermMemory.user_id == user_id)
                    .count()
                )

                # Category breakdown
                categories = {}

                # Short-term categories
                short_categories = (
                    session.query(
                        ShortTermMemory.category_primary,
                        func.count(ShortTermMemory.memory_id).label("count"),
                    )
                    .filter(ShortTermMemory.user_id == user_id)
                    .group_by(ShortTermMemory.category_primary)
                    .all()
                )

                for cat, count in short_categories:
                    categories[cat] = categories.get(cat, 0) + count

                # Long-term categories
                long_categories = (
                    session.query(
                        LongTermMemory.category_primary,
                        func.count(LongTermMemory.memory_id).label("count"),
                    )
                    .filter(LongTermMemory.user_id == user_id)
                    .group_by(LongTermMemory.category_primary)
                    .all()
                )

                for cat, count in long_categories:
                    categories[cat] = categories.get(cat, 0) + count

                stats["memories_by_category"] = categories

                # Average importance
                short_avg = (
                    session.query(func.avg(ShortTermMemory.importance_score))
                    .filter(ShortTermMemory.user_id == user_id)
                    .scalar()
                    or 0
                )

                long_avg = (
                    session.query(func.avg(LongTermMemory.importance_score))
                    .filter(LongTermMemory.user_id == user_id)
                    .scalar()
                    or 0
                )

                total_memories = stats["short_term_count"] + stats["long_term_count"]
                if total_memories > 0:
                    # Weight averages by count
                    total_avg = (
                        (short_avg * stats["short_term_count"])
                        + (long_avg * stats["long_term_count"])
                    ) / total_memories
                    stats["average_importance"] = float(total_avg) if total_avg else 0.0
                else:
                    stats["average_importance"] = 0.0

                # Database info
                stats["database_type"] = self.database_type
                stats["database_url"] = (
                    self.database_connect.split("@")[-1]
                    if "@" in self.database_connect
                    else self.database_connect
                )

                return stats

            except SQLAlchemyError as e:
                raise DatabaseError(f"Failed to get memory stats: {e}")

    def clear_memory(self, user_id: str = "default", memory_type: str | None = None):
        """Clear memory data"""
        with self.SessionLocal() as session:
            try:
                if memory_type == "short_term":
                    session.query(ShortTermMemory).filter(
                        ShortTermMemory.user_id == user_id
                    ).delete()
                elif memory_type == "long_term":
                    session.query(LongTermMemory).filter(
                        LongTermMemory.user_id == user_id
                    ).delete()
                elif memory_type == "chat_history":
                    session.query(ChatHistory).filter(
                        ChatHistory.user_id == user_id
                    ).delete()
                else:  # Clear all
                    session.query(ShortTermMemory).filter(
                        ShortTermMemory.user_id == user_id
                    ).delete()
                    session.query(LongTermMemory).filter(
                        LongTermMemory.user_id == user_id
                    ).delete()
                    session.query(ChatHistory).filter(
                        ChatHistory.user_id == user_id
                    ).delete()

                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                raise DatabaseError(f"Failed to clear memory: {e}")

    def execute_with_translation(self, query: str, parameters: dict[str, Any] = None):
        """
        Execute a query with automatic parameter translation for cross-database compatibility.

        Args:
            query: SQL query string
            parameters: Query parameters

        Returns:
            Query result
        """
        translated_params = parameters or {}

        with self.engine.connect() as conn:
            result = conn.execute(text(query), translated_params)
            conn.commit()
            return result

    def _get_connection(self):
        """
        Compatibility method for legacy code that expects raw database connections.

        Returns a context manager that provides a SQLAlchemy connection with
        automatic parameter translation support.

        This is used by memory.py for direct SQL queries.
        """
        from contextlib import contextmanager

        @contextmanager
        def connection_context():
            class TranslatingConnection:
                """Wrapper that adds parameter translation to SQLAlchemy connections"""

                def __init__(self, conn, translator):
                    self._conn = conn
                    self._translator = translator

                def execute(self, query, parameters=None):
                    """Execute query with automatic parameter translation"""
                    if parameters:
                        # Handle both text() queries and raw strings
                        if hasattr(query, "text"):
                            # SQLAlchemy text() object
                            translated_params = self._translator.translate_parameters(
                                parameters
                            )
                            return self._conn.execute(query, translated_params)
                        else:
                            # Raw string query
                            translated_params = self._translator.translate_parameters(
                                parameters
                            )
                            return self._conn.execute(
                                text(str(query)), translated_params
                            )
                    else:
                        return self._conn.execute(query)

                def commit(self):
                    """Commit transaction"""
                    return self._conn.commit()

                def rollback(self):
                    """Rollback transaction"""
                    return self._conn.rollback()

                def close(self):
                    """Close connection"""
                    return self._conn.close()

                def fetchall(self):
                    """Compatibility method for cursor-like usage"""
                    # This is for backwards compatibility with code that expects cursor.fetchall()
                    return []

                def scalar(self):
                    """Compatibility method for cursor-like usage"""
                    return None

                def __getattr__(self, name):
                    """Delegate unknown attributes to the underlying connection"""
                    return getattr(self._conn, name)

            conn = self.engine.connect()
            try:
                yield conn
            finally:
                conn.close()

        return connection_context()

    def close(self):
        """Close database connections"""
        if self._search_service and hasattr(self._search_service, "session"):
            self._search_service.session.close()

        if hasattr(self, "engine"):
            self.engine.dispose()

    def check_fts_integrity(self) -> dict[str, Any]:
        """
        Check SQLite FTS5 index integrity and rebuild if necessary.
        
        This should be called periodically (e.g., on app startup or via scheduled task)
        to ensure the FTS index is in sync with the source tables.
        
        Returns:
            Dict with status and any errors found
        """
        if self.database_type != "sqlite":
            return {"status": "skipped", "reason": "Not SQLite database"}
        
        try:
            with self.engine.connect() as conn:
                # Check if FTS table exists
                result = conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search_fts'"
                )).fetchone()
                
                if not result:
                    return {"status": "skipped", "reason": "FTS table not configured"}
                
                # Run integrity check - this validates the FTS index structure
                try:
                    conn.execute(text(
                        "INSERT INTO memory_search_fts(memory_search_fts) VALUES('integrity-check')"
                    ))
                    conn.commit()
                    logger.info("FTS integrity check passed")
                    return {"status": "ok", "message": "FTS index integrity verified"}
                except Exception as integrity_error:
                    logger.warning(f"FTS integrity check failed: {integrity_error}")
                    # Attempt rebuild
                    return self.rebuild_fts_index()
                    
        except Exception as e:
            logger.error(f"FTS integrity check error: {e}")
            return {"status": "error", "error": str(e)}

    def rebuild_fts_index(self) -> dict[str, Any]:
        """
        Rebuild the SQLite FTS5 index from source tables.
        
        This is useful if the index gets out of sync with the source tables
        (e.g., due to a transaction rollback that didn't propagate to triggers).
        
        Returns:
            Dict with rebuild status
        """
        if self.database_type != "sqlite":
            return {"status": "skipped", "reason": "Not SQLite database"}
        
        try:
            with self.engine.connect() as conn:
                logger.info("Rebuilding FTS index...")
                
                # Delete all entries from FTS table
                conn.execute(text("DELETE FROM memory_search_fts"))
                
                # Re-populate from short_term_memory
                conn.execute(text("""
                    INSERT INTO memory_search_fts(memory_id, memory_type, user_id, assistant_id, session_id, searchable_content, summary, category_primary)
                    SELECT memory_id, 'short_term', user_id, assistant_id, session_id, searchable_content, summary, category_primary
                    FROM short_term_memory
                """))
                
                # Re-populate from long_term_memory
                conn.execute(text("""
                    INSERT INTO memory_search_fts(memory_id, memory_type, user_id, assistant_id, session_id, searchable_content, summary, category_primary)
                    SELECT memory_id, 'long_term', user_id, assistant_id, session_id, searchable_content, summary, category_primary
                    FROM long_term_memory
                """))
                
                conn.commit()
                
                # Get row counts for reporting
                fts_count = conn.execute(text(
                    "SELECT COUNT(*) FROM memory_search_fts"
                )).scalar()
                
                logger.info(f"FTS index rebuilt successfully with {fts_count} entries")
                return {
                    "status": "rebuilt",
                    "message": f"FTS index rebuilt with {fts_count} entries"
                }
                
        except Exception as e:
            logger.error(f"FTS index rebuild failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_database_info(self) -> dict[str, Any]:
        """Get database information and capabilities"""
        base_info = {
            "database_type": self.database_type,
            "database_url": (
                self.database_connect.split("@")[-1]
                if "@" in self.database_connect
                else self.database_connect
            ),
            "driver": self.engine.dialect.driver,
            "server_version": getattr(self.engine.dialect, "server_version_info", None),
            "supports_fulltext": True,  # Assume true for SQLAlchemy managed connections
            "auto_creation_enabled": self.enable_auto_creation,
        }

        # Add auto-creation specific information
        if hasattr(self, "auto_creator"):
            creation_info = self.auto_creator.get_database_info(self.database_connect)
            base_info.update(creation_info)

        return base_info
