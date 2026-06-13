"""
PostgreSQL database connection module using asyncpg.
Supports traditional relational data storage.
Vector search is handled by ChromaDB (see rag/chromadb_store.py).
"""

import os
import logging
import re
import json
from typing import Optional, List, Dict, Any, Literal
from contextlib import asynccontextmanager
import numpy as np

from core.config.app_config import get_app_config

# Try to import database drivers
try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    asyncpg = None

logger = logging.getLogger(__name__)
config = get_app_config()

class PostgresDatabase:
    """Database connector for PostgreSQL with vector search support."""

    def __init__(self):
        self.pool: Optional[object] = None
        self.initialized = False
        
        # Connection settings from AppConfig
        self.host = config.database.host
        self.port = config.database.port
        self.user = config.database.user
        self.password = config.database.password
        self.database = config.database.database

    async def initialize(self):
        """Initialize database connection pool."""
        if not POSTGRES_AVAILABLE:
            logger.warning("PostgreSQL drivers (asyncpg) not available")
            return False

        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                min_size=config.database.pool_min_size if hasattr(config.database, 'pool_min_size') else 10,
                max_size=config.database.pool_max_size if hasattr(config.database, 'pool_max_size') else 30,
            )
            
            logger.info(f"✓ PostgreSQL pool created at {self.host}:{self.port}")
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            return False

    async def _verify_schema(self):
        """Verify that required tables exist (minimal implementation for compatibility)."""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            existing_tables = [t['table_name'] for t in tables]
            logger.info(f"Existing tables: {existing_tables}")

    @asynccontextmanager
    async def get_connection(self):
        """
        Context manager to acquire a database connection from the pool.
        
        Usage:
            async with db.get_connection() as conn:
                await conn.execute("SELECT 1")
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized. Call await initialize() first.")
        
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)

    async def execute_query(
        self, 
        query: str, 
        params: tuple = None,
        operation: Literal["read", "write"] = "write",
        fetch_one: bool = False,
        fetch_all: bool = False
    ):
        """Execute query with asyncpg."""
        if not self.pool:
            return None
            
        # Convert %s to $1, $2, etc. for asyncpg
        query = self._convert_placeholders(query)
        
        async with self.pool.acquire() as conn:
            if fetch_one:
                result = await conn.fetchrow(query, *(params or ()))
                return dict(result) if result else None
            elif fetch_all:
                results = await conn.fetch(query, *(params or ()))
                return [dict(r) for r in results]
            else:
                return await conn.execute(query, *(params or ()))

    def _convert_placeholders(self, query: str) -> str:
        """Convert MySQL %s placeholders to PostgreSQL $1, $2, etc."""
        count = 1
        while '%s' in query:
            query = query.replace('%s', f'${count}', 1)
            count += 1
        return query

    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        return await self.execute_query(query, params, operation="read", fetch_all=True) or []

    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        return await self.execute_query(query, params, operation="read", fetch_one=True)

    async def execute_select(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute SELECT with named parameters."""
        if params is None:
            params = {}
        
        # Find all :param_name occurrences
        param_names = re.findall(r':(\w+)', query)
        param_values = tuple(params.get(name) for name in param_names)
        
        # Convert :param_name to $1, $2, etc.
        converted_query = query
        for i, name in enumerate(param_names):
            converted_query = converted_query.replace(f':{name}', f'${i+1}')
            
        return await self.fetch_all(converted_query, param_values)

    async def store_vitals(
        self,
        user_id: str,
        device_id: str,
        metric_type: str,
        value: float,
        unit: str = "",
    ) -> bool:
        try:
            await self.execute_query(
                "INSERT INTO vitals (user_id, device_id, metric_type, value, unit) VALUES (%s, %s, %s, %s, %s)",
                (user_id, device_id, metric_type, value, unit)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store vitals: {e}")
            return False

    async def get_user_vitals_history(
        self, user_id: str, metric_type: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            query = "SELECT device_id, metric_type, value, unit, recorded_at FROM vitals WHERE user_id = %s "
            params = [user_id]
            if metric_type:
                query += "AND metric_type = %s "
                params.append(metric_type)
            query += "ORDER BY recorded_at DESC LIMIT %s"
            params.append(limit)
            
            results = await self.fetch_all(query, tuple(params))
            return [
                {
                    "device_id": row["device_id"],
                    "metric_type": row["metric_type"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "timestamp": row["recorded_at"],
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve vitals: {e}")
            return []

    async def store_medical_knowledge(
        self,
        content: str,
        content_type: str,
        embedding: List[float],
        metadata: Dict = None,
    ) -> bool:
        try:
            # Store embedding as JSON string or array (vector search handled by ChromaDB), 
            # here we use JSONB for embedding as per schema
            await self.execute_query(
                "INSERT INTO medical_knowledge_base (content, content_type, embedding, metadata) VALUES ($1, $2, $3, $4)",
                (content, content_type, json.dumps(embedding), json.dumps(metadata or {}))
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            return False

    async def search_similar_knowledge(
        self, query_embedding: List[float], content_type: str = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar knowledge using vector similarity.
        
        **SECURITY**: Application-level fallback is restricted to prevent DoS.
        
        Note: Primary vector search is handled by ChromaDB (rag/chromadb_store.py).
        This is a PostgreSQL-level fallback for the medical_knowledge_base table.
        
        In development: Uses application-level similarity with a HARD LIMIT of 1000 rows.
        In production: Should use ChromaDB vector store instead.
        
        This prevents memory exhaustion from fetching entire knowledge bases into memory.
        """
        try:
            # Application-level similarity (fallback only; use ChromaDB for production)
            
            is_production = os.getenv("APP_ENV", "development").lower() == "production"
            
            # Build query with strict LIMIT to prevent DoS
            SAFE_LIMIT = 1000  # Max rows to fetch for application-level similarity
            
            query = f"SELECT id, content, content_type, metadata, embedding FROM medical_knowledge_base "
            params = []
            if content_type:
                query += "WHERE content_type = $1 "
                params.append(content_type)
            
            # Add HARD LIMIT to prevent full table scan
            query += f"LIMIT {SAFE_LIMIT}"
            
            results = await self.fetch_all(query, tuple(params))
            
            # If we got no results in production, warn
            if not results and is_production:
                logger.error(
                    "CRITICAL: Vector search returning no results in production. "
                    "Consider using ChromaDB vector store (rag/chromadb_store.py) "
                    "for production vector search."
                )
                return []
            
            # Application-level similarity (development/fallback only)
            scored_results = []
            q_vec = np.array(query_embedding)
            
            for row in results:
                try:
                    row_vec = np.array(json.loads(row['embedding']))
                    
                    # Prevent division by zero
                    q_norm = np.linalg.norm(q_vec)
                    row_norm = np.linalg.norm(row_vec)
                    if q_norm == 0 or row_norm == 0:
                        continue
                    
                    similarity = np.dot(q_vec, row_vec) / (q_norm * row_norm)
                    scored_results.append({
                        "id": row['id'],
                        "content": row['content'],
                        "content_type": row['content_type'],
                        "metadata": row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'] or '{}'),
                        "similarity": float(similarity)
                    })
                except Exception as e:
                    logger.debug(f"Failed to compute similarity for row: {e}")
                    continue
            
            scored_results.sort(key=lambda x: x['similarity'], reverse=True)
            return scored_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []


    async def store_chat_message(
        self, session_id: str, message_type: str, content: str, metadata: Dict = None
    ) -> bool:
        """Store chat message in the database."""
        if not self.pool:
            return False

        try:
            # Ensure session exists first
            await self.execute_query(
                """
                INSERT INTO chat_sessions (session_id, user_id)
                VALUES ($1, $2)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, "user123"), # Default to user123 if not provided
            )
            
            await self.execute_query(
                """
                INSERT INTO chat_messages (session_id, message_type, content, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                (session_id, message_type, content, json.dumps(metadata or {})),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store chat message: {e}")
            return False

    async def get_chat_history(
        self, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session."""
        if not self.pool:
            return []

        try:
            results = await self.fetch_all(
                """
                SELECT message_type, content, metadata, timestamp
                FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp ASC
                LIMIT $2
                """,
                (session_id, limit),
            )
            
            return [
                {
                    "message_type": row["message_type"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "timestamp": row["timestamp"],
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve chat history: {e}")
            return []

    async def get_pool_status(self) -> dict:
        """Get pool connection statistics."""
        if not self.pool:
            return {"error": "Database not initialized"}
        
        return {
            "min_size": self.pool._min_size,
            "max_size": self.pool._max_size,
            "current_size": len(self.pool._holders),
            "free": self.pool._queue.qsize(), # Approx
        }

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.initialized = False
            logger.info("PostgreSQL pool closed")

    async def _ensure_schema(self):
        """Ensure all required tables exist."""
        if not self.pool:
            return

        # Define tables to create
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255),
                email VARCHAR(255),
                date_of_birth DATE,
                gender VARCHAR(20),
                weight_kg FLOAT,
                height_cm FLOAT,
                known_conditions JSONB,
                medications JSONB,
                allergies JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS patient_records (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                record_type VARCHAR(100),
                data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS vitals (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                device_id VARCHAR(255),
                metric_type VARCHAR(50),
                value FLOAT,
                unit VARCHAR(20),
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS health_alerts (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                alert_type VARCHAR(50),
                severity VARCHAR(20),
                message TEXT,
                is_resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP NULL
            )
            """,
             """
            CREATE TABLE IF NOT EXISTS medical_knowledge_base (
                id SERIAL PRIMARY KEY,
                content TEXT,
                content_type VARCHAR(100),
                embedding JSONB, 
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) UNIQUE NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                message_type VARCHAR(50) NOT NULL,
                content TEXT,
                metadata JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS notification_failures (
                id SERIAL PRIMARY KEY,
                notification_type VARCHAR(50) NOT NULL,
                recipient VARCHAR(255) NOT NULL,
                subject VARCHAR(500),
                content TEXT,
                original_attempt_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retry_count INT DEFAULT 0,
                max_retries INT DEFAULT 5,
                next_retry_at TIMESTAMP,
                last_error TEXT,
                status VARCHAR(50) DEFAULT 'pending',
                user_id VARCHAR(255),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # ---------- Appointment System Tables ----------
            """
            CREATE TABLE IF NOT EXISTS providers (
                id SERIAL PRIMARY KEY,
                provider_id VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                specialty VARCHAR(100) NOT NULL,
                qualifications VARCHAR(255),
                rating FLOAT DEFAULT 0.0,
                review_count INT DEFAULT 0,
                photo_url VARCHAR(500),
                clinic_name VARCHAR(255),
                address TEXT,
                languages JSONB,
                telehealth_available BOOLEAN DEFAULT FALSE,
                accepted_insurances JSONB,
                bio TEXT,
                experience_years INT DEFAULT 0,
                accepts_new_patients BOOLEAN DEFAULT TRUE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS provider_availability (
                id SERIAL PRIMARY KEY,
                provider_id VARCHAR(50) NOT NULL,
                date VARCHAR(10) NOT NULL,
                time_slot VARCHAR(5) NOT NULL,
                is_booked BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id SERIAL PRIMARY KEY,
                appointment_id VARCHAR(100) UNIQUE NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                provider_id VARCHAR(50) NOT NULL,
                doctor_name VARCHAR(255) NOT NULL,
                specialty VARCHAR(100),
                doctor_rating FLOAT,
                date VARCHAR(10) NOT NULL,
                time VARCHAR(5) NOT NULL,
                duration_minutes INT DEFAULT 30,
                appointment_type VARCHAR(20) NOT NULL DEFAULT 'in-person',
                location VARCHAR(255),
                virtual_link VARCHAR(500),
                reason TEXT,
                intake_summary TEXT,
                consultation_summary TEXT,
                shared_chart_data JSONB,
                insurance_provider VARCHAR(255),
                insurance_member_id VARCHAR(100),
                insurance_group_id VARCHAR(100),
                status VARCHAR(20) NOT NULL DEFAULT 'scheduled',
                cancellation_reason TEXT,
                estimated_cost FLOAT DEFAULT 150.0,
                actual_cost FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS insurance_info (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                insurance_provider VARCHAR(255) NOT NULL,
                member_id VARCHAR(100) NOT NULL,
                group_id VARCHAR(100),
                plan_type VARCHAR(50),
                is_primary BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # ---------- User Devices (Smartwatch/Wearable) ----------
            """
            CREATE TABLE IF NOT EXISTS user_devices (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                device_id VARCHAR(255) UNIQUE NOT NULL,
                device_type VARCHAR(100),
                device_name VARCHAR(255),
                firmware_version VARCHAR(100),
                battery INT DEFAULT 100,
                status VARCHAR(50) DEFAULT 'connected',
                last_sync TIMESTAMP,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # ---------- Device Time-Series ----------
            """
            CREATE TABLE IF NOT EXISTS device_timeseries (
                id SERIAL PRIMARY KEY,
                device_id VARCHAR(255) NOT NULL,
                metric_type VARCHAR(50) NOT NULL,
                value FLOAT NOT NULL,
                unit VARCHAR(20) DEFAULT '',
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                idempotency_key VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # ---------- Consent System ----------
            """
            CREATE TABLE IF NOT EXISTS user_consents (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                consent_type VARCHAR(100) NOT NULL,
                granted BOOLEAN NOT NULL DEFAULT FALSE,
                description TEXT,
                required BOOLEAN DEFAULT FALSE,
                granted_at TIMESTAMP,
                revoked_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, consent_type)
            )
            """,
            # ---------- Calendar System ----------
            """
            CREATE TABLE IF NOT EXISTS calendar_credentials (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL UNIQUE,
                provider VARCHAR(50) DEFAULT 'google',
                access_token TEXT,
                refresh_token TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS calendar_events (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(100) UNIQUE NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                title VARCHAR(255) NOT NULL,
                start_time VARCHAR(30) NOT NULL,
                end_time VARCHAR(30) NOT NULL,
                location VARCHAR(255),
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS calendar_reminders (
                id SERIAL PRIMARY KEY,
                reminder_id VARCHAR(100) UNIQUE NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                appointment_id VARCHAR(100),
                title VARCHAR(255) NOT NULL,
                description TEXT,
                scheduled_for VARCHAR(30) NOT NULL,
                reminder_minutes_before INT DEFAULT 30,
                status VARCHAR(20) DEFAULT 'scheduled',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # ---------- Push Devices ----------
            """
            CREATE TABLE IF NOT EXISTS push_devices (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                device_token VARCHAR(500) NOT NULL,
                platform VARCHAR(20) NOT NULL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, device_token)
            )
            """,
            # ---------- Content Verification ----------
            """
            CREATE TABLE IF NOT EXISTS content_verifications (
                id SERIAL PRIMARY KEY,
                item_id VARCHAR(100) UNIQUE NOT NULL,
                content TEXT NOT NULL,
                content_type VARCHAR(50) NOT NULL,
                submitted_by VARCHAR(255),
                status VARCHAR(20) DEFAULT 'pending',
                reviewer_notes TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP
            )
            """,
            # ---------- Prediction History ----------
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id SERIAL PRIMARY KEY,
                prediction_id VARCHAR(100) UNIQUE NOT NULL,
                user_id VARCHAR(255),
                input_data JSONB,
                prediction INT,
                probability FLOAT,
                risk_level VARCHAR(20),
                confidence FLOAT,
                clinical_interpretation TEXT,
                processing_time_ms FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        try:
            async with self.pool.acquire() as conn:
                # 1. Run migrations for legacy schema compatibility
                # Check for metadata_json in chat_messages and rename to metadata
                try:
                    chat_cols = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = 'chat_messages'")
                    chat_col_names = [r['column_name'] for r in chat_cols]
                    
                    if 'metadata_json' in chat_col_names and 'metadata' not in chat_col_names:
                        logger.info("Migrating chat_messages: metadata_json -> metadata")
                        await conn.execute("ALTER TABLE chat_messages RENAME COLUMN metadata_json TO metadata")
                except Exception as e:
                    logger.warning(f"Migration check for chat_messages failed: {e}")

                # Check for metadata_json in medical_knowledge_base and rename to metadata
                try:
                    kb_cols = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = 'medical_knowledge_base'")
                    kb_col_names = [r['column_name'] for r in kb_cols]
                    
                    if 'metadata_json' in kb_col_names and 'metadata' not in kb_col_names:
                        logger.info("Migrating medical_knowledge_base: metadata_json -> metadata")
                        await conn.execute("ALTER TABLE medical_knowledge_base RENAME COLUMN metadata_json TO metadata")
                except Exception as e:
                    logger.warning(f"Migration check for medical_knowledge_base failed: {e}")

                # 2. Create tables if not exist
                for sql in tables_sql:
                    await conn.execute(sql)
                
                # Seed default user if not exists
                await conn.execute("""
                    INSERT INTO users (user_id, name, email, date_of_birth, gender, weight_kg, height_cm)
                    VALUES ('user123', 'John Doe', 'john@example.com', '1980-01-01', 'Male', 80.0, 180.0)
                    ON CONFLICT (user_id) DO NOTHING
                """)
                
            logger.info("Schema ensured (tables created/migrated)")
        except Exception as e:
            logger.error(f"Failed to ensure schema: {e}")

    # ========================================================================
    # Appointment System Database Methods
    # ========================================================================

    # --- Providers ---

    async def get_providers(
        self,
        specialty: str = None,
        search: str = None,
        accepts_new: bool = None,
        telehealth: bool = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get providers with optional filters."""
        try:
            query = "SELECT * FROM providers WHERE is_active = true"
            params = []
            idx = 1

            if specialty and specialty != 'All':
                query += f" AND specialty = ${idx}"
                params.append(specialty)
                idx += 1

            if search:
                query += f" AND (LOWER(name) LIKE ${idx} OR LOWER(specialty) LIKE ${idx})"
                params.append(f"%{search.lower()}%")
                idx += 1

            if accepts_new is not None:
                query += f" AND accepts_new_patients = ${idx}"
                params.append(accepts_new)
                idx += 1

            if telehealth is not None:
                query += f" AND telehealth_available = ${idx}"
                params.append(telehealth)
                idx += 1

            query += f" ORDER BY rating DESC LIMIT ${idx} OFFSET ${idx + 1}"
            params.extend([limit, offset])

            return await self.fetch_all(query, tuple(params))
        except Exception as e:
            logger.error(f"Failed to get providers: {e}")
            return []

    async def get_provider_by_id(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get a single provider by provider_id."""
        try:
            return await self.fetch_one(
                "SELECT * FROM providers WHERE provider_id = %s AND is_active = true",
                (provider_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get provider {provider_id}: {e}")
            return None

    async def get_provider_specialties(self) -> List[str]:
        """Get distinct specialties from active providers."""
        try:
            rows = await self.fetch_all(
                "SELECT DISTINCT specialty FROM providers WHERE is_active = true ORDER BY specialty"
            )
            return [r['specialty'] for r in rows]
        except Exception as e:
            logger.error(f"Failed to get specialties: {e}")
            return []

    # --- Availability ---

    async def get_provider_availability(
        self, provider_id: str, date: str
    ) -> List[str]:
        """Get available (unbooked) time slots for a provider on a date."""
        try:
            rows = await self.fetch_all(
                "SELECT time_slot FROM provider_availability "
                "WHERE provider_id = %s AND date = %s AND is_booked = false "
                "ORDER BY time_slot",
                (provider_id, date)
            )
            if rows:
                return [r['time_slot'] for r in rows]

            # If no slots in DB yet, generate default weekday slots
            from datetime import datetime as dt
            d = dt.strptime(date, "%Y-%m-%d")
            if d.weekday() < 5:  # Mon-Fri
                default_slots = ['09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '14:00', '14:30', '15:00', '15:30']
                # Insert them so they persist
                for slot in default_slots:
                    await self.execute_query(
                        "INSERT INTO provider_availability (provider_id, date, time_slot, is_booked) "
                        "VALUES (%s, %s, %s, false) ON CONFLICT DO NOTHING",
                        (provider_id, date, slot)
                    )
                return default_slots
            return []
        except Exception as e:
            logger.error(f"Failed to get availability for {provider_id} on {date}: {e}")
            return []

    async def book_slot(self, provider_id: str, date: str, time_slot: str) -> bool:
        """Mark a time slot as booked."""
        try:
            result = await self.execute_query(
                "UPDATE provider_availability SET is_booked = true "
                "WHERE provider_id = %s AND date = %s AND time_slot = %s AND is_booked = false",
                (provider_id, date, time_slot)
            )
            # asyncpg returns 'UPDATE N' — check if at least 1 row was affected
            return result and '0' not in result
        except Exception as e:
            logger.error(f"Failed to book slot: {e}")
            return False

    async def release_slot(self, provider_id: str, date: str, time_slot: str) -> bool:
        """Release a booked time slot (for cancellation)."""
        try:
            await self.execute_query(
                "UPDATE provider_availability SET is_booked = false "
                "WHERE provider_id = %s AND date = %s AND time_slot = %s",
                (provider_id, date, time_slot)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to release slot: {e}")
            return False

    # --- Appointments ---

    async def create_appointment(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new appointment record."""
        try:
            import json as _json
            shared_data = data.get('shared_chart_data')
            if shared_data and isinstance(shared_data, dict):
                shared_data = _json.dumps(shared_data)

            await self.execute_query(
                """INSERT INTO appointments (
                    appointment_id, user_id, provider_id, doctor_name, specialty,
                    doctor_rating, date, time, duration_minutes, appointment_type,
                    location, virtual_link, reason, intake_summary, shared_chart_data,
                    insurance_provider, insurance_member_id, insurance_group_id,
                    status, estimated_cost
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                )""",
                (
                    data['appointment_id'], data['user_id'], data['provider_id'],
                    data['doctor_name'], data.get('specialty'),
                    data.get('doctor_rating'), data['date'], data['time'],
                    data.get('duration_minutes', 30), data.get('appointment_type', 'in-person'),
                    data.get('location'), data.get('virtual_link'),
                    data.get('reason'), data.get('intake_summary'),
                    shared_data,
                    data.get('insurance_provider'), data.get('insurance_member_id'),
                    data.get('insurance_group_id'),
                    data.get('status', 'scheduled'), data.get('estimated_cost', 150.0),
                )
            )

            # Mark the time slot as booked
            await self.book_slot(data['provider_id'], data['date'], data['time'])

            return await self.get_appointment_by_id(data['appointment_id'])
        except Exception as e:
            logger.error(f"Failed to create appointment: {e}")
            return None

    async def get_appointment_by_id(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Get a single appointment by appointment_id."""
        try:
            return await self.fetch_one(
                "SELECT * FROM appointments WHERE appointment_id = %s",
                (appointment_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get appointment {appointment_id}: {e}")
            return None

    async def get_user_appointments(
        self,
        user_id: str,
        status: str = None,
        upcoming_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get appointments for a user."""
        try:
            query = "SELECT * FROM appointments WHERE user_id = %s"
            params: list = [user_id]
            idx = 2

            if status:
                query += f" AND status = ${idx}"
                params.append(status)
                idx += 1

            if upcoming_only:
                query += f" AND date >= ${idx} AND status NOT IN ('cancelled', 'completed')"
                from datetime import date as dt_date
                params.append(dt_date.today().isoformat())
                idx += 1

            query += f" ORDER BY date ASC, time ASC LIMIT ${idx} OFFSET ${idx + 1}"
            params.extend([limit, offset])

            return await self.fetch_all(query, tuple(params))
        except Exception as e:
            logger.error(f"Failed to get appointments for {user_id}: {e}")
            return []

    async def update_appointment(
        self, appointment_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update appointment fields."""
        try:
            allowed = {
                'status', 'cancellation_reason', 'consultation_summary',
                'reason', 'intake_summary', 'actual_cost', 'appointment_type',
                'insurance_provider', 'insurance_member_id', 'insurance_group_id',
                'virtual_link', 'location',
            }
            filtered = {k: v for k, v in updates.items() if k in allowed}
            if not filtered:
                return await self.get_appointment_by_id(appointment_id)

            set_clauses = []
            params = []
            idx = 1
            for key, val in filtered.items():
                set_clauses.append(f"{key} = ${idx}")
                params.append(val)
                idx += 1

            set_clauses.append(f"updated_at = CURRENT_TIMESTAMP")
            query = f"UPDATE appointments SET {', '.join(set_clauses)} WHERE appointment_id = ${idx}"
            params.append(appointment_id)

            await self.execute_query(query, tuple(params))
            return await self.get_appointment_by_id(appointment_id)
        except Exception as e:
            logger.error(f"Failed to update appointment {appointment_id}: {e}")
            return None

    async def cancel_appointment(
        self, appointment_id: str, reason: str = None
    ) -> Optional[Dict[str, Any]]:
        """Cancel an appointment and release the time slot."""
        try:
            appt = await self.get_appointment_by_id(appointment_id)
            if not appt:
                return None

            updates = {'status': 'cancelled'}
            if reason:
                updates['cancellation_reason'] = reason

            result = await self.update_appointment(appointment_id, updates)

            # Release the booked slot
            await self.release_slot(appt['provider_id'], appt['date'], appt['time'])

            return result
        except Exception as e:
            logger.error(f"Failed to cancel appointment {appointment_id}: {e}")
            return None

    async def complete_appointment(
        self, appointment_id: str, summary: str = None
    ) -> Optional[Dict[str, Any]]:
        """Mark appointment as completed with optional clinical summary."""
        try:
            updates: Dict[str, Any] = {'status': 'completed'}
            if summary:
                updates['consultation_summary'] = summary
            return await self.update_appointment(appointment_id, updates)
        except Exception as e:
            logger.error(f"Failed to complete appointment {appointment_id}: {e}")
            return None

    # --- Insurance ---

    async def save_insurance(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Save or update insurance info for a user."""
        try:
            existing = await self.fetch_one(
                "SELECT id FROM insurance_info WHERE user_id = %s AND insurance_provider = %s",
                (data['user_id'], data['insurance_provider'])
            )
            if existing:
                await self.execute_query(
                    "UPDATE insurance_info SET member_id = %s, group_id = %s, "
                    "plan_type = %s, is_verified = %s, updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = %s",
                    (data.get('member_id'), data.get('group_id'),
                     data.get('plan_type'), data.get('is_verified', False),
                     existing['id'])
                )
            else:
                await self.execute_query(
                    "INSERT INTO insurance_info (user_id, insurance_provider, member_id, group_id, plan_type, is_verified) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (data['user_id'], data['insurance_provider'],
                     data.get('member_id'), data.get('group_id'),
                     data.get('plan_type'), data.get('is_verified', False))
                )
            return await self.fetch_one(
                "SELECT * FROM insurance_info WHERE user_id = %s AND insurance_provider = %s",
                (data['user_id'], data['insurance_provider'])
            )
        except Exception as e:
            logger.error(f"Failed to save insurance: {e}")
            return None

    async def get_user_insurance(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all insurance info for a user."""
        try:
            return await self.fetch_all(
                "SELECT * FROM insurance_info WHERE user_id = %s ORDER BY is_primary DESC",
                (user_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get insurance for {user_id}: {e}")
            return []

    # ========================================================================
    # Consent System Database Methods
    # ========================================================================

    async def get_user_consents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all consent records for a user."""
        try:
            return await self.fetch_all(
                "SELECT * FROM user_consents WHERE user_id = %s ORDER BY consent_type",
                (user_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get consents for {user_id}: {e}")
            return []

    async def upsert_consent(self, user_id: str, consent_type: str, granted: bool,
                             description: str = None, required: bool = False) -> Optional[Dict[str, Any]]:
        """Insert or update a consent record."""
        try:
            now_ts = "CURRENT_TIMESTAMP"
            granted_at_val = "CURRENT_TIMESTAMP" if granted else "NULL"
            revoked_at_val = "NULL" if granted else "CURRENT_TIMESTAMP"
            result = await self.fetch_one(
                """INSERT INTO user_consents (user_id, consent_type, granted, description, required,
                       granted_at, revoked_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s,
                       CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE NULL END,
                       CASE WHEN %s THEN NULL ELSE CURRENT_TIMESTAMP END,
                       CURRENT_TIMESTAMP)
                   ON CONFLICT (user_id, consent_type) DO UPDATE SET
                       granted = EXCLUDED.granted,
                       description = COALESCE(EXCLUDED.description, user_consents.description),
                       granted_at = CASE WHEN EXCLUDED.granted THEN CURRENT_TIMESTAMP ELSE user_consents.granted_at END,
                       revoked_at = CASE WHEN EXCLUDED.granted THEN NULL ELSE CURRENT_TIMESTAMP END,
                       updated_at = CURRENT_TIMESTAMP
                   RETURNING *""",
                (user_id, consent_type, granted, description, required, granted, granted)
            )
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to upsert consent {consent_type} for {user_id}: {e}")
            return None

    async def revoke_consent(self, user_id: str, consent_type: str) -> bool:
        """Revoke a specific consent."""
        try:
            result = await self.execute_query(
                """UPDATE user_consents SET granted = FALSE, revoked_at = CURRENT_TIMESTAMP,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = %s AND consent_type = %s AND required = FALSE""",
                (user_id, consent_type)
            )
            return result is not None
        except Exception as e:
            logger.error(f"Failed to revoke consent {consent_type} for {user_id}: {e}")
            return False

    # ========================================================================
    # Calendar System Database Methods
    # ========================================================================

    async def save_calendar_credentials(self, user_id: str, provider: str,
                                         access_token: str = None, refresh_token: str = None) -> Optional[Dict[str, Any]]:
        """Save or update calendar credentials."""
        try:
            return await self.fetch_one(
                """INSERT INTO calendar_credentials (user_id, provider, access_token, refresh_token, updated_at)
                   VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                   ON CONFLICT (user_id) DO UPDATE SET
                       provider = EXCLUDED.provider,
                       access_token = COALESCE(EXCLUDED.access_token, calendar_credentials.access_token),
                       refresh_token = COALESCE(EXCLUDED.refresh_token, calendar_credentials.refresh_token),
                       updated_at = CURRENT_TIMESTAMP
                   RETURNING *""",
                (user_id, provider, access_token, refresh_token)
            )
        except Exception as e:
            logger.error(f"Failed to save calendar credentials for {user_id}: {e}")
            return None

    async def get_calendar_credentials(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get calendar credentials for a user."""
        try:
            return await self.fetch_one(
                "SELECT * FROM calendar_credentials WHERE user_id = %s",
                (user_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get calendar credentials for {user_id}: {e}")
            return None

    async def save_calendar_event(self, event_id: str, user_id: str, title: str,
                                   start_time: str, end_time: str,
                                   location: str = None, description: str = None) -> Optional[Dict[str, Any]]:
        """Save a calendar event."""
        try:
            return await self.fetch_one(
                """INSERT INTO calendar_events (event_id, user_id, title, start_time, end_time, location, description)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (event_id) DO UPDATE SET
                       title = EXCLUDED.title, start_time = EXCLUDED.start_time,
                       end_time = EXCLUDED.end_time, location = EXCLUDED.location,
                       description = EXCLUDED.description
                   RETURNING *""",
                (event_id, user_id, title, start_time, end_time, location, description)
            )
        except Exception as e:
            logger.error(f"Failed to save calendar event {event_id}: {e}")
            return None

    async def get_calendar_events(self, user_id: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get calendar events for a user, optionally filtered by date range."""
        try:
            query = "SELECT * FROM calendar_events WHERE user_id = %s"
            params = [user_id]
            if start_date:
                query += " AND start_time >= %s"
                params.append(start_date)
            if end_date:
                query += " AND start_time <= %s"
                params.append(end_date)
            query += " ORDER BY start_time"
            return await self.fetch_all(query, tuple(params))
        except Exception as e:
            logger.error(f"Failed to get calendar events for {user_id}: {e}")
            return []

    async def delete_calendar_events(self, user_id: str) -> bool:
        """Delete all calendar events for a user (before re-sync)."""
        try:
            await self.execute_query("DELETE FROM calendar_events WHERE user_id = %s", (user_id,))
            return True
        except Exception as e:
            logger.error(f"Failed to delete calendar events for {user_id}: {e}")
            return False

    async def save_calendar_reminder(self, reminder_id: str, user_id: str, appointment_id: str,
                                      title: str, scheduled_for: str, description: str = None,
                                      reminder_minutes_before: int = 30) -> Optional[Dict[str, Any]]:
        """Save a calendar reminder."""
        try:
            return await self.fetch_one(
                """INSERT INTO calendar_reminders
                       (reminder_id, user_id, appointment_id, title, description, scheduled_for, reminder_minutes_before)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (reminder_id, user_id, appointment_id, title, description, scheduled_for, reminder_minutes_before)
            )
        except Exception as e:
            logger.error(f"Failed to save calendar reminder {reminder_id}: {e}")
            return None

    async def get_calendar_reminders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all reminders for a user."""
        try:
            return await self.fetch_all(
                "SELECT * FROM calendar_reminders WHERE user_id = %s ORDER BY scheduled_for",
                (user_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get calendar reminders for {user_id}: {e}")
            return []

    # ========================================================================
    # Push Device Database Methods
    # ========================================================================

    async def register_push_device(self, user_id: str, device_token: str, platform: str) -> Optional[Dict[str, Any]]:
        """Register a push notification device."""
        try:
            return await self.fetch_one(
                """INSERT INTO push_devices (user_id, device_token, platform)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (user_id, device_token) DO UPDATE SET
                       platform = EXCLUDED.platform,
                       registered_at = CURRENT_TIMESTAMP
                   RETURNING *""",
                (user_id, device_token, platform)
            )
        except Exception as e:
            logger.error(f"Failed to register push device for {user_id}: {e}")
            return None

    async def get_user_push_devices(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all push devices for a user."""
        try:
            return await self.fetch_all(
                "SELECT * FROM push_devices WHERE user_id = %s ORDER BY registered_at DESC",
                (user_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get push devices for {user_id}: {e}")
            return []

    async def count_user_push_devices(self, user_id: str) -> int:
        """Count push devices for a user."""
        try:
            result = await self.fetch_one(
                "SELECT COUNT(*) as cnt FROM push_devices WHERE user_id = %s",
                (user_id,)
            )
            return result.get('cnt', 0) if result else 0
        except Exception:
            return 0

    async def unregister_push_device(self, user_id: str, device_token: str) -> bool:
        """Unregister a push device."""
        try:
            await self.execute_query(
                "DELETE FROM push_devices WHERE user_id = %s AND device_token = %s",
                (user_id, device_token)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to unregister push device for {user_id}: {e}")
            return False

    async def log_notification_failure(self, notification_type: str, recipient: str,
                                        subject: str = None, content: str = None,
                                        error: str = None, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Log a notification delivery failure for retry."""
        try:
            return await self.fetch_one(
                """INSERT INTO notification_failures
                       (notification_type, recipient, subject, content, last_error, user_id, status)
                   VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                   RETURNING *""",
                (notification_type, recipient, subject, content, error, user_id)
            )
        except Exception as e:
            logger.error(f"Failed to log notification failure: {e}")
            return None

    # ========================================================================
    # Content Verification Database Methods
    # ========================================================================

    async def add_verification_item(self, item_id: str, content: str, content_type: str,
                                     submitted_by: str = None) -> Optional[Dict[str, Any]]:
        """Add a content verification item to the queue."""
        try:
            return await self.fetch_one(
                """INSERT INTO content_verifications (item_id, content, content_type, submitted_by)
                   VALUES (%s, %s, %s, %s) RETURNING *""",
                (item_id, content, content_type, submitted_by)
            )
        except Exception as e:
            logger.error(f"Failed to add verification item {item_id}: {e}")
            return None

    async def get_pending_verifications(self) -> List[Dict[str, Any]]:
        """Get all pending content verification items."""
        try:
            return await self.fetch_all(
                "SELECT * FROM content_verifications WHERE status = 'pending' ORDER BY submitted_at"
            )
        except Exception as e:
            logger.error(f"Failed to get pending verifications: {e}")
            return []

    async def submit_verification_decision(self, item_id: str, verified: bool,
                                            reviewer_notes: str = None) -> Optional[Dict[str, Any]]:
        """Submit a verification decision."""
        try:
            status = "verified" if verified else "rejected"
            return await self.fetch_one(
                """UPDATE content_verifications
                   SET status = %s, reviewer_notes = %s, reviewed_at = CURRENT_TIMESTAMP
                   WHERE item_id = %s RETURNING *""",
                (status, reviewer_notes, item_id)
            )
        except Exception as e:
            logger.error(f"Failed to submit verification for {item_id}: {e}")
            return None

    # ========================================================================
    # Prediction History Database Methods
    # ========================================================================

    async def store_prediction(self, prediction_id: str, user_id: str, input_data: dict,
                                prediction: int, probability: float, risk_level: str,
                                confidence: float, clinical_interpretation: str = None,
                                processing_time_ms: float = None) -> Optional[Dict[str, Any]]:
        """Store a heart disease prediction result."""
        try:
            import json
            return await self.fetch_one(
                """INSERT INTO prediction_history
                       (prediction_id, user_id, input_data, prediction, probability,
                        risk_level, confidence, clinical_interpretation, processing_time_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (prediction_id, user_id, json.dumps(input_data), prediction, probability,
                 risk_level, confidence, clinical_interpretation, processing_time_ms)
            )
        except Exception as e:
            logger.error(f"Failed to store prediction {prediction_id}: {e}")
            return None

    async def get_prediction_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get prediction history for a user."""
        try:
            return await self.fetch_all(
                "SELECT * FROM prediction_history WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                (user_id, limit)
            )
        except Exception as e:
            logger.error(f"Failed to get prediction history for {user_id}: {e}")
            return []

    # ========================================================================
    # Smartwatch / Vitals Extended Methods
    # ========================================================================

    async def register_device(self, device_id: str, user_id: str, device_type: str,
                               device_name: str = None, firmware_version: str = None) -> Optional[Dict[str, Any]]:
        """Register a smartwatch/wearable device in the user_devices table."""
        try:
            return await self.fetch_one(
                """INSERT INTO user_devices (user_id, device_type, device_name, device_id, firmware_version, status)
                   VALUES (%s, %s, %s, %s, %s, 'active')
                   ON CONFLICT DO NOTHING
                   RETURNING *""",
                (user_id, device_type, device_name or device_type, device_id, firmware_version)
            )
        except Exception as e:
            logger.error(f"Failed to register device {device_id}: {e}")
            return None

    async def store_device_timeseries(self, device_id: str, user_id: str,
                                       metric_type: str, value: float,
                                       unit: str = '', timestamp: str = None) -> bool:
        """Store a single vitals data point in device_timeseries."""
        try:
            ts = timestamp or "CURRENT_TIMESTAMP"
            await self.execute_query(
                """INSERT INTO device_timeseries (device_id, metric_type, value, unit, recorded_at)
                   VALUES (%s, %s, %s, %s, COALESCE(%s::TIMESTAMP, CURRENT_TIMESTAMP))""",
                (device_id, metric_type, value, unit, timestamp)
            )
            # Also store in vitals table for user-centric queries
            await self.store_vitals(user_id, device_id, metric_type, value, unit)
            return True
        except Exception as e:
            logger.error(f"Failed to store timeseries for device {device_id}: {e}")
            return False

    async def get_device_timeseries(self, device_id: str, metric_type: str,
                                     hours: int = 24) -> List[Dict[str, Any]]:
        """Get time-series vitals for a device within a time window."""
        try:
            return await self.fetch_all(
                """SELECT * FROM device_timeseries
                   WHERE device_id = %s AND metric_type = %s
                     AND recorded_at >= (CURRENT_TIMESTAMP - INTERVAL '%s hours')
                   ORDER BY recorded_at""",
                (device_id, metric_type, hours)
            )
        except Exception as e:
            logger.error(f"Failed to get timeseries for device {device_id}: {e}")
            return []

    # ========================================================================
    # Weekly Summary / Integrations Query Methods
    # ========================================================================

    async def get_weekly_vitals_summary(self, user_id: str) -> Dict[str, Any]:
        """Get aggregated vitals for the current week."""
        try:
            result = await self.fetch_one(
                """SELECT
                       COUNT(*) FILTER (WHERE metric_type = 'heart_rate') as hr_count,
                       AVG(value) FILTER (WHERE metric_type = 'heart_rate') as avg_heart_rate,
                       COUNT(*) FILTER (WHERE metric_type = 'blood_pressure') as bp_count,
                       AVG(value) FILTER (WHERE metric_type = 'steps') as avg_steps,
                       COUNT(*) as total_readings
                   FROM vitals
                   WHERE user_id = %s AND recorded_at >= (CURRENT_DATE - INTERVAL '7 days')""",
                (user_id,)
            )
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Failed to get weekly vitals summary for {user_id}: {e}")
            return {}

    async def get_user_medications_list(self, user_id: str) -> List[Dict[str, Any]]:
        """Get current medications for a user."""
        try:
            return await self.fetch_all(
                "SELECT * FROM medications WHERE user_id = %s ORDER BY name",
                (user_id,)
            )
        except Exception as e:
            logger.error(f"Failed to get medications for {user_id}: {e}")
            return []

    async def get_user_conditions(self, user_id: str) -> List[str]:
        """Get known conditions for a user from the users table."""
        try:
            user = await self.fetch_one(
                "SELECT known_conditions FROM users WHERE user_id = %s",
                (user_id,)
            )
            if user and user.get('known_conditions'):
                conditions = user['known_conditions']
                if isinstance(conditions, list):
                    return conditions
                if isinstance(conditions, str):
                    import json
                    return json.loads(conditions)
            return []
        except Exception as e:
            logger.error(f"Failed to get conditions for {user_id}: {e}")
            return []

    async def get_recent_appointments_count(self, user_id: str) -> int:
        """Count appointments in the last 30 days."""
        try:
            result = await self.fetch_one(
                """SELECT COUNT(*) as cnt FROM appointments
                   WHERE user_id = %s AND created_at >= (CURRENT_DATE - INTERVAL '30 days')""",
                (user_id,)
            )
            return result.get('cnt', 0) if result else 0
        except Exception as e:
            return 0

    async def seed_default_providers(self):
        """Seed provider data if table is empty."""
        try:
            count = await self.fetch_one("SELECT COUNT(*) as cnt FROM providers")
            if count and count.get('cnt', 0) > 0:
                return

            providers = [
                ('p_101', 'Jane Doe', 'Cardiologist', 'MD, PhD', 4.9, 128,
                 'https://randomuser.me/api/portraits/women/44.jpg',
                 'Heart Health Center', '123 Health Ave, New York, NY',
                 '["English", "Spanish"]', True, '["Aetna", "BlueCross"]',
                 'Dr. Jane Doe is a board-certified cardiologist with over 15 years of experience.', 15, True),
                ('p_102', 'John Smith', 'Nutritionist', 'RD, CDN', 4.8, 97,
                 'https://randomuser.me/api/portraits/men/32.jpg',
                 'Wellness Dietetics', '456 Food St, San Francisco, CA',
                 '["English"]', True, '["UnitedHealthcare"]',
                 'John Smith is a registered dietitian specializing in heart health nutrition.', 10, True),
                ('p_103', 'Emily White', 'Cardiologist', 'MD', 4.7, 81,
                 'https://randomuser.me/api/portraits/women/68.jpg',
                 'CardioCare Clinic', '789 Heart Blvd, Chicago, IL',
                 '["English", "French"]', False, '["Aetna", "Cigna"]',
                 'Dr. Emily White is a compassionate cardiologist focused on preventive care.', 8, True),
                ('p_104', 'Sarah Connor', 'Cardiologist', 'MD, FACC', 4.9, 210,
                 'https://randomuser.me/api/portraits/women/45.jpg',
                 'Advanced Heart Institute', '500 Medical Center Dr, Boston, MA',
                 '["English", "German"]', True, '["BlueCross", "Medicare", "Aetna"]',
                 'Dr. Connor specializes in interventional cardiology and structural heart disease.', 18, True),
                ('p_105', 'Michael Chen', 'Sports Cardiologist', 'MD, MPH', 4.8, 156,
                 'https://randomuser.me/api/portraits/men/62.jpg',
                 'Elite Performance Cardio', '88 Victory Lane, Los Angeles, CA',
                 '["English", "Mandarin"]', True, '["Cigna", "UnitedHealthcare"]',
                 'Dr. Chen focuses on cardiovascular health in athletes.', 12, True),
            ]
            for p in providers:
                await self.execute_query(
                    """INSERT INTO providers (
                        provider_id, name, specialty, qualifications, rating, review_count,
                        photo_url, clinic_name, address, languages, telehealth_available,
                        accepted_insurances, bio, experience_years, accepts_new_patients
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (provider_id) DO NOTHING""",
                    p
                )
            logger.info(f"Seeded {len(providers)} default providers")
        except Exception as e:
            logger.error(f"Failed to seed providers: {e}")

# ============================================================================
# SINGLETON INSTANCE & FACTORY FUNCTION
# ============================================================================

_postgres_instance: Optional[PostgresDatabase] = None

async def get_database() -> PostgresDatabase:
    """Get or create the PostgreSQL database singleton instance."""
    global _postgres_instance
    
    if _postgres_instance is None:
        _postgres_instance = PostgresDatabase()
        await _postgres_instance.initialize()
        await _postgres_instance._ensure_schema()
        await _postgres_instance.seed_default_providers()
    
    return _postgres_instance
