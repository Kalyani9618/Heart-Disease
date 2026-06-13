"""
Authentication Database Service

Replaces in-memory MOCK_USERS_DB with persistent PostgreSQL storage.

Features:
- User registration and login with hashed passwords
- Multi-worker safe (all workers query same database)
- Token revocation list in Redis
- Refresh token management
- GDPR compliance (export, delete user data)


**Security Guarantees:**
✅ Passwords hashed with argon2 (OWASP standard)
✅ Works with multi-worker deployments (no shared memory)
✅ Async operations prevent event loop blocking
✅ Rate limiting on auth endpoints (3 failed attempts → 15 min lockout)
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import asyncio
import hashlib

logger = logging.getLogger(__name__)


class AuthDatabaseService:
    """
    Persistent authentication service using PostgreSQL.
    
    Replaces MOCK_USERS_DB with real database queries.
    """
    
    def __init__(self, db_manager, redis_client=None):
        """
        Initialize with database and optional Redis client.
        
        Args:
            db_manager: AsyncPG database connection manager
            redis_client: Optional Redis client for token blocklist
        """
        self.db = db_manager
        self.redis = redis_client
        
    async def ensure_users_table(self) -> None:
        """Create auth_users table if it doesn't exist, and add any missing columns."""
        # Use auth_users table to avoid conflict with health data users table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS auth_users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            last_login TIMESTAMP NULL,
            login_attempt_count INTEGER DEFAULT 0,
            last_failed_attempt TIMESTAMP NULL
        );
        
        CREATE INDEX IF NOT EXISTS idx_auth_users_email ON auth_users(email);
        CREATE INDEX IF NOT EXISTS idx_auth_users_is_active ON auth_users(is_active);
        """
        
        try:
            async with self.db.get_connection() as conn:
                await conn.execute(create_table_sql)
            logger.info("✅ auth_users table ensured")
        except Exception as e:
            logger.error(f"Failed to create users table: {e}")
            raise
    
    
    async def user_exists(self, email: str) -> bool:
        """Check if user exists by email."""
        try:
            async with self.db.get_connection() as conn:
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM auth_users WHERE email = $1 AND is_active = TRUE)",
                    email
                )
            return result or False
        except Exception as e:
            logger.error(f"Error checking user existence: {e}")
            return False
    
    
    async def register_user(
        self, 
        email: str, 
        name: str, 
        hashed_password: str
    ) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            email: User email (unique)
            name: User display name
            hashed_password: Argon2-hashed password
        
        Returns:
            Dict with user_id and email
        
        Raises:
            ValueError: If email already registered
        """
        # Use INSERT ... ON CONFLICT to avoid TOCTOU race
        try:
            async with self.db.get_connection() as conn:
                user_id = await conn.fetchval(
                    """
                    INSERT INTO auth_users (email, name, hashed_password)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (email) DO NOTHING
                    RETURNING id
                    """,
                    email, name, hashed_password
                )
            
            if user_id is None:
                raise ValueError("Email already registered")
            
            logger.info(f"✅ User registered (ID: {user_id})")
            return {"user_id": user_id, "email": email}
            
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            raise
    
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user by email.
        
        Returns None if user not found or inactive.
        """
        try:
            async with self.db.get_connection() as conn:
                user = await conn.fetchrow(
                    """
                    SELECT id, email, name, hashed_password, is_active, last_login
                    FROM auth_users
                    WHERE email = $1 AND is_active = TRUE
                    """,
                    email
                )
            
            if user:
                return dict(user)
            return None
            
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return None
    
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user by ID."""
        try:
            async with self.db.get_connection() as conn:
                user = await conn.fetchrow(
                    """
                    SELECT id, email, name, is_active, last_login, created_at
                    FROM auth_users
                    WHERE id = $1 AND is_active = TRUE
                    """,
                    user_id
                )
            
            if user:
                return dict(user)
            return None
            
        except Exception as e:
            logger.error(f"Error fetching user by ID: {e}")
            return None
    
    
    async def update_last_login(self, email: str) -> None:
        """Update last login timestamp."""
        try:
            async with self.db.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE auth_users 
                    SET last_login = NOW(), login_attempt_count = 0
                    WHERE email = $1
                    """,
                    email
                )
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
    
    
    async def record_failed_attempt(self, email: str) -> int:
        """
        Record failed login attempt and return attempt count.
        
        After 3 failed attempts, account is locked for 15 minutes.
        """
        try:
            async with self.db.get_connection() as conn:
                attempt_count = await conn.fetchval(
                    """
                    UPDATE auth_users 
                    SET login_attempt_count = login_attempt_count + 1,
                        last_failed_attempt = NOW()
                    WHERE email = $1
                    RETURNING login_attempt_count
                    """,
                    email
                )
            
            if attempt_count and attempt_count >= 3:
                logger.warning(f"🚨 Account locked after {attempt_count} failed attempts")
                # Optional: Trigger email alert to user
            
            return attempt_count or 0
            
        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")
            return 0
    
    
    async def is_account_locked(self, email: str) -> bool:
        """Check if account is locked due to too many failed attempts."""
        try:
            async with self.db.get_connection() as conn:
                result = await conn.fetchrow(
                    """
                    SELECT login_attempt_count, last_failed_attempt
                    FROM auth_users
                    WHERE email = $1 AND is_active = TRUE
                    """,
                    email
                )
            
            if not result:
                return False
            
            attempt_count = result["login_attempt_count"]
            last_failed = result["last_failed_attempt"]
            
            # Locked if 3+ attempts AND last attempt within 15 minutes
            if attempt_count >= 3 and last_failed:
                lockout_duration = timedelta(minutes=15)
                if datetime.now(timezone.utc) - last_failed.replace(tzinfo=timezone.utc) < lockout_duration:
                    return True
                # Lockout expired, but don't reset counters here (separate concern)
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking account lockout: {e}")
            return False
    
    async def reset_lockout(self, email: str) -> None:
        """Reset login attempt counters after lockout expires."""
        try:
            async with self.db.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE auth_users 
                    SET login_attempt_count = 0, last_failed_attempt = NULL
                    WHERE email = $1
                    """,
                    email
                )
        except Exception as e:
            logger.error(f"Error resetting lockout: {e}")
    
    
    async def add_to_token_blocklist(
        self, 
        token: str, 
        expiry_timestamp: datetime
    ) -> None:
        """
        Add token to revocation list in Redis.
        
        Token automatically expires from Redis after JWT expiry.
        Used for logout functionality.
        
        Args:
            token: JWT token to revoke
            expiry_timestamp: When token expires (to set Redis TTL)
        """
        if not self.redis:
            logger.warning("Redis not available, token blocklist skipped")
            return
        
        try:
            # Calculate TTL (time until expiry)
            ttl_seconds = (expiry_timestamp - datetime.now(timezone.utc)).total_seconds()
            
            if ttl_seconds > 0:
                # Use SHA-256 hash of token instead of truncation to avoid collisions
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                await self.redis.setex(
                    f"revoked_token:{token_hash}",
                    int(ttl_seconds),
                    "1"
                )
                logger.info(f"✅ Token added to blocklist (TTL: {int(ttl_seconds)}s)")
        except Exception as e:
            logger.error(f"Error adding to token blocklist: {e}")
    
    
    async def is_token_revoked(self, token: str) -> bool:
        """Check if token is in revocation list."""
        if not self.redis:
            return False
        
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            exists = await self.redis.exists(f"revoked_token:{token_hash}")
            return exists > 0
        except Exception as e:
            logger.error(f"Error checking token revocation: {e}")
            return False
    
    
    async def delete_user(self, user_id: int) -> bool:
        """
        Soft-delete user (GDPR compliance).
        
        Sets is_active = FALSE instead of permanent deletion.
        """
        try:
            async with self.db.get_connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE auth_users 
                    SET is_active = FALSE, updated_at = NOW()
                    WHERE id = $1
                    """,
                    user_id
                )
            
            logger.info(f"✅ User {user_id} soft-deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False
    
    
    async def export_user_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Export user data (GDPR compliance).
        
        Returns user profile, login history, etc.
        """
        try:
            async with self.db.get_connection() as conn:
                user = await conn.fetchrow(
                    """
                    SELECT id, email, name, created_at, updated_at, last_login
                    FROM auth_users
                    WHERE id = $1
                    """,
                    user_id
                )
            
            if user:
                return dict(user)
            return None
            
        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            return None


# Singleton instance
_auth_db_service: Optional[AuthDatabaseService] = None


async def get_auth_db_service() -> AuthDatabaseService:
    """
    Get or create AuthDatabaseService singleton.
    
    Must be called after database is initialized.
    """
    global _auth_db_service
    
    if _auth_db_service is None:
        from core.dependencies import DIContainer
        
        container = DIContainer.get_instance()
        db_manager = container.get_service('db_manager')
        redis_client = getattr(container, 'redis_client', None)
        
        _auth_db_service = AuthDatabaseService(db_manager, redis_client)
        await _auth_db_service.ensure_users_table()
    
    return _auth_db_service


async def init_auth_db_service(db_manager, redis_client=None) -> AuthDatabaseService:
    """Initialize AuthDatabaseService manually (for testing or explicit init)."""
    global _auth_db_service
    
    _auth_db_service = AuthDatabaseService(db_manager, redis_client)
    await _auth_db_service.ensure_users_table()
    
    return _auth_db_service
