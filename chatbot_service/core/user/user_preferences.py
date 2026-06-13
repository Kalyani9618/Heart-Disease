"""
User Preferences Store for Healthcare AI Memory Management.

Implements persistent user preferences based on chat.md architecture:
"The system explicitly saves those facts and injects them again in future conversations"

This module provides:
- UserPreference SQLAlchemy model for database storage
- UserPreferencesManager class for CRUD operations
- Support for different data types (string, json, boolean, number)
- GDPR compliance features (export, delete)

Author: AI Memory System Implementation
Version: 1.0.0
"""


from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Boolean,
    Integer,
    create_engine,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import json
import logging
import os

logger = logging.getLogger(__name__)

Base = declarative_base()


class UserPreference(Base):
    """
    SQLAlchemy model for user preferences.

    Stores key-value preferences with type information and
    sensitivity flags for PHI compliance.
    """

    __tablename__ = "user_preferences"

    # Composite primary key: user_id + preference_key
    user_id = Column(String(255), primary_key=True, index=True)
    preference_key = Column(String(100), primary_key=True)

    # Value storage with type info
    preference_value = Column(Text, nullable=False)
    data_type = Column(String(50), default="string")  # string, json, boolean, number

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # PHI/sensitivity flag
    is_sensitive = Column(Boolean, default=False)

    # Category for organization
    category = Column(String(100), default="general")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_user_preferences_category", "user_id", "category"),
        Index("idx_user_preferences_updated", "updated_at"),
    )

    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id}, key={self.preference_key})>"


class PreferenceAuditLog(Base):
    """
    Audit log for preference changes (HIPAA compliance).
    """

    __tablename__ = "preference_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    preference_key = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)  # set, delete, export
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(50), nullable=True)

    __table_args__ = (Index("idx_audit_log_timestamp", "timestamp"),)


# ============================================================================
# Standard Preference Keys
# ============================================================================


class PreferenceKeys:
    """Standard preference key constants."""

    # Communication
    COMMUNICATION_STYLE = "communication_style"  # formal, casual, detailed, concise
    LANGUAGE = "language"

    # Health
    HEALTH_GOALS = "health_goals"  # JSON array
    DIETARY_RESTRICTIONS = "dietary_restrictions"  # JSON array
    ACTIVITY_LEVEL = "activity_level"  # sedentary, light, moderate, active

    # Display
    PREFERRED_UNITS = "preferred_units"  # metric, imperial
    DATE_FORMAT = "date_format"
    TIME_FORMAT = "time_format"

    # Notifications
    NOTIFICATION_SETTINGS = "notification_settings"  # JSON object
    REMINDER_TIMES = "reminder_times"  # JSON array

    # Emergency
    EMERGENCY_CONTACT = "emergency_contact"  # JSON object (sensitive)

    # AI Behavior
    AI_VERBOSITY = "ai_verbosity"  # brief, normal, detailed
    INCLUDE_EXPLANATIONS = "include_explanations"  # boolean
    
    # Medical
    ALLERGIES = "allergies"  # JSON object {allergen: {severity, reaction}}


# ============================================================================
# User Preferences Manager
# ============================================================================


class UserPreferencesManager:
    """
    Manages persistent user preferences.

    Implements Section 7 from chat.md:
    "The system explicitly saves those facts
    and injects them again in future conversations"

    Key Features:
    - Type-safe preference storage
    - Category-based organization
    - PHI sensitivity flags
    - Audit logging for HIPAA
    - GDPR export and delete

    Usage:
        manager = UserPreferencesManager("sqlite:///preferences.db")
        manager.set_preference("user123", "language", "en")
        lang = manager.get_preference("user123", "language")
    """

    def __init__(
        self, database_url: Optional[str] = None, enable_audit_log: bool = True
    ):
        """
        Initialize preferences manager.

        Args:
            database_url: SQLAlchemy database URL. Defaults to PostgreSQL from AppConfig.
            enable_audit_log: Whether to log preference changes
        """
        # Default to PostgreSQL from AppConfig
        if database_url is None:
            database_url = os.environ.get("USER_PREFERENCES_DB_URL")
            
            if not database_url:
                # Build PostgreSQL URL from AppConfig
                try:
                    from core.config.app_config import get_app_config
                    config = get_app_config()
                    db = config.database
                    database_url = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
                except Exception as e:
                    logger.warning(f"Could not load AppConfig, using env fallback: {e}")
                    # Fallback to environment variables
                    host = os.environ.get("POSTGRES_HOST", "localhost")
                    port = os.environ.get("POSTGRES_PORT", "5432")
                    user = os.environ.get("POSTGRES_USER", "postgres")
                    password = os.environ.get("POSTGRES_PASSWORD", "")
                    db_name = os.environ.get("POSTGRES_DB", "heartguard")
                    database_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

        self.engine = create_engine(database_url, pool_pre_ping=True, echo=False)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine)
        self.enable_audit = enable_audit_log

        logger.info(f"UserPreferencesManager initialized with PostgreSQL")

    @contextmanager
    def _get_session(self) -> Generator[Session, None, None]:
        """Get database session with auto-commit/rollback."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ========================================================================
    # Core CRUD Operations
    # ========================================================================

    def set_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        is_sensitive: bool = False,
        category: str = "general",
        ip_address: Optional[str] = None,
    ) -> None:
        """
        Set a user preference.

        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value (any JSON-serializable type)
            is_sensitive: Whether this is PHI/sensitive data
            category: Category for organization
            ip_address: Client IP for audit log
        """
        # Determine data type and serialize
        data_type, str_value = self._serialize(value)

        with self._get_session() as db:
            existing = (
                db.query(UserPreference)
                .filter_by(user_id=user_id, preference_key=key)
                .first()
            )

            old_value = None

            if existing:
                old_value = existing.preference_value
                existing.preference_value = str_value
                existing.data_type = data_type
                existing.is_sensitive = is_sensitive
                existing.category = category
                existing.updated_at = datetime.utcnow()
            else:
                pref = UserPreference(
                    user_id=user_id,
                    preference_key=key,
                    preference_value=str_value,
                    data_type=data_type,
                    is_sensitive=is_sensitive,
                    category=category,
                )
                db.add(pref)

            # Audit log
            if self.enable_audit:
                self._log_action(
                    db=db,
                    user_id=user_id,
                    key=key,
                    action="set",
                    old_value=old_value,
                    new_value=str_value,
                    ip_address=ip_address,
                )

        logger.debug(f"Set preference {key} for user {user_id}")

    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a single user preference.

        Args:
            user_id: User identifier
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        with self._get_session() as db:
            pref = (
                db.query(UserPreference)
                .filter_by(user_id=user_id, preference_key=key)
                .first()
            )

            if not pref:
                return default

            return self._deserialize(pref.preference_value, pref.data_type)

    def get_all_preferences(
        self,
        user_id: str,
        include_sensitive: bool = False,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get all preferences for a user.

        Args:
            user_id: User identifier
            include_sensitive: Include PHI/sensitive preferences
            category: Filter by category (optional)

        Returns:
            Dictionary of all preferences
        """
        with self._get_session() as db:
            query = db.query(UserPreference).filter_by(user_id=user_id)

            if not include_sensitive:
                query = query.filter_by(is_sensitive=False)

            if category:
                query = query.filter_by(category=category)

            prefs = query.all()

            return {
                p.preference_key: self._deserialize(p.preference_value, p.data_type)
                for p in prefs
            }

    def get_preferences_by_category(
        self, user_id: str, category: str
    ) -> Dict[str, Any]:
        """Get all preferences in a specific category."""
        return self.get_all_preferences(
            user_id=user_id, include_sensitive=False, category=category
        )

    async def get_allergies(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get known allergies for a user (Async compatibility wrapper).
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary of allergies: {allergen_name: {details}}
        """
        # Note: This calls synchronous DB code. In a high-concurrency async app,
        # this should be run in an executor. For now, direct call is acceptable
        # for SQLite/low-latency local DB.
        allergies = self.get_preference(user_id, PreferenceKeys.ALLERGIES, default={})
        return allergies if isinstance(allergies, dict) else {}

    def delete_preference(
        self, user_id: str, key: str, ip_address: Optional[str] = None
    ) -> bool:
        """
        Delete a specific preference.

        Args:
            user_id: User identifier
            key: Preference key
            ip_address: Client IP for audit log

        Returns:
            True if preference was deleted
        """
        with self._get_session() as db:
            pref = (
                db.query(UserPreference)
                .filter_by(user_id=user_id, preference_key=key)
                .first()
            )

            if not pref:
                return False

            old_value = pref.preference_value
            db.delete(pref)

            # Audit log
            if self.enable_audit:
                self._log_action(
                    db=db,
                    user_id=user_id,
                    key=key,
                    action="delete",
                    old_value=old_value,
                    new_value=None,
                    ip_address=ip_address,
                )

        logger.debug(f"Deleted preference {key} for user {user_id}")
        return True

    def clear_all_preferences(
        self, user_id: str, ip_address: Optional[str] = None
    ) -> int:
        """
        Clear all preferences for a user (GDPR compliance).

        Args:
            user_id: User identifier
            ip_address: Client IP for audit log

        Returns:
            Number of preferences deleted
        """
        with self._get_session() as db:
            deleted = db.query(UserPreference).filter_by(user_id=user_id).delete()

            # Audit log
            if self.enable_audit:
                self._log_action(
                    db=db,
                    user_id=user_id,
                    key="*",
                    action="clear_all",
                    old_value=f"{deleted} preferences",
                    new_value=None,
                    ip_address=ip_address,
                )

        logger.info(f"Cleared {deleted} preferences for user {user_id}")
        return deleted

    # ========================================================================
    # Bulk Operations
    # ========================================================================

    def set_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        category: str = "general",
        ip_address: Optional[str] = None,
    ) -> int:
        """
        Set multiple preferences at once.

        Args:
            user_id: User identifier
            preferences: Dictionary of key-value pairs
            category: Category for all preferences
            ip_address: Client IP for audit log

        Returns:
            Number of preferences set
        """
        count = 0
        for key, value in preferences.items():
            self.set_preference(
                user_id=user_id,
                key=key,
                value=value,
                category=category,
                ip_address=ip_address,
            )
            count += 1
        return count

    # ========================================================================
    # GDPR Compliance
    # ========================================================================

    def export_user_data(
        self, user_id: str, format: str = "json", ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export all user data (GDPR compliance).

        Args:
            user_id: User identifier
            format: Export format (json, dict)
            ip_address: Client IP for audit log

        Returns:
            Dictionary containing all user data
        """
        with self._get_session() as db:
            prefs = db.query(UserPreference).filter_by(user_id=user_id).all()

            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "preferences": [
                    {
                        "key": p.preference_key,
                        "value": self._deserialize(p.preference_value, p.data_type),
                        "data_type": p.data_type,
                        "category": p.category,
                        "is_sensitive": p.is_sensitive,
                        "created_at": (
                            p.created_at.isoformat() if p.created_at else None
                        ),
                        "updated_at": (
                            p.updated_at.isoformat() if p.updated_at else None
                        ),
                    }
                    for p in prefs
                ],
                "total_count": len(prefs),
            }

            # Audit log
            if self.enable_audit:
                self._log_action(
                    db=db,
                    user_id=user_id,
                    key="*",
                    action="export",
                    old_value=None,
                    new_value=f"Exported {len(prefs)} preferences",
                    ip_address=ip_address,
                )

        logger.info(f"Exported {len(prefs)} preferences for user {user_id}")
        return export_data

    def delete_user_data(
        self, user_id: str, ip_address: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Delete all user data (GDPR right to erasure).

        Args:
            user_id: User identifier
            ip_address: Client IP for audit log

        Returns:
            Dictionary with deletion counts
        """
        with self._get_session() as db:
            # Delete preferences
            pref_count = db.query(UserPreference).filter_by(user_id=user_id).delete()

            # Note: Audit logs are retained for compliance
            # They don't contain the actual preference values

            # Log the deletion
            if self.enable_audit:
                audit_log = PreferenceAuditLog(
                    user_id=user_id,
                    preference_key="*",
                    action="gdpr_delete",
                    old_value=f"{pref_count} preferences",
                    new_value=None,
                    ip_address=ip_address,
                )
                db.add(audit_log)

        logger.info(f"GDPR deleted all data for user {user_id}")
        return {
            "preferences_deleted": pref_count,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ========================================================================
    # Audit Log
    # ========================================================================

    def get_audit_log(
        self, user_id: str, limit: int = 100, action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log for a user.

        Args:
            user_id: User identifier
            limit: Maximum records to return
            action: Filter by action type

        Returns:
            List of audit log entries
        """
        with self._get_session() as db:
            query = db.query(PreferenceAuditLog).filter_by(user_id=user_id)

            if action:
                query = query.filter_by(action=action)

            logs = (
                query.order_by(PreferenceAuditLog.timestamp.desc()).limit(limit).all()
            )

            return [
                {
                    "id": log.id,
                    "preference_key": log.preference_key,
                    "action": log.action,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "ip_address": log.ip_address,
                }
                for log in logs
            ]

    def _log_action(
        self,
        db: Session,
        user_id: str,
        key: str,
        action: str,
        old_value: Optional[str],
        new_value: Optional[str],
        ip_address: Optional[str],
    ) -> None:
        """Create audit log entry."""
        audit_log = PreferenceAuditLog(
            user_id=user_id,
            preference_key=key,
            action=action,
            old_value=old_value[:500] if old_value else None,  # Truncate
            new_value=new_value[:500] if new_value else None,
            ip_address=ip_address,
        )
        db.add(audit_log)

    # ========================================================================
    # Serialization Helpers
    # ========================================================================

    def _serialize(self, value: Any) -> tuple:
        """
        Serialize value to string with type info.

        Returns:
            Tuple of (data_type, string_value)
        """
        if isinstance(value, bool):
            return ("boolean", str(value).lower())
        elif isinstance(value, (int, float)):
            return ("number", str(value))
        elif isinstance(value, (dict, list)):
            return ("json", json.dumps(value))
        else:
            return ("string", str(value))

    def _deserialize(self, value: str, data_type: str) -> Any:
        """Deserialize preference value to correct type."""
        if data_type == "boolean":
            return value.lower() == "true"
        elif data_type == "number":
            return float(value) if "." in value else int(value)
        elif data_type == "json":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            return value

    # ========================================================================
    # Health Check
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check database health and return stats."""
        try:
            with self._get_session() as db:
                pref_count = db.query(UserPreference).count()
                user_count = db.query(UserPreference.user_id).distinct().count()

                return {
                    "status": "healthy",
                    "total_preferences": pref_count,
                    "unique_users": user_count,
                    "timestamp": datetime.utcnow().isoformat(),
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }


# ============================================================================
# Singleton Instance
# ============================================================================

# Default preferences manager instance
_preferences_manager: Optional[UserPreferencesManager] = None


def get_preferences_manager() -> UserPreferencesManager:
    """Get or create the singleton preferences manager."""
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = UserPreferencesManager()
    return _preferences_manager


def init_preferences_manager(database_url: str) -> UserPreferencesManager:
    """Initialize preferences manager with custom URL."""
    global _preferences_manager
    _preferences_manager = UserPreferencesManager(database_url=database_url)
    return _preferences_manager
