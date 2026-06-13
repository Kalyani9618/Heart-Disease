"""
Services module exports.

Provides unified access to all service layer functionality:
- Encryption service (AES-256-GCM)
- Health service (database CRUD)
"""

from core.services.encryption_service import (
    EncryptionService,
    get_encryption_service,
    reset_encryption_service,
)

from core.services.health_service import (
    HealthService,
    HealthRecordDB,
    get_health_service,
    reset_health_service,
)

__all__ = [
    # Encryption
    "EncryptionService",
    "get_encryption_service",
    "reset_encryption_service",
    # Health Service
    "HealthService",
    "HealthRecordDB",
    "get_health_service",
    "reset_health_service",
]
