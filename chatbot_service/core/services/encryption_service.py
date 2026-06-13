"""
Encryption service for PHI (Protected Health Information).
Uses AES-256-GCM encryption (authenticated encryption).

Phase 2: Encryption Service

Security: Cache is sized to prevent memory exhaustion from high-cardinality salts.
"""


from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import json
import logging
import base64
from typing import Union, Dict, Any

try:
    from cachetools import LRUCache
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Service for encrypting/decrypting PHI.
    Uses AES-256-GCM for authenticated encryption with integrity checking.
    """

    def __init__(self, master_key: str = None):
        """
        Initialize encryption service with master key.
        
        Args:
            master_key: Master encryption key (MUST be from secure config)
            
        Raises:
            ValueError: If master key is missing or uses default value
        """
        # Get key from parameter or environment
        key = master_key or os.getenv("ENCRYPTION_MASTER_KEY")
        
        # STRICT VALIDATION: Reject missing or default keys
        if not key:
            raise ValueError(
                "ENCRYPTION_MASTER_KEY environment variable is required. "
                "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        
        # STRICT VALIDATION: Reject known default keys
        FORBIDDEN_KEYS = [
            "default-dev-key-change-in-production",
            "changeme",
            "test",
            "dev",
            "localhost",
        ]
        
        if key.lower() in FORBIDDEN_KEYS:
            raise ValueError(
                f"Cannot use default/test master key: '{key}'. "
                "You MUST provide a cryptographically secure random key. "
                "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        
        # STRICT VALIDATION: Minimum key length
        if len(key) < 32:
            raise ValueError(
                f"Master key too short ({len(key)} characters). "
                "Minimum 32 characters required for security."
            )
        
        self.master_key = key
        self.algorithm = "AES-256-GCM"
        
        # Initialize cache with size limit to prevent memory exhaustion
        # Note: Encryption uses random salts each time, so cache hits only on decryption.
        # LRUCache with maxsize=1000 prevents unbounded growth.
        if HAS_CACHETOOLS:
            self._key_cache = LRUCache(maxsize=1000)  # Max 1000 cached keys (~32KB)
            logger.info(f"✅ Encryption service initialized with LRUCache (max 1000 keys, ~32KB)")
        else:
            # Fallback: regular dict (but warn about memory risk)
            self._key_cache = {}
            logger.warning(
                "⚠️  cachetools not installed - using unbounded dict cache for derived keys. "
                "Install cachetools to prevent memory leaks: pip install cachetools"
            )
        
        logger.info(f"Encryption service initialized with algorithm: {self.algorithm}")

    def _derive_key(self, salt: bytes) -> bytes:
        """
        Derive encryption key from master key using PBKDF2.
        Caches derived keys to avoid expensive 100K iteration PBKDF2 on every encrypt/decrypt.

        Args:
            salt: Random salt bytes

        Returns:
            32-byte key for AES-256
        """
        # Check cache first (O(1) lookup avoids 100ms PBKDF2 computation)
        if salt in self._key_cache:
            return self._key_cache[salt]

        # Expensive operation: PBKDF2 with 100K iterations (~100ms per call)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits for AES-256
            salt=salt,
            iterations=100000,  # NIST recommended minimum
            backend=None,  # Uses default backend
        )
        derived_key = kdf.derive(self.master_key.encode())

        # Cache the derived key (salt is random, so cache hit on decrypt)
        self._key_cache[salt] = derived_key

        return derived_key

    def encrypt(self, plaintext: Union[str, Dict[str, Any]]) -> str:
        """
        Encrypt data and return base64-encoded string.

        Format: base64(salt || IV || ciphertext)
        - salt: 16 bytes (used for key derivation)
        - IV: 12 bytes (nonce for GCM)
        - ciphertext: encrypted data with authentication tag

        Args:
            plaintext: String or dict to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        try:
            # Convert dict to JSON if needed
            if isinstance(plaintext, dict):
                # Use custom encoder for datetime serialization
                pass

                plaintext = json.dumps(plaintext, default=str)

            # Generate random salt and IV
            salt = os.urandom(16)
            iv = os.urandom(12)  # 96-bit IV for GCM

            # Derive key from master key and salt
            key = self._derive_key(salt)

            # Encrypt using AES-256-GCM
            cipher = AESGCM(key)
            ciphertext = cipher.encrypt(iv, plaintext.encode(), None)

            # Combine salt, IV, and ciphertext
            encrypted_data = salt + iv + ciphertext

            # Encode to base64 for storage
            encoded = base64.b64encode(encrypted_data).decode()

            logger.debug(f"Data encrypted successfully ({len(plaintext)} bytes)")
            return encoded

        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise

    def decrypt(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """
        Decrypt base64-encoded data.

        Args:
            encrypted_data: Base64-encoded encrypted string from encrypt()

        Returns:
            Decrypted string or dict

        Raises:
            Exception: If decryption fails (wrong key or corrupted data)
        """
        try:
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data)

            # Extract components
            salt = encrypted_bytes[:16]
            iv = encrypted_bytes[16:28]
            ciphertext = encrypted_bytes[28:]

            # Derive key from master key and salt
            key = self._derive_key(salt)

            # Decrypt using AES-256-GCM
            cipher = AESGCM(key)
            plaintext = cipher.decrypt(iv, ciphertext, None)

            # Decode to string
            plaintext_str = plaintext.decode()

            # Try to parse as JSON
            try:
                return json.loads(plaintext_str)
            except json.JSONDecodeError:
                return plaintext_str

        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise

    def encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a dict.

        Sensitive fields: patient_id, name, ssn, phone, email, medical_record_number, diagnosis

        Args:
            data: Dictionary with potentially sensitive fields

        Returns:
            Dictionary with sensitive fields encrypted
        """
        sensitive_fields = [
            "patient_id",
            "name",
            "ssn",
            "phone",
            "email",
            "medical_record_number",
            "diagnosis",
            "treatment",
        ]

        encrypted = {}
        for key, value in data.items():
            if key in sensitive_fields and value:
                try:
                    encrypted[key] = self.encrypt(str(value))
                except Exception as e:
                    logger.error(f"Failed to encrypt field {key}: {e}")
                    encrypted[key] = value
            else:
                encrypted[key] = value

        return encrypted

    def decrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt sensitive fields in a dict.

        Args:
            data: Dictionary with encrypted sensitive fields

        Returns:
            Dictionary with sensitive fields decrypted
        """
        sensitive_fields = [
            "patient_id",
            "name",
            "ssn",
            "phone",
            "email",
            "medical_record_number",
            "diagnosis",
            "treatment",
        ]

        decrypted = {}
        for key, value in data.items():
            if key in sensitive_fields and isinstance(value, str):
                try:
                    decrypted[key] = self.decrypt(value)
                except Exception as e:
                    logger.error(f"Failed to decrypt field {key}: {e}")
                    decrypted[key] = value
            else:
                decrypted[key] = value

        return decrypted


# Singleton instance
_encryption_service: Union[EncryptionService, None] = None


def get_encryption_service(master_key: str = None) -> EncryptionService:
    """
    Get or create encryption service singleton.
    
    Args:
        master_key: Optional master key (uses env var if not provided)
        
    Returns:
        EncryptionService instance
        
    Raises:
        ValueError: If master key is not configured
    """
    global _encryption_service
    if _encryption_service is None:
        # NO DEFAULT VALUE - will raise ValueError if missing
        _encryption_service = EncryptionService(master_key)
    return _encryption_service


def reset_encryption_service() -> None:
    """Reset encryption service singleton (useful for testing)."""
    global _encryption_service
    _encryption_service = None
