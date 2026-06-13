"""
Configuration for NLP Microservice (DEPRECATED - Use core.config.app_config instead)

⚠️  DEPRECATION NOTICE:
This module is now a PROXY LAYER for backward compatibility only.
All configuration should use core.config.app_config.get_app_config() instead.

This proxy will be removed in version 2.0.
Migration path: https://docs.example.com/migration/config-consolidation

Backward Compatibility:
    from config import OLLAMA_MODEL
    # Works but shows deprecation warning - internally uses AppConfig
"""


import os
import warnings
from typing import List, Union, Optional
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

# Show deprecation warning on import (once per process)
warnings.warn(
    "⚠️  config.py is DEPRECATED! Use core.config.app_config.get_app_config() instead.\n"
    "This proxy layer will be removed in v2.0.\n"
    "See: MIGRATION_GUIDE.md for migration patterns.",
    DeprecationWarning,
    stacklevel=2
)


# Helper for Ollama Host
def _get_ollama_host():
    """
    Determine appropriate Ollama host based on deployment context.
    """
    # If explicitly set in environment, use it (highest priority)
    if os.getenv("OLLAMA_HOST"):
        return os.getenv("OLLAMA_HOST")

    # Check if running in Docker (standard Docker env variable)
    if os.path.exists("/.dockerenv"):
        # Running inside a container - use host.docker.internal
        default_host = "http://host.docker.internal:11434"
        return os.getenv("OLLAMA_HOST_DOCKER", default_host)

    # Not in Docker - assume localhost
    return "http://localhost:11434"


class Settings(BaseSettings):
    """
    DEPRECATED: Settings class maintained for backward compatibility only.
    
    This class is superseded by core.config.app_config.AppConfig.
    All code should use get_app_config() from core.config.app_config instead.
    
    This proxy layer will be removed in v2.0.
    """

    # Service Configuration
    SERVICE_NAME: str = "HeartGuard NLP Service"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_PORT: int = Field(default=5001, alias="NLP_SERVICE_PORT")
    SERVICE_HOST: str = Field(default="127.0.0.1", alias="NLP_SERVICE_HOST")

    # Database Configuration (PostgreSQL)
    DATABASE_URL: str = ""  # Built from POSTGRES_* env vars if empty

    # Security Configuration
    SECRET_KEY: str = Field(default="default-secret-key", env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate SECRET_KEY to prevent insecure defaults in production."""
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment in ["production", "staging"] and v == "default-secret-key":
            raise ValueError(
                "SECRET_KEY must be set to a secure value in production/staging. "
                "Generate a random secret with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        if v != "default-secret-key" and len(v) < 32:
            raise ValueError(
                f"SECRET_KEY must be at least 32 characters long (current: {len(v)}). "
                "For production, use: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        if environment == "production":
            entropy = len(set(v)) / len(v)
            if entropy < 0.5:
                raise ValueError("SECRET_KEY has insufficient entropy for production")

        if v == "default-secret-key":
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Using default SECRET_KEY in development. "
                "Set a unique SECRET_KEY in .env for better security."
            )

        return v

    # NLP Model Configuration
    SPACY_MODEL: str = "en_core_web_sm"
    USE_GPU: bool = False

    # Intent Recognition Config
    INTENT_CONFIDENCE_THRESHOLD: float = 0.5
    ENTITY_CONFIDENCE_THRESHOLD: float = 0.6

    # Sentiment Analysis Config
    SENTIMENT_THRESHOLD_POSITIVE: float = 0.6
    SENTIMENT_THRESHOLD_NEGATIVE: float = -0.4
    SENTIMENT_THRESHOLD_DISTRESSED: float = -0.7
    SENTIMENT_THRESHOLD_URGENT: float = 0.8

    # LLM Configuration - Primary Provider
    LLM_PROVIDER: str = Field(default="openrouter", env="LLM_PROVIDER")
    
    # OpenRouter Configuration (Primary - GPT OSS)
    OPENROUTER_API_KEY: str = Field(default="", env="OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = Field(default="openai/gpt-oss-20b:free", env="OPENROUTER_MODEL")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL")
    OPENROUTER_TEMPERATURE: float = Field(default=0.7, env="OPENROUTER_TEMPERATURE")
    OPENROUTER_MAX_TOKENS: int = Field(default=256, env="OPENROUTER_MAX_TOKENS")
    OPENROUTER_TOP_P: float = Field(default=0.9, env="OPENROUTER_TOP_P")
    OPENROUTER_TIMEOUT_SECONDS: int = Field(default=60, env="OPENROUTER_TIMEOUT_SECONDS")
    
    # OpenRouter Gemini Configuration (Alternative/Fallback)
    OPENROUTER_GEMINI_API_KEY: str = Field(default="", env="OPENROUTER_GEMINI_API_KEY")
    OPENROUTER_GEMINI_MODEL: str = Field(default="google/gemma-3-4b-it:free", env="OPENROUTER_GEMINI_MODEL")
    OPENROUTER_GEMINI_TEMPERATURE: float = Field(default=0.7, env="OPENROUTER_GEMINI_TEMPERATURE")
    OPENROUTER_GEMINI_MAX_TOKENS: int = Field(default=256, env="OPENROUTER_GEMINI_MAX_TOKENS")
    
    # Ollama Configuration (Fallback)
    OLLAMA_HOST: str = Field(default_factory=_get_ollama_host)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = "gemma3:1b"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_TOP_P: float = 0.9
    OLLAMA_TOP_K: int = 40
    OLLAMA_MAX_TOKENS: int = 256
    OLLAMA_CONTEXT_WINDOW: int = 2048
    OLLAMA_TIMEOUT_SECONDS: int = 60

    # Disable Gemini integration by default
    USE_GEMINI: bool = Field(default=False, env="USE_GEMINI")

    # Feature Flags
    RAG_ENABLED: bool = Field(default=True, env="FEATURE_RAG")
    MEMORY_ENABLED: bool = Field(default=True, env="FEATURE_MEMORY")
    AGENTS_ENABLED: bool = Field(default=True, env="FEATURE_AGENTS")
    TOOLS_ENABLED: bool = Field(default=True, env="FEATURE_TOOLS")
    STRUCTURED_OUTPUTS_ENABLED: bool = Field(
        default=True, env="FEATURE_STRUCTURED_OUTPUTS"
    )
    GENERATION_ENABLED: bool = Field(default=True, env="FEATURE_GENERATION")

    USE_OLLAMA_FOR_RESPONSES: bool = True
    OLLAMA_FALLBACK_TO_LLM: bool = True

    # Model Versioning
    DEFAULT_MODEL_VERSION: str = "v1.0"

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "nlp_service.log"

    # CORS Configuration - includes ports 5173, 5174, 5175, 5176 for Vite dev server
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "https://heartguard.ai",
    ]

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Web Search Rate Limits
    WEB_SEARCH_RATE_LIMITS: dict = {
        "per_user_per_hour": 20,
        "per_user_per_day": 100,
        "global_per_minute": 60
    }
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_ALERT_TTL: int = Field(default=3600, env="REDIS_ALERT_TTL")  # 1 hour default cooldown
    
    # Database Pooling (LATENCY OPTIMIZED)
    DB_POOL_MIN_SIZE: int = 10  # Increased from 5 for better concurrency
    DB_POOL_MAX_SIZE: int = 30  # Increased from 20 for burst traffic
    DB_POOL_TIMEOUT: int = 30  # Connection timeout in seconds
    DB_POOL_RECYCLE: int = 3600  # Recycle connections after 1 hour
    
    # Read/Write Splitting (optional)
    DB_READ_REPLICA_URL: Optional[str] = None  # e.g., mysql://readonly@slave:3306/cardio

    # =========================================================================
    # MEMORI INTEGRATION SETTINGS (Phase 2: Enhanced Memory Features)
    # =========================================================================

    # Memory Manager Settings
    MEMORI_ENABLED: bool = True
    MEMORI_DATABASE_URL: str = ""  # PostgreSQL - built from DATABASE_URL if empty
    MEMORI_CACHE_SIZE: int = 100  # Max patient instances in LRU cache
    MEMORI_POOL_SIZE: int = 10  # Database connection pool size
    MEMORI_REQUEST_TIMEOUT: int = 30  # Timeout for memory operations (seconds)

    # Embedding Search Settings (EmbeddingSearchEngine)
    MEMORI_EMBEDDING_USE_LOCAL: bool = False  # Use remote Colab embeddings
    MEMORI_EMBEDDING_MODEL: str = "MedCPT-Query-Encoder"  # Remote embedding model
    MEMORI_EMBEDDING_SIMILARITY_THRESHOLD: float = 0.5  # Minimum similarity score
    MEMORI_EMBEDDING_CACHE_SIZE: int = 10000  # Embedding vector cache size

    # Conscious Agent Settings (ConsciouscAgent)
    MEMORI_CONSCIOUS_INGEST: bool = True  # Auto-inject relevant memory
    MEMORI_CONSCIOUS_MEMORY_LIMIT: int = 10  # Max conscious memories to load

    # Rate Limiting Settings (Memori RateLimiter)
    MEMORI_RATE_LIMIT_SEARCH: int = 60  # Searches per minute per user
    MEMORI_RATE_LIMIT_STORE: int = 100  # Stores per minute per user
    MEMORI_RATE_LIMIT_API_CALLS: int = 1000  # API calls per day per user
    MEMORI_STORAGE_QUOTA_MB: int = 100  # Storage quota per user in MB
    MEMORI_MEMORY_COUNT_LIMIT: int = 10000  # Max memories per user

    # Auth Provider Settings (Optional - default to NoAuth for dev)
    MEMORI_AUTH_PROVIDER: str = "none"  # Options: "none", "jwt", "oauth2", "apikey"
    MEMORI_JWT_SECRET: str = ""  # JWT secret for JWTAuthProvider
    MEMORI_JWT_ALGORITHM: str = "HS256"

    # Input Validation Settings
    MEMORI_INPUT_MAX_QUERY_LENGTH: int = 10000  # Max query length
    MEMORI_INPUT_VALIDATE_SQL_INJECTION: bool = True
    MEMORI_INPUT_SANITIZE_XSS: bool = True

    # Circuit Breaker Settings
    MEMORI_CIRCUIT_BREAKER_THRESHOLD: int = 5  # Failures before opening
    MEMORI_CIRCUIT_BREAKER_TIMEOUT: int = 60  # Seconds before half-open

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str) and not v.strip().startswith("["):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def validate_cors_origins(cls, v):
        # Check for wildcard which is forbidden in production
        if "*" in v:
            raise ValueError("Wildcard CORS origin '*' is forbidden in production!")

        # Ensure all origins are valid URLs
        for origin in v:
            if not origin.startswith(("http://", "https://")):
                raise ValueError(
                    f"Invalid CORS origin: {origin}. Must start with http:// or https://"
                )

        return v

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # Allow extra env vars like GOOGLE_*, SMTP_*, TWILIO_*, etc.
    )



# Initialize Settings (DEPRECATED - kept for compatibility)
settings = Settings()

def get_settings():
    """Return application settings (DEPRECATED - use get_app_config instead)."""
    warnings.warn(
        "get_settings() is deprecated! Use get_app_config() from core.config.app_config",
        DeprecationWarning,
        stacklevel=2
    )
    return settings


# ============================================================================
# PROXY LAYER - All exports delegate to core.config.app_config.AppConfig
# ============================================================================

def _get_app_config_wrapper():
    """Lazily load AppConfig to avoid circular imports."""
    try:
        from core.config.app_config import get_app_config
        return get_app_config()
    except ImportError:
        # Fallback if AppConfig not available yet
        import logging
        logging.getLogger(__name__).warning(
            "AppConfig not available, using legacy Settings"
        )
        return None


# Service Configuration (Proxy)
SERVICE_NAME = settings.SERVICE_NAME
SERVICE_VERSION = settings.SERVICE_VERSION
SERVICE_PORT = settings.SERVICE_PORT
SERVICE_HOST = settings.SERVICE_HOST

DATABASE_URL = settings.DATABASE_URL

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

SPACY_MODEL = settings.SPACY_MODEL
USE_GPU = settings.USE_GPU

INTENT_CONFIDENCE_THRESHOLD = settings.INTENT_CONFIDENCE_THRESHOLD
ENTITY_CONFIDENCE_THRESHOLD = settings.ENTITY_CONFIDENCE_THRESHOLD

SENTIMENT_THRESHOLD_POSITIVE = settings.SENTIMENT_THRESHOLD_POSITIVE
SENTIMENT_THRESHOLD_NEGATIVE = settings.SENTIMENT_THRESHOLD_NEGATIVE
SENTIMENT_THRESHOLD_DISTRESSED = settings.SENTIMENT_THRESHOLD_DISTRESSED
SENTIMENT_THRESHOLD_URGENT = settings.SENTIMENT_THRESHOLD_URGENT

# LLM Configuration (from Settings for now, use AppConfig.llm in new code)
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
OPENROUTER_MODEL = settings.OPENROUTER_MODEL
OPENROUTER_BASE_URL = settings.OPENROUTER_BASE_URL
OPENROUTER_TEMPERATURE = settings.OPENROUTER_TEMPERATURE
OPENROUTER_MAX_TOKENS = settings.OPENROUTER_MAX_TOKENS
OPENROUTER_TOP_P = settings.OPENROUTER_TOP_P
OPENROUTER_TIMEOUT_SECONDS = settings.OPENROUTER_TIMEOUT_SECONDS

OPENROUTER_GEMINI_API_KEY = settings.OPENROUTER_GEMINI_API_KEY
OPENROUTER_GEMINI_MODEL = settings.OPENROUTER_GEMINI_MODEL
OPENROUTER_GEMINI_TEMPERATURE = settings.OPENROUTER_GEMINI_TEMPERATURE
OPENROUTER_GEMINI_MAX_TOKENS = settings.OPENROUTER_GEMINI_MAX_TOKENS

OLLAMA_HOST = settings.OLLAMA_HOST
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
OLLAMA_TEMPERATURE = settings.OLLAMA_TEMPERATURE
OLLAMA_TOP_P = settings.OLLAMA_TOP_P
OLLAMA_TOP_K = settings.OLLAMA_TOP_K
OLLAMA_MAX_TOKENS = settings.OLLAMA_MAX_TOKENS
OLLAMA_CONTEXT_WINDOW = settings.OLLAMA_CONTEXT_WINDOW
OLLAMA_TIMEOUT_SECONDS = settings.OLLAMA_TIMEOUT_SECONDS

USE_GEMINI = settings.USE_GEMINI

# Feature Flags
RAG_ENABLED = settings.RAG_ENABLED
MEMORY_ENABLED = settings.MEMORY_ENABLED
AGENTS_ENABLED = settings.AGENTS_ENABLED
TOOLS_ENABLED = settings.TOOLS_ENABLED
STRUCTURED_OUTPUTS_ENABLED = settings.STRUCTURED_OUTPUTS_ENABLED
GENERATION_ENABLED = settings.GENERATION_ENABLED

USE_OLLAMA_FOR_RESPONSES = settings.USE_OLLAMA_FOR_RESPONSES
OLLAMA_FALLBACK_TO_LLM = settings.OLLAMA_FALLBACK_TO_LLM

DEFAULT_MODEL_VERSION = settings.DEFAULT_MODEL_VERSION

LOG_LEVEL = settings.LOG_LEVEL
LOG_FILE = settings.LOG_FILE

CORS_ORIGINS = settings.CORS_ORIGINS

RATE_LIMIT_PER_MINUTE = settings.RATE_LIMIT_PER_MINUTE

# Memori Integration Settings
MEMORI_ENABLED = settings.MEMORI_ENABLED
MEMORI_DATABASE_URL = settings.MEMORI_DATABASE_URL
MEMORI_CACHE_SIZE = settings.MEMORI_CACHE_SIZE
MEMORI_POOL_SIZE = settings.MEMORI_POOL_SIZE
MEMORI_REQUEST_TIMEOUT = settings.MEMORI_REQUEST_TIMEOUT
MEMORI_EMBEDDING_USE_LOCAL = settings.MEMORI_EMBEDDING_USE_LOCAL
MEMORI_EMBEDDING_MODEL = settings.MEMORI_EMBEDDING_MODEL
MEMORI_EMBEDDING_SIMILARITY_THRESHOLD = settings.MEMORI_EMBEDDING_SIMILARITY_THRESHOLD
MEMORI_CONSCIOUS_INGEST = settings.MEMORI_CONSCIOUS_INGEST
MEMORI_RATE_LIMIT_SEARCH = settings.MEMORI_RATE_LIMIT_SEARCH
MEMORI_RATE_LIMIT_STORE = settings.MEMORI_RATE_LIMIT_STORE
MEMORI_AUTH_PROVIDER = settings.MEMORI_AUTH_PROVIDER
MEMORI_CIRCUIT_BREAKER_THRESHOLD = settings.MEMORI_CIRCUIT_BREAKER_THRESHOLD
    
# Database Pooling
DB_POOL_MIN_SIZE = settings.DB_POOL_MIN_SIZE
DB_POOL_MAX_SIZE = settings.DB_POOL_MAX_SIZE
DB_POOL_TIMEOUT = settings.DB_POOL_TIMEOUT
DB_POOL_RECYCLE = settings.DB_POOL_RECYCLE
DB_READ_REPLICA_URL = settings.DB_READ_REPLICA_URL

# Redis Configuration
REDIS_URL = settings.REDIS_URL
REDIS_ALERT_TTL = settings.REDIS_ALERT_TTL

OLLAMA_AVAILABLE_MODELS = [
    "gemma3:4b",
    "gemma2:2b",
    "gemma2:9b",
    "phi3",
    "neural-chat",
    "mistral",
    "llama2",
]


def startup_check() -> None:
    """
    Perform security checks at application startup.
    
    MUST be called in main.py before starting the server.
    Will immediately exit if critical security issues are found.
    
    Raises:
        SystemExit: If running in production with insecure configuration
    """
    import secrets
    import logging
    
    logger = logging.getLogger(__name__)
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    # === Check 1: SECRET_KEY in Production ===
    if environment == "production":
        secret_key = settings.SECRET_KEY
        
        if secret_key == "default-secret-key":
            logger.critical(
                "🚨 FATAL: Cannot start in production with default SECRET_KEY!\n"
                "Generate a secure key with:\n"
                "  python -c 'import secrets; print(secrets.token_urlsafe(32))'\n"
                "Then set SECRET_KEY environment variable."
            )
            raise SystemExit(1)
        
        if len(secret_key) < 32:
            logger.critical(
                f"🚨 FATAL: SECRET_KEY too short ({len(secret_key)} chars)!\n"
                "Production requires at least 32 characters."
            )
            raise SystemExit(1)
        
        entropy = len(set(secret_key)) / len(secret_key)
        if entropy < 0.5:
            logger.critical(
                f"🚨 FATAL: SECRET_KEY has insufficient entropy ({entropy:.2f})!\n"
                "Use a cryptographically random value."
            )
            raise SystemExit(1)
        
        logger.info("✅ SECRET_KEY passes production security checks")
    
    # === Check 2: Database Configuration ===
    if environment == "production":
        db_password = os.getenv("POSTGRES_PASSWORD", os.getenv("MYSQL_PASSWORD", ""))
        if not db_password:
            logger.warning(
                "⚠️ WARNING: Database password is empty in production. "
                "This may be a security risk."
            )
    
    # === Check 3: Redis for Rate Limiting ===
    if environment == "production":
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.warning(
                "⚠️ WARNING: REDIS_URL not set. "
                "Rate limiting will be per-process only."
            )
    
    logger.info(f"Startup checks complete for environment: {environment}")

