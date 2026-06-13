"""
Configuration Compatibility Layer

Provides backward compatibility for old config.py interface.
Maps old settings to new AppConfig.

This layer allows gradual migration of legacy code.
DEPRECATED: Direct imports from this module. Use core.config.app_config instead.
"""

from core.config.app_config import get_app_config
import os



class SettingsCompat:
    """Compatibility wrapper for old Settings class interface."""
    
    def __init__(self):
        self._config = get_app_config()
    
    # Service Configuration
    @property
    def SERVICE_NAME(self) -> str:
        return "HeartGuard NLP Service"
    
    @property
    def SERVICE_VERSION(self) -> str:
        return "1.0.0"
    
    @property
    def SERVICE_PORT(self) -> int:
        return self._config.api.port
    
    @property
    def SERVICE_HOST(self) -> str:
        return self._config.api.host
    
    # Database Configuration
    @property
    def DATABASE_URL(self) -> str:
        """Build database URL from config."""
        db = self._config.database
        if db.backend == "mysql":
            return f"mysql://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
        else:
            # Default to PostgreSQL
            return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
    
    @property
    def DB_POOL_MIN_SIZE(self) -> int:
        return int(os.environ.get("DB_POOL_MIN_SIZE", "5"))
    
    @property
    def DB_POOL_MAX_SIZE(self) -> int:
        return int(os.environ.get("DB_POOL_MAX_SIZE", "20"))
    
    @property
    def DB_POOL_RECYCLE(self) -> int:
        return int(os.environ.get("DB_POOL_RECYCLE", "3600"))
    
    @property
    def DB_READ_REPLICA_URL(self) -> str:
        return os.environ.get("DB_READ_REPLICA_URL", "")
    
    # Security Configuration
    @property
    def SECRET_KEY(self) -> str:
        return os.environ.get("SECRET_KEY", "default-secret-key")
    
    @property
    def ALGORITHM(self) -> str:
        return "HS256"
    
    @property
    def ACCESS_TOKEN_EXPIRE_MINUTES(self) -> int:
        return 30
    
    # LLM Configuration
    @property
    def OLLAMA_HOST(self) -> str:
        return self._config.llm.api_host
    
    @property
    def OLLAMA_MODEL(self) -> str:
        return self._config.llm.model_name
    
    @property
    def OLLAMA_TEMPERATURE(self) -> float:
        return self._config.llm.temperature
    
    @property
    def OLLAMA_TOP_P(self) -> float:
        return float(os.environ.get("OLLAMA_TOP_P", "0.95"))
    
    @property
    def OLLAMA_TOP_K(self) -> int:
        return int(os.environ.get("OLLAMA_TOP_K", "40"))
    
    @property
    def OLLAMA_MAX_TOKENS(self) -> int:
        return self._config.llm.max_tokens
    
    # Feature Flags
    @property
    def RAG_ENABLED(self) -> bool:
        return self._config.rag.enabled
    
    @property
    def MEMORY_ENABLED(self) -> bool:
        return True
    
    @property
    def AGENTS_ENABLED(self) -> bool:
        return True
    
    @property
    def TOOLS_ENABLED(self) -> bool:
        return True
    
    @property
    def STRUCTURED_OUTPUTS_ENABLED(self) -> bool:
        return True
    
    # API Configuration
    @property
    def CORS_ORIGINS(self) -> list:
        return ["*"]
    
    @property
    def LOG_LEVEL(self) -> str:
        return os.environ.get("LOG_LEVEL", "INFO")


# Global instance
_compat_settings = None


def get_settings() -> SettingsCompat:
    """Get compatibility settings wrapper."""
    global _compat_settings
    if _compat_settings is None:
        _compat_settings = SettingsCompat()
    return _compat_settings


def reset_settings():
    """Reset settings (for testing)."""
    global _compat_settings
    _compat_settings = None
