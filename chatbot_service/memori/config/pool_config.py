"""Database connection pool configuration for SQLAlchemy"""

# Default connection pool settings (accessed as attributes)
# These can be overridden when creating a DatabaseManager instance

DEFAULT_POOL_SIZE = 5            # Number of connections to maintain
DEFAULT_MAX_OVERFLOW = 10        # Max additional connections beyond pool_size  
DEFAULT_POOL_TIMEOUT = 30        # Seconds to wait for connection
DEFAULT_POOL_RECYCLE = 3600      # Recycle connections after 1 hour (seconds)
DEFAULT_POOL_PRE_PING = True     # Test connections before use

# Alternative: pool_config dict for backward compatibility
pool_config = {
    "pool_size": DEFAULT_POOL_SIZE,
    "max_overflow": DEFAULT_MAX_OVERFLOW,
    "pool_timeout": DEFAULT_POOL_TIMEOUT,
    "pool_recycle": DEFAULT_POOL_RECYCLE,
    "pool_pre_ping": DEFAULT_POOL_PRE_PING,
}
