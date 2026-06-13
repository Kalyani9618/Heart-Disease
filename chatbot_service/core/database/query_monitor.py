"""
Database Query Monitoring and Health Check Module
=================================================
Following ChatGPT's PostgreSQL optimization recommendations:
- Query timeout handling
- Slow query logging
- Connection pool monitoring
- Health check endpoints
- Performance metrics collection
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict
import statistics
import functools

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""
    query_hash: str
    query_template: str
    execution_time_ms: float
    rows_affected: int
    timestamp: datetime
    was_cached: bool = False
    timeout_occurred: bool = False
    error: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for query analysis"""
    total_queries: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    timeouts: int = 0
    errors: int = 0
    execution_times: List[float] = field(default_factory=list)
    
    def calculate_percentiles(self):
        """Calculate percentile metrics"""
        if self.execution_times:
            sorted_times = sorted(self.execution_times)
            n = len(sorted_times)
            self.p95_time_ms = sorted_times[int(n * 0.95)] if n > 0 else 0
            self.p99_time_ms = sorted_times[int(n * 0.99)] if n > 0 else 0
            self.avg_time_ms = statistics.mean(sorted_times)
            self.max_time_ms = max(sorted_times)
            self.min_time_ms = min(sorted_times)


class SlowQueryLogger:
    """
    Logs slow queries for analysis and optimization.
    Implements ChatGPT's recommendation for identifying performance bottlenecks.
    """
    
    def __init__(
        self,
        slow_query_threshold_ms: float = 100.0,
        max_log_entries: int = 1000,
        log_file: Optional[str] = None
    ):
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.max_log_entries = max_log_entries
        self.log_file = log_file
        self.slow_queries: List[QueryMetrics] = []
        self._lock = asyncio.Lock()
    
    async def log_query(self, metrics: QueryMetrics):
        """Log a slow query if it exceeds threshold"""
        if metrics.execution_time_ms >= self.slow_query_threshold_ms:
            async with self._lock:
                self.slow_queries.append(metrics)
                
                # Keep only recent entries
                if len(self.slow_queries) > self.max_log_entries:
                    self.slow_queries = self.slow_queries[-self.max_log_entries:]
                
                # Log to file if configured
                if self.log_file:
                    await self._write_to_file(metrics)
                
                logger.warning(
                    f"SLOW QUERY detected: {metrics.execution_time_ms:.2f}ms - "
                    f"Query: {metrics.query_template[:100]}..."
                )
    
    async def _write_to_file(self, metrics: QueryMetrics):
        """Write slow query to log file"""
        try:
            log_entry = (
                f"{metrics.timestamp.isoformat()} | "
                f"{metrics.execution_time_ms:.2f}ms | "
                f"Rows: {metrics.rows_affected} | "
                f"Query: {metrics.query_template}\n"
            )
            
            # Use async file writing
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: open(self.log_file, 'a').write(log_entry)
            )
        except Exception as e:
            logger.error(f"Failed to write slow query log: {e}")
    
    def get_slow_queries(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[QueryMetrics]:
        """Get recent slow queries"""
        queries = self.slow_queries
        if since:
            queries = [q for q in queries if q.timestamp >= since]
        return queries[-limit:]
    
    def get_top_slow_queries(self, limit: int = 10) -> List[QueryMetrics]:
        """Get top N slowest queries"""
        return sorted(
            self.slow_queries,
            key=lambda x: x.execution_time_ms,
            reverse=True
        )[:limit]


class QueryPerformanceMonitor:
    """
    Monitors query performance and collects metrics.
    Provides insights for database optimization.
    """
    
    def __init__(
        self,
        collection_interval_seconds: int = 60,
        retention_hours: int = 24
    ):
        self.collection_interval = collection_interval_seconds
        self.retention_hours = retention_hours
        self.metrics_by_query: Dict[str, List[QueryMetrics]] = defaultdict(list)
        self.global_metrics: List[QueryMetrics] = []
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the metrics cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
        logger.info("Query performance monitor started")
    
    async def stop(self):
        """Stop the metrics cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Query performance monitor stopped")
    
    async def record_query(self, metrics: QueryMetrics):
        """Record query metrics"""
        async with self._lock:
            self.metrics_by_query[metrics.query_hash].append(metrics)
            self.global_metrics.append(metrics)
    
    async def _cleanup_old_metrics(self):
        """Periodically clean up old metrics"""
        while True:
            await asyncio.sleep(self.collection_interval)
            cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
            
            async with self._lock:
                # Clean query-specific metrics
                for query_hash in list(self.metrics_by_query.keys()):
                    self.metrics_by_query[query_hash] = [
                        m for m in self.metrics_by_query[query_hash]
                        if m.timestamp >= cutoff
                    ]
                    if not self.metrics_by_query[query_hash]:
                        del self.metrics_by_query[query_hash]
                
                # Clean global metrics
                self.global_metrics = [
                    m for m in self.global_metrics
                    if m.timestamp >= cutoff
                ]
    
    def get_aggregated_metrics(
        self,
        query_hash: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> AggregatedMetrics:
        """Get aggregated metrics for analysis"""
        metrics_list = (
            self.metrics_by_query.get(query_hash, [])
            if query_hash else self.global_metrics
        )
        
        if since:
            metrics_list = [m for m in metrics_list if m.timestamp >= since]
        
        aggregated = AggregatedMetrics()
        aggregated.total_queries = len(metrics_list)
        
        for m in metrics_list:
            aggregated.total_time_ms += m.execution_time_ms
            aggregated.execution_times.append(m.execution_time_ms)
            if m.was_cached:
                aggregated.cache_hits += 1
            else:
                aggregated.cache_misses += 1
            if m.timeout_occurred:
                aggregated.timeouts += 1
            if m.error:
                aggregated.errors += 1
        
        aggregated.calculate_percentiles()
        return aggregated
    
    def get_query_patterns(self) -> Dict[str, AggregatedMetrics]:
        """Get metrics grouped by query pattern"""
        patterns = {}
        for query_hash, metrics_list in self.metrics_by_query.items():
            if metrics_list:
                patterns[query_hash] = self.get_aggregated_metrics(query_hash)
        return patterns


class ConnectionPoolMonitor:
    """
    Monitors PostgreSQL connection pool health.
    Tracks pool utilization and connection lifecycle.
    """
    
    def __init__(self, pool: Any):
        """
        Args:
            pool: asyncpg connection pool instance
        """
        self.pool = pool
        self.connection_events: List[Dict] = []
        self._lock = asyncio.Lock()
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics"""
        if not self.pool:
            return {"status": "pool_not_initialized"}
        
        try:
            return {
                "size": self.pool.get_size(),
                "min_size": self.pool.get_min_size(),
                "max_size": self.pool.get_max_size(),
                "free_size": self.pool.get_idle_size(),
                "used_size": self.pool.get_size() - self.pool.get_idle_size(),
                "utilization_percent": (
                    (self.pool.get_size() - self.pool.get_idle_size()) / 
                    self.pool.get_max_size() * 100
                ) if self.pool.get_max_size() > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting pool stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def record_connection_event(
        self,
        event_type: str,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Record a connection event"""
        async with self._lock:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "duration_ms": duration_ms,
                "error": error
            }
            self.connection_events.append(event)
            
            # Keep only recent events
            if len(self.connection_events) > 1000:
                self.connection_events = self.connection_events[-1000:]
    
    def get_connection_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get recent connection events"""
        events = self.connection_events
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        return events[-limit:]


class DatabaseHealthChecker:
    """
    Health check endpoint for database monitoring.
    Used by load balancers and orchestration systems.
    """
    
    def __init__(
        self,
        db_pool: Any,
        redis_client: Any = None,
        check_interval_seconds: int = 30
    ):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.check_interval = check_interval_seconds
        self.last_health_check: Optional[Dict] = None
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start periodic health checks"""
        self._health_check_task = asyncio.create_task(self._periodic_health_check())
        logger.info("Database health checker started")
    
    async def stop(self):
        """Stop health checks"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Database health checker stopped")
    
    async def _periodic_health_check(self):
        """Run periodic health checks"""
        while True:
            self.last_health_check = await self.check_health()
            await asyncio.sleep(self.check_interval)
    
    async def check_health(self, timeout_seconds: float = 5.0) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dict with health status for each component
        """
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "components": {}
        }
        
        # Check PostgreSQL
        health["components"]["postgresql"] = await self._check_postgresql(timeout_seconds)
        
        # Check Redis if configured
        if self.redis_client:
            health["components"]["redis"] = await self._check_redis(timeout_seconds)
        
        # Determine overall status
        component_statuses = [c["status"] for c in health["components"].values()]
        if all(s == "healthy" for s in component_statuses):
            health["status"] = "healthy"
        elif any(s == "unhealthy" for s in component_statuses):
            health["status"] = "unhealthy"
        else:
            health["status"] = "degraded"
        
        return health
    
    async def _check_postgresql(self, timeout: float) -> Dict[str, Any]:
        """Check PostgreSQL connectivity and performance"""
        result = {
            "status": "unknown",
            "latency_ms": None,
            "pool_stats": None,
            "error": None
        }
        
        try:
            start = time.perf_counter()
            
            async with asyncio.timeout(timeout):
                if self.db_pool:
                    async with self.db_pool.acquire() as conn:
                        # Simple health query
                        await conn.fetchval("SELECT 1")
                        
                        result["latency_ms"] = (time.perf_counter() - start) * 1000
                        result["status"] = "healthy"
                        
                        # Get pool stats
                        result["pool_stats"] = {
                            "size": self.db_pool.get_size(),
                            "idle": self.db_pool.get_idle_size(),
                            "max": self.db_pool.get_max_size()
                        }
                else:
                    result["status"] = "unhealthy"
                    result["error"] = "Connection pool not initialized"
                    
        except asyncio.TimeoutError:
            result["status"] = "unhealthy"
            result["error"] = f"Health check timed out after {timeout}s"
        except Exception as e:
            result["status"] = "unhealthy"
            result["error"] = str(e)
        
        return result
    
    async def _check_redis(self, timeout: float) -> Dict[str, Any]:
        """Check Redis connectivity"""
        result = {
            "status": "unknown",
            "latency_ms": None,
            "error": None
        }
        
        try:
            start = time.perf_counter()
            
            async with asyncio.timeout(timeout):
                await self.redis_client.ping()
                result["latency_ms"] = (time.perf_counter() - start) * 1000
                result["status"] = "healthy"
                
        except asyncio.TimeoutError:
            result["status"] = "unhealthy"
            result["error"] = f"Redis health check timed out after {timeout}s"
        except Exception as e:
            result["status"] = "unhealthy"
            result["error"] = str(e)
        
        return result


def query_timeout(timeout_seconds: float = 30.0):
    """
    Decorator for adding timeout to database queries.
    
    Usage:
        @query_timeout(10.0)
        async def my_query():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                async with asyncio.timeout(timeout_seconds):
                    return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                logger.error(f"Query timeout after {timeout_seconds}s: {func.__name__}")
                raise TimeoutError(f"Query timed out after {timeout_seconds} seconds")
        return wrapper
    return decorator


def track_query_performance(monitor: QueryPerformanceMonitor, slow_logger: SlowQueryLogger):
    """
    Decorator for tracking query performance.
    
    Usage:
        @track_query_performance(monitor, slow_logger)
        async def my_query():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            query_hash = func.__name__
            error = None
            result = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                execution_time = (time.perf_counter() - start) * 1000
                
                metrics = QueryMetrics(
                    query_hash=query_hash,
                    query_template=func.__name__,
                    execution_time_ms=execution_time,
                    rows_affected=len(result) if isinstance(result, list) else 1,
                    timestamp=datetime.utcnow(),
                    error=error
                )
                
                await monitor.record_query(metrics)
                await slow_logger.log_query(metrics)
        
        return wrapper
    return decorator


class QueryTimeoutManager:
    """
    Manages query timeouts with different tiers.
    Different query types get different timeout limits.
    """
    
    # Default timeout tiers in seconds
    TIMEOUT_TIERS = {
        "fast": 5.0,      # Simple lookups
        "normal": 30.0,   # Standard queries
        "slow": 60.0,     # Complex aggregations
        "batch": 120.0,   # Batch operations
        "maintenance": 300.0  # Maintenance operations
    }
    
    def __init__(self, custom_tiers: Optional[Dict[str, float]] = None):
        self.tiers = {**self.TIMEOUT_TIERS, **(custom_tiers or {})}
    
    def get_timeout(self, tier: str) -> float:
        """Get timeout for a specific tier"""
        return self.tiers.get(tier, self.tiers["normal"])
    
    async def execute_with_timeout(
        self,
        coro,
        tier: str = "normal",
        fallback: Any = None
    ) -> Any:
        """
        Execute coroutine with tier-based timeout.
        
        Args:
            coro: Coroutine to execute
            tier: Timeout tier
            fallback: Value to return on timeout (if None, raises TimeoutError)
        """
        timeout = self.get_timeout(tier)
        
        try:
            async with asyncio.timeout(timeout):
                return await coro
        except asyncio.TimeoutError:
            logger.warning(f"Query timed out after {timeout}s (tier: {tier})")
            if fallback is not None:
                return fallback
            raise TimeoutError(f"Query timed out after {timeout}s")


# Global instances for easy access
_performance_monitor: Optional[QueryPerformanceMonitor] = None
_slow_query_logger: Optional[SlowQueryLogger] = None
_health_checker: Optional[DatabaseHealthChecker] = None
_timeout_manager: Optional[QueryTimeoutManager] = None


async def initialize_monitoring(
    db_pool: Any,
    redis_client: Any = None,
    slow_query_threshold_ms: float = 100.0,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initialize all monitoring components.
    
    Returns:
        Dict with all monitoring instances
    """
    global _performance_monitor, _slow_query_logger, _health_checker, _timeout_manager
    
    _slow_query_logger = SlowQueryLogger(
        slow_query_threshold_ms=slow_query_threshold_ms,
        log_file=log_file
    )
    
    _performance_monitor = QueryPerformanceMonitor()
    await _performance_monitor.start()
    
    _health_checker = DatabaseHealthChecker(
        db_pool=db_pool,
        redis_client=redis_client
    )
    await _health_checker.start()
    
    _timeout_manager = QueryTimeoutManager()
    
    logger.info("Database monitoring initialized")
    
    return {
        "performance_monitor": _performance_monitor,
        "slow_query_logger": _slow_query_logger,
        "health_checker": _health_checker,
        "timeout_manager": _timeout_manager
    }


async def shutdown_monitoring():
    """Shutdown all monitoring components"""
    global _performance_monitor, _health_checker
    
    if _performance_monitor:
        await _performance_monitor.stop()
    
    if _health_checker:
        await _health_checker.stop()
    
    logger.info("Database monitoring shutdown complete")


def get_performance_monitor() -> Optional[QueryPerformanceMonitor]:
    """Get global performance monitor instance"""
    return _performance_monitor


def get_slow_query_logger() -> Optional[SlowQueryLogger]:
    """Get global slow query logger instance"""
    return _slow_query_logger


def get_health_checker() -> Optional[DatabaseHealthChecker]:
    """Get global health checker instance"""
    return _health_checker


def get_timeout_manager() -> Optional[QueryTimeoutManager]:
    """Get global timeout manager instance"""
    return _timeout_manager
