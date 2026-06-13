"""
Database Health and Monitoring Routes
=====================================
Exposes database health check and performance metrics endpoints.

Endpoints:
- GET /db/health - Database health status
- GET /db/metrics - Performance metrics summary
- GET /db/slow-queries - Recent slow queries
"""


import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================

class ComponentHealth(BaseModel):
    """Health status for a single component."""
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    pool_stats: Optional[Dict[str, Any]] = None


class DatabaseHealthResponse(BaseModel):
    """Database health check response."""
    timestamp: str
    status: str  # healthy, degraded, unhealthy
    components: Dict[str, ComponentHealth]


class QueryMetricsResponse(BaseModel):
    """Aggregated query metrics response."""
    total_queries: int
    total_time_ms: float
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    cache_hits: int
    cache_misses: int
    cache_hit_ratio: float
    timeouts: int
    errors: int


class SlowQueryResponse(BaseModel):
    """Slow query log entry."""
    query_hash: str
    query_template: str
    execution_time_ms: float
    rows_affected: int
    timestamp: str
    was_cached: bool
    timeout_occurred: bool
    error: Optional[str] = None


# ============================================================================
# Health Check Endpoints
# ============================================================================

@router.get("/health", response_model=DatabaseHealthResponse)
async def database_health_check():
    """
    Get current database health status.
    
    Checks connectivity and latency for:
    - PostgreSQL database
    - Redis cache (if configured)
    
    Returns overall health status and component details.
    """
    try:
        from app_lifespan import get_db_health_status
        
        health_status = get_db_health_status()
        
        if health_status.get("status") == "unknown":
            # Health checker not initialized, do a quick check
            return DatabaseHealthResponse(
                timestamp=datetime.utcnow().isoformat(),
                status="unknown",
                components={
                    "postgresql": ComponentHealth(
                        status="unknown",
                        error="Health checker not initialized"
                    )
                }
            )
        
        # Convert to response model
        components = {}
        for name, data in health_status.get("components", {}).items():
            components[name] = ComponentHealth(
                status=data.get("status", "unknown"),
                latency_ms=data.get("latency_ms"),
                error=data.get("error"),
                pool_stats=data.get("pool_stats")
            )
        
        return DatabaseHealthResponse(
            timestamp=health_status.get("timestamp", datetime.utcnow().isoformat()),
            status=health_status.get("status", "unknown"),
            components=components
        )
        
    except ImportError:
        return DatabaseHealthResponse(
            timestamp=datetime.utcnow().isoformat(),
            status="unknown",
            components={
                "postgresql": ComponentHealth(
                    status="unknown",
                    error="app_lifespan module not available"
                )
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return DatabaseHealthResponse(
            timestamp=datetime.utcnow().isoformat(),
            status="unhealthy",
            components={
                "postgresql": ComponentHealth(
                    status="unhealthy",
                    error=str(e)
                )
            }
        )


@router.get("/metrics", response_model=QueryMetricsResponse)
async def get_query_metrics(
    minutes: int = Query(default=60, ge=1, le=1440, description="Time window in minutes")
):
    """
    Get aggregated query performance metrics.
    
    Returns metrics for the specified time window:
    - Total queries executed
    - Average/max/min execution times
    - P95/P99 latencies
    - Cache hit ratio
    - Error and timeout counts
    """
    try:
        from core.database.query_monitor import get_performance_monitor
        
        monitor = get_performance_monitor()
        if not monitor:
            raise HTTPException(
                status_code=503,
                detail="Performance monitor not initialized"
            )
        
        since = datetime.utcnow() - timedelta(minutes=minutes)
        metrics = monitor.get_aggregated_metrics(since=since)
        
        # Calculate cache hit ratio
        total_cache_ops = metrics.cache_hits + metrics.cache_misses
        cache_hit_ratio = (
            metrics.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        )
        
        return QueryMetricsResponse(
            total_queries=metrics.total_queries,
            total_time_ms=metrics.total_time_ms,
            avg_time_ms=metrics.avg_time_ms,
            max_time_ms=metrics.max_time_ms if metrics.max_time_ms != float('inf') else 0.0,
            min_time_ms=metrics.min_time_ms if metrics.min_time_ms != float('inf') else 0.0,
            p95_time_ms=metrics.p95_time_ms,
            p99_time_ms=metrics.p99_time_ms,
            cache_hits=metrics.cache_hits,
            cache_misses=metrics.cache_misses,
            cache_hit_ratio=cache_hit_ratio,
            timeouts=metrics.timeouts,
            errors=metrics.errors
        )
        
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Query monitor module not available"
        )
    except Exception as e:
        logger.error(f"Failed to get query metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/slow-queries", response_model=List[SlowQueryResponse])
async def get_slow_queries(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum queries to return"),
    minutes: int = Query(default=60, ge=1, le=1440, description="Time window in minutes")
):
    """
    Get recent slow queries.
    
    Returns queries that exceeded the slow query threshold (default 100ms).
    Useful for identifying performance bottlenecks.
    """
    try:
        from core.database.query_monitor import get_slow_query_logger
        
        slow_logger = get_slow_query_logger()
        if not slow_logger:
            raise HTTPException(
                status_code=503,
                detail="Slow query logger not initialized"
            )
        
        since = datetime.utcnow() - timedelta(minutes=minutes)
        slow_queries = slow_logger.get_slow_queries(since=since, limit=limit)
        
        return [
            SlowQueryResponse(
                query_hash=q.query_hash,
                query_template=q.query_template[:500],  # Truncate long queries
                execution_time_ms=q.execution_time_ms,
                rows_affected=q.rows_affected,
                timestamp=q.timestamp.isoformat(),
                was_cached=q.was_cached,
                timeout_occurred=q.timeout_occurred,
                error=q.error
            )
            for q in slow_queries
        ]
        
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Query monitor module not available"
        )
    except Exception as e:
        logger.error(f"Failed to get slow queries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get slow queries: {str(e)}"
        )


@router.get("/top-slow-queries", response_model=List[SlowQueryResponse])
async def get_top_slow_queries(
    limit: int = Query(default=10, ge=1, le=50, description="Number of slowest queries")
):
    """
    Get the N slowest queries overall.
    
    Returns the slowest queries ever recorded (within retention period).
    """
    try:
        from core.database.query_monitor import get_slow_query_logger
        
        slow_logger = get_slow_query_logger()
        if not slow_logger:
            raise HTTPException(
                status_code=503,
                detail="Slow query logger not initialized"
            )
        
        top_queries = slow_logger.get_top_slow_queries(limit=limit)
        
        return [
            SlowQueryResponse(
                query_hash=q.query_hash,
                query_template=q.query_template[:500],
                execution_time_ms=q.execution_time_ms,
                rows_affected=q.rows_affected,
                timestamp=q.timestamp.isoformat(),
                was_cached=q.was_cached,
                timeout_occurred=q.timeout_occurred,
                error=q.error
            )
            for q in top_queries
        ]
        
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Query monitor module not available"
        )
    except Exception as e:
        logger.error(f"Failed to get top slow queries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get top slow queries: {str(e)}"
        )


@router.get("/pool-stats")
async def get_connection_pool_stats():
    """
    Get PostgreSQL connection pool statistics.
    
    Returns:
    - Pool size (current connections)
    - Idle connections
    - Max pool size
    - Utilization percentage
    """
    try:
        from app_lifespan import get_db_monitoring
        
        monitoring = get_db_monitoring()
        if not monitoring:
            raise HTTPException(
                status_code=503,
                detail="Database monitoring not initialized"
            )
        
        health_checker = monitoring.get("health_checker")
        if not health_checker:
            raise HTTPException(
                status_code=503,
                detail="Health checker not available"
            )
        
        # Get pool stats from last health check
        last_check = health_checker.last_health_check
        if last_check and "components" in last_check:
            pg_component = last_check["components"].get("postgresql", {})
            pool_stats = pg_component.get("pool_stats", {})
            
            return {
                "timestamp": last_check.get("timestamp"),
                "pool_size": pool_stats.get("size", 0),
                "idle_connections": pool_stats.get("idle", 0),
                "max_size": pool_stats.get("max", 0),
                "utilization_percent": (
                    (pool_stats.get("size", 0) - pool_stats.get("idle", 0)) / 
                    pool_stats.get("max", 1) * 100
                ) if pool_stats.get("max", 0) > 0 else 0
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": "No pool stats available"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pool stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pool stats: {str(e)}"
        )
