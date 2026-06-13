"""
RAG and Memory Health Check Routes

Provides endpoints for monitoring RAG and memory system performance,
cache statistics, and optimization metrics.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag-memory-health"])


# Response Models
class CacheStats(BaseModel):
    size: int
    max_size: int
    hits: int
    misses: int
    hit_rate: float
    ttl: int


class TieredCacheStats(BaseModel):
    l1: CacheStats
    l2_available: bool


class BatchWriterStats(BaseModel):
    pending_items: int
    total_items_written: int
    total_batches: int
    avg_flush_time_ms: float
    time_since_last_flush: float


class RAGHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    vector_store_available: bool
    embedding_service_available: bool
    cache_stats: Optional[Dict[str, Any]] = None
    collections: Optional[Dict[str, int]] = None


class MemoryHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    cache_stats: Optional[TieredCacheStats] = None
    batch_writer_stats: Optional[BatchWriterStats] = None


class RAGMetricsResponse(BaseModel):
    period_minutes: int
    total_searches: int
    avg_search_latency_ms: float
    cache_hit_rate: float
    embedding_generation_time_ms: float


class MemoryMetricsResponse(BaseModel):
    period_minutes: int
    queries: Dict[str, Any]
    writes: Dict[str, Any]


# Global references to optimization instances
_rag_optimizer = None
_memory_optimizer = None
_rag_performance_monitor = None
_memory_performance_monitor = None


def set_rag_optimizer(optimizer, monitor=None):
    """Set the RAG optimizer instance for health checks."""
    global _rag_optimizer, _rag_performance_monitor
    _rag_optimizer = optimizer
    _rag_performance_monitor = monitor


def set_memory_optimizer(optimizer, monitor=None):
    """Set the memory optimizer instance for health checks."""
    global _memory_optimizer, _memory_performance_monitor
    _memory_optimizer = optimizer
    _memory_performance_monitor = monitor


@router.get("/rag/health", response_model=RAGHealthResponse)
async def get_rag_health():
    """Get RAG system health status."""
    try:
        status = "healthy"
        vector_store_available = False
        embedding_service_available = False
        cache_stats = None
        collections = None
        
        # Check if RAG optimizer is available
        if _rag_optimizer:
            cache_stats = _rag_optimizer.get_stats()
            vector_store_available = True
            embedding_service_available = True
            
            # Try to get collection counts
            try:
                collections = await _rag_optimizer.get_collection_counts()
            except Exception as e:
                logger.warning(f"Could not get collection counts: {e}")
        else:
            # Try to import and check basic components
            try:
                from rag.embedding import get_embedding_service
                embedding_service = get_embedding_service()
                embedding_service_available = embedding_service is not None
            except Exception:
                pass
            
            try:
                from rag.store.chromadb_store import ChromaDBVectorStore
                vector_store_available = True
            except Exception:
                pass
        
        if not vector_store_available:
            status = "degraded"
        
        return RAGHealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            vector_store_available=vector_store_available,
            embedding_service_available=embedding_service_available,
            cache_stats=cache_stats,
            collections=collections
        )
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return RAGHealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            vector_store_available=False,
            embedding_service_available=False
        )


@router.get("/memory/health", response_model=MemoryHealthResponse)
async def get_memory_health():
    """Get memory system health status."""
    try:
        status = "healthy"
        cache_stats = None
        batch_writer_stats = None
        
        if _memory_optimizer:
            stats = _memory_optimizer.get_stats()
            cache_stats = stats.get("cache")
            batch_writer_stats = stats.get("batch_writer")
        else:
            status = "degraded"
        
        return MemoryHealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            cache_stats=cache_stats,
            batch_writer_stats=batch_writer_stats
        )
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        return MemoryHealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow()
        )


@router.get("/rag/metrics")
async def get_rag_metrics(minutes: int = Query(60, ge=1, le=1440)):
    """Get RAG performance metrics for the last N minutes."""
    if not _rag_performance_monitor:
        raise HTTPException(
            status_code=503, 
            detail="RAG performance monitor not available"
        )
    
    try:
        metrics = _rag_performance_monitor.get_metrics(minutes)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get RAG metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/metrics")
async def get_memory_metrics(minutes: int = Query(60, ge=1, le=1440)):
    """Get memory performance metrics for the last N minutes."""
    if not _memory_performance_monitor:
        raise HTTPException(
            status_code=503,
            detail="Memory performance monitor not available"
        )
    
    try:
        metrics = _memory_performance_monitor.get_metrics(minutes)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get memory metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/cache/clear")
async def clear_rag_cache():
    """Clear RAG caches."""
    if not _rag_optimizer:
        raise HTTPException(
            status_code=503,
            detail="RAG optimizer not available"
        )
    
    try:
        # Clear embedding cache
        _rag_optimizer.embedding_cache.l1.clear()
        
        # Clear result cache
        _rag_optimizer.result_cache.l1.clear()
        
        return {"status": "success", "message": "RAG caches cleared"}
    except Exception as e:
        logger.error(f"Failed to clear RAG cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/cache/clear")
async def clear_memory_cache():
    """Clear memory caches."""
    if not _memory_optimizer:
        raise HTTPException(
            status_code=503,
            detail="Memory optimizer not available"
        )
    
    try:
        _memory_optimizer.cache.l1.clear()
        return {"status": "success", "message": "Memory cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear memory cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/flush")
async def flush_memory_batch():
    """Force flush pending memory batch writes."""
    if not _memory_optimizer:
        raise HTTPException(
            status_code=503,
            detail="Memory optimizer not available"
        )
    
    try:
        count = await _memory_optimizer.flush_pending()
        return {
            "status": "success",
            "message": f"Flushed {count} pending items"
        }
    except Exception as e:
        logger.error(f"Failed to flush memory batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/combined/health")
async def get_combined_health():
    """Get combined health status for RAG and memory systems."""
    rag_health = await get_rag_health()
    memory_health = await get_memory_health()
    
    # Determine overall status
    if rag_health.status == "unhealthy" or memory_health.status == "unhealthy":
        overall_status = "unhealthy"
    elif rag_health.status == "degraded" or memory_health.status == "degraded":
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow(),
        "rag": rag_health.dict(),
        "memory": memory_health.dict()
    }


@router.get("/vector/indexes")
async def get_vector_indexes():
    """Get information about vector indexes in the database."""
    try:
        from core.database.postgres_db import get_database
        db = await get_database()
        
        async with db.pool.acquire() as conn:
            # Get HNSW indexes
            indexes = await conn.fetch("""
                SELECT 
                    indexname,
                    tablename,
                    indexdef
                FROM pg_indexes
                WHERE indexdef LIKE '%hnsw%' OR indexdef LIKE '%vector%'
                ORDER BY tablename
            """)
            
            return {
                "indexes": [
                    {
                        "name": idx["indexname"],
                        "table": idx["tablename"],
                        "definition": idx["indexdef"]
                    }
                    for idx in indexes
                ]
            }
    except Exception as e:
        logger.error(f"Failed to get vector indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/materialized-views")
async def get_materialized_views():
    """Get RAG materialized view information."""
    try:
        from core.database.postgres_db import get_database
        db = await get_database()
        
        async with db.pool.acquire() as conn:
            # Get materialized views
            views = await conn.fetch("""
                SELECT 
                    matviewname,
                    hasindexes,
                    ispopulated
                FROM pg_matviews
                WHERE matviewname LIKE 'mv_%'
            """)
            
            # Get collection stats if available
            collection_stats = None
            try:
                stats = await conn.fetch("""
                    SELECT * FROM mv_vector_collection_stats
                """)
                collection_stats = [dict(s) for s in stats]
            except Exception:
                pass
            
            return {
                "views": [
                    {
                        "name": v["matviewname"],
                        "has_indexes": v["hasindexes"],
                        "is_populated": v["ispopulated"]
                    }
                    for v in views
                ],
                "collection_stats": collection_stats
            }
    except Exception as e:
        logger.error(f"Failed to get materialized views: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/materialized-views/refresh")
async def refresh_materialized_views(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Refresh RAG materialized views."""
    try:
        from core.database.postgres_db import get_database
        db = await get_database()
        
        async with db.pool.acquire() as conn:
            await conn.execute("SELECT refresh_rag_materialized_views()")
        
        return {"status": "success", "message": "Materialized views refreshed"}
    except Exception as e:
        logger.error(f"Failed to refresh materialized views: {e}")
        raise HTTPException(status_code=500, detail=str(e))

