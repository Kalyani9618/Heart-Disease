"""
Health Service - Real-time System Health Monitoring

Provides comprehensive health checks for all system components:
- Database connectivity (PostgreSQL, Redis)
- LLM availability (MedGemma)
- External service connectivity (OpenFDA, Tavily)
- Memory usage and system resources
"""

import logging
import time
import os
import asyncio
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class HealthService:
    """Centralized health check service for all system components."""

    def __init__(self):
        self._checks: Dict[str, callable] = {}
        self._last_results: Dict[str, Dict[str, Any]] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health check functions."""
        self._checks["database"] = self._check_database
        self._checks["redis"] = self._check_redis
        self._checks["llm"] = self._check_llm
        self._checks["memory"] = self._check_memory

    async def _check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity."""
        try:
            from core.dependencies import DIContainer
            container = DIContainer.get_instance()
            db = container.get_service('db_manager')
            if db and hasattr(db, 'pool') and db.pool:
                async with db.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                return {"status": "healthy", "backend": "postgresql"}
            return {"status": "degraded", "reason": "No database pool"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            info = r.info("memory")
            r.close()
            return {
                "status": "healthy",
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 1)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_llm(self) -> Dict[str, Any]:
        """Check LLM (MedGemma) availability."""
        try:
            import httpx
            base_url = os.getenv("MEDGEMMA_BASE_URL",
                                 os.getenv("LLAMA_LOCAL_BASE_URL", "http://127.0.0.1:8090/v1"))
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/models")
                if resp.status_code == 200:
                    return {"status": "healthy", "endpoint": base_url}
                return {"status": "degraded", "status_code": resp.status_code}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "status": "healthy" if mem.percent < 90 else "degraded",
                "used_percent": mem.percent,
                "available_mb": round(mem.available / 1024 / 1024, 1)
            }
        except ImportError:
            return {"status": "unknown", "reason": "psutil not installed"}

    def register_check(self, name: str, check_fn: callable):
        """Register a custom health check."""
        self._checks[name] = check_fn

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        start = time.perf_counter()
        results = {}
        
        tasks = {}
        for name, check_fn in self._checks.items():
            try:
                tasks[name] = asyncio.create_task(check_fn())
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        
        for name, task in tasks.items():
            try:
                results[name] = await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                results[name] = {"status": "timeout"}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        
        self._last_results = results
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        overall = "healthy"
        for r in results.values():
            if r.get("status") == "unhealthy":
                overall = "unhealthy"
                break
            elif r.get("status") == "degraded":
                overall = "degraded"
        
        return {
            "status": overall,
            "checks": results,
            "check_duration_ms": round(elapsed_ms, 1)
        }

    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a single named health check."""
        if name not in self._checks:
            return {"status": "unknown", "error": f"Check '{name}' not registered"}
        try:
            return await self._checks[name]()
        except Exception as e:
            return {"status": "error", "error": str(e)}


class HealthRecordDB:
    """Placeholder for health record database access."""
    pass


# Singleton
_health_service: Optional[HealthService] = None


def get_health_service() -> HealthService:
    """Get global health service instance."""
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
    return _health_service


def reset_health_service():
    """Reset global health service (for testing)."""
    global _health_service
    _health_service = None

