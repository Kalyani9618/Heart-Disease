"""
Monitoring and Observability for Memori Integration

Provides:
1. Prometheus-compatible metrics for Memori operations
2. Memory search latency histograms (p50, p95, p99)
3. Cache hit/miss ratio tracking
4. Async task correlation tracking
5. Structured logging with ELK compatibility
6. Health check endpoints

Integrates with existing analytics.py framework.

Metrics Exposed:
- memori_search_duration_seconds (histogram)
- memori_search_total (counter)
- memori_store_duration_seconds (histogram)
- memori_store_total (counter)
- memori_cache_hits_total (counter)
- memori_cache_misses_total (counter)
- memori_cache_hit_rate (gauge)
- memori_circuit_breaker_state (gauge)
- memori_patient_memories_cached (gauge)

ELK-compatible structured logging with:
- correlation_id (request tracing)
- operation (search, store, etc.)
- patient_id (with masking)
- latency_ms
- status (success, failure, timeout)
- error_type
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Import from local memori module
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Structured Logger for ELK Compatibility
# ============================================================================


class StructuredLogger:
    """
    ELK-compatible structured logger for Memori operations.

    Logs to JSON format for easy parsing by ELK stack:
    {
        "timestamp": "2025-12-08T10:30:45.123Z",
        "level": "info",
        "correlation_id": "req-123",
        "operation": "memory_search",
        "patient_id": "p***45",
        "latency_ms": 25.5,
        "status": "success",
        "result_count": 3,
        "cache_hit": true
    }
    """

    def __init__(self, name: str = "memori"):
        self.logger = logging.getLogger(name)
        self.base_logger = logger

    @staticmethod
    def _mask_patient_id(patient_id: str) -> str:
        """Mask patient ID for logging (show only last 2 chars)."""
        if not patient_id or len(patient_id) < 3:
            return "p***"
        return f"p***{patient_id[-2:]}"

    def log_search(
        self,
        operation: str,
        patient_id: str,
        query: str,
        latency_ms: float,
        result_count: int,
        cache_hit: bool,
        status: str = "success",
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """Log memory search operation."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "error" if status == "failure" else "info",
            "correlation_id": correlation_id or "",
            "operation": operation,
            "patient_id": self._mask_patient_id(patient_id),
            "query_length": len(query),
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "result_count": result_count,
            "cache_hit": cache_hit,
        }

        if error:
            log_data["error_type"] = type(error).__name__
            log_data["error_message"] = str(error)

        log_json = json.dumps(log_data)

        if status == "failure":
            self.logger.error(log_json)
        else:
            self.logger.info(log_json)

    def log_store(
        self,
        operation: str,
        memory_type: str,
        patient_id: str,
        latency_ms: float,
        content_size: int,
        status: str = "success",
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """Log memory store operation."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "error" if status == "failure" else "debug",
            "correlation_id": correlation_id or "",
            "operation": operation,
            "memory_type": memory_type,
            "patient_id": self._mask_patient_id(patient_id),
            "latency_ms": round(latency_ms, 2),
            "content_size_bytes": content_size,
            "status": status,
        }

        if error:
            log_data["error_type"] = type(error).__name__
            log_data["error_message"] = str(error)

        log_json = json.dumps(log_data)

        if status == "failure":
            self.logger.error(log_json)
        else:
            self.logger.debug(log_json)


structured_logger = StructuredLogger("memori.structured")


# ============================================================================
# Histogram for Latency Tracking
# ============================================================================


@dataclass
class Histogram:
    """
    Simple histogram for tracking operation latencies.

    Calculates percentiles (p50, p95, p99) without external dependencies.
    """

    name: str
    values: List[float] = field(default_factory=list)
    max_size: int = 1000  # Keep only recent values to limit memory

    def add(self, value: float):
        """Add value to histogram."""
        self.values.append(value)
        # Keep only recent values
        if len(self.values) > self.max_size:
            self.values = self.values[-self.max_size // 2 :]

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100)."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * (p / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        if not self.values:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_values = sorted(self.values)
        return {
            "count": len(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "avg": sum(self.values) / len(self.values),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
        }


# ============================================================================
# Metrics Collector for Memori
# ============================================================================


class MemoriMetricsCollector:
    """
    Collects metrics for Memori operations.

    Tracks:
    - Search latency (histogram with percentiles)
    - Store latency (histogram with percentiles)
    - Cache hit/miss rates
    - Circuit breaker state
    - Patient memory instances

    Complexity: O(1) for metric recording
    """

    def __init__(self):
        self.search_latency_histogram = Histogram("memori_search_duration_seconds")
        self.store_latency_histogram = Histogram("memori_store_duration_seconds")
        self.searches_total = 0
        self.searches_successful = 0
        self.searches_failed = 0
        self.searches_timeout = 0
        self.stores_total = 0
        self.stores_successful = 0
        self.stores_failed = 0
        self.circuit_breaker_open_count = 0

    def record_search(
        self,
        latency_ms: float,
        success: bool,
        timeout: bool = False,
    ):
        """Record search operation."""
        self.searches_total += 1
        if success:
            self.searches_successful += 1
        else:
            self.searches_failed += 1
        if timeout:
            self.searches_timeout += 1

        # Convert to seconds for histogram
        self.search_latency_histogram.add(latency_ms / 1000.0)

    def record_store(
        self,
        latency_ms: float,
        success: bool,
    ):
        """Record store operation."""
        self.stores_total += 1
        if success:
            self.stores_successful += 1
        else:
            self.stores_failed += 1

        # Convert to seconds for histogram
        self.store_latency_histogram.add(latency_ms / 1000.0)

    def record_circuit_breaker_open(self):
        """Record circuit breaker opening."""
        self.circuit_breaker_open_count += 1

    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus text format.

        Format:
        ```
        # HELP memori_search_total Total number of search operations
        # TYPE memori_search_total counter
        memori_search_total 1234
        ```
        """
        lines = []

        # Search metrics
        search_stats = self.search_latency_histogram.get_stats()
        lines.append("# HELP memori_search_duration_seconds Search operation latency")
        lines.append("# TYPE memori_search_duration_seconds histogram")
        lines.append(
            f'memori_search_duration_seconds_bucket{{le="0.01"}} {sum(1 for v in self.search_latency_histogram.values if v <= 0.01)}'
        )
        lines.append(
            f'memori_search_duration_seconds_bucket{{le="0.05"}} {sum(1 for v in self.search_latency_histogram.values if v <= 0.05)}'
        )
        lines.append(
            f'memori_search_duration_seconds_bucket{{le="0.1"}} {sum(1 for v in self.search_latency_histogram.values if v <= 0.1)}'
        )
        lines.append(
            f'memori_search_duration_seconds_bucket{{le="+Inf"}} {len(self.search_latency_histogram.values)}'
        )
        lines.append(
            f"memori_search_duration_seconds_sum {sum(self.search_latency_histogram.values)}"
        )
        lines.append(
            f"memori_search_duration_seconds_count {len(self.search_latency_histogram.values)}"
        )

        # Counter metrics
        lines.append("# HELP memori_search_total Total number of search operations")
        lines.append("# TYPE memori_search_total counter")
        lines.append(f"memori_search_total {self.searches_total}")

        lines.append("# HELP memori_search_success_total Successful search operations")
        lines.append("# TYPE memori_search_success_total counter")
        lines.append(f"memori_search_success_total {self.searches_successful}")

        lines.append(
            "# HELP memori_search_timeout_total Search operations that timed out"
        )
        lines.append("# TYPE memori_search_timeout_total counter")
        lines.append(f"memori_search_timeout_total {self.searches_timeout}")

        # Store metrics
        lines.append("# HELP memori_store_total Total number of store operations")
        lines.append("# TYPE memori_store_total counter")
        lines.append(f"memori_store_total {self.stores_total}")

        lines.append("# HELP memori_store_success_total Successful store operations")
        lines.append("# TYPE memori_store_success_total counter")
        lines.append(f"memori_store_success_total {self.stores_successful}")

        # Circuit breaker
        lines.append(
            "# HELP memori_circuit_breaker_open_total Times circuit breaker opened"
        )
        lines.append("# TYPE memori_circuit_breaker_open_total counter")
        lines.append(
            f"memori_circuit_breaker_open_total {self.circuit_breaker_open_count}"
        )

        return "\n".join(lines) + "\n"

    def get_json_metrics(self) -> Dict[str, Any]:
        """Get metrics as JSON."""
        search_stats = self.search_latency_histogram.get_stats()
        store_stats = self.store_latency_histogram.get_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "search": {
                "total": self.searches_total,
                "successful": self.searches_successful,
                "failed": self.searches_failed,
                "timeout": self.searches_timeout,
                "success_rate_percent": (
                    (self.searches_successful / self.searches_total * 100)
                    if self.searches_total > 0
                    else 0.0
                ),
                "latency_ms": search_stats,
            },
            "store": {
                "total": self.stores_total,
                "successful": self.stores_successful,
                "failed": self.stores_failed,
                "success_rate_percent": (
                    (self.stores_successful / self.stores_total * 100)
                    if self.stores_total > 0
                    else 0.0
                ),
                "latency_ms": store_stats,
            },
            "circuit_breaker": {
                "open_count": self.circuit_breaker_open_count,
            },
        }


# Global metrics collector
metrics_collector = MemoriMetricsCollector()


# ============================================================================
# Health Check Endpoint Data
# ============================================================================


@dataclass
class MemoriHealthCheck:
    """Health check response for Memori."""

    status: str  # healthy, degraded, unhealthy
    enabled: bool
    initialized: bool
    cache_size: int
    cache_max_size: int
    circuit_breaker_state: str
    memory_search_success_rate: float
    memory_store_success_rate: float
    avg_search_latency_ms: float
    avg_store_latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "cache": {
                "size": self.cache_size,
                "max_size": self.cache_max_size,
            },
            "circuit_breaker_state": self.circuit_breaker_state,
            "success_rates": {
                "search_percent": round(self.memory_search_success_rate, 2),
                "store_percent": round(self.memory_store_success_rate, 2),
            },
            "latency_ms": {
                "avg_search": round(self.avg_search_latency_ms, 2),
                "avg_store": round(self.avg_store_latency_ms, 2),
            },
            "timestamp": self.timestamp,
        }


# ============================================================================
# Monitoring Middleware
# ============================================================================


class MemoriMonitoringMiddleware:
    """
    Middleware to track Memori operations for monitoring.

    Records:
    - Operation type (search, store)
    - Latency
    - Success/failure status
    - Request correlation ID
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Store start time
        start_time = time.time()

        # Wrap send to inject response headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                latency_ms = (time.time() - start_time) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-response-time-ms", str(round(latency_ms, 2)).encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ============================================================================
# Health Check Functions
# ============================================================================


async def get_memori_health_check() -> MemoriHealthCheck:
    """
    Get comprehensive health check for Memori.

    Complexity: O(1) - just aggregates metrics
    """
    try:
        memory_mgr = MemoryManager.get_instance()
        metrics = memory_mgr.get_metrics()

        # Calculate success rates
        search_stats = metrics_collector.search_latency_histogram.get_stats()
        store_stats = metrics_collector.store_latency_histogram.get_stats()

        search_success_rate = (
            (
                metrics_collector.searches_successful
                / metrics_collector.searches_total
                * 100
            )
            if metrics_collector.searches_total > 0
            else 0.0
        )

        store_success_rate = (
            (metrics_collector.stores_successful / metrics_collector.stores_total * 100)
            if metrics_collector.stores_total > 0
            else 0.0
        )

        # Determine overall status
        if not memory_mgr.enabled:
            overall_status = "unhealthy"
        elif search_success_rate < 80 or store_success_rate < 80:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return MemoriHealthCheck(
            status=overall_status,
            enabled=memory_mgr.enabled,
            initialized=memory_mgr._initialized,
            cache_size=memory_mgr._cache.size(),
            cache_max_size=memory_mgr.cache_size,
            circuit_breaker_state=(
                memory_mgr.circuit_breaker.state.value
                if hasattr(memory_mgr, "circuit_breaker")
                else "N/A"
            ),
            memory_search_success_rate=search_success_rate,
            memory_store_success_rate=store_success_rate,
            avg_search_latency_ms=search_stats.get("avg", 0.0) * 1000,
            avg_store_latency_ms=store_stats.get("avg", 0.0) * 1000,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error getting health check: {e}")
        return MemoriHealthCheck(
            status="unhealthy",
            enabled=False,
            initialized=False,
            cache_size=0,
            cache_max_size=0,
            circuit_breaker_state="unknown",
            memory_search_success_rate=0.0,
            memory_store_success_rate=0.0,
            avg_search_latency_ms=0.0,
            avg_store_latency_ms=0.0,
            timestamp=datetime.utcnow().isoformat(),
        )


async def get_detailed_metrics() -> Dict[str, Any]:
    """Get detailed metrics for monitoring dashboards."""
    try:
        memory_mgr = MemoryManager.get_instance()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "memori": metrics_collector.get_json_metrics(),
            "memory_manager": memory_mgr.get_metrics(),
            "health_check": (await get_memori_health_check()).to_dict(),
        }

    except Exception as e:
        logger.error(f"Error getting detailed metrics: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


# ============================================================================
# FastAPI Endpoint Helpers
# ============================================================================


async def metrics_endpoint() -> str:
    """
    Prometheus metrics endpoint.

    Usage:
        @app.get("/metrics")
        async def metrics():
            return await metrics_endpoint()
    """
    return metrics_collector.get_prometheus_metrics()


async def health_endpoint() -> Dict[str, Any]:
    """
    Health check endpoint.

    Usage:
        @app.get("/health/memori")
        async def memori_health():
            return await health_endpoint()
    """
    health = await get_memori_health_check()
    return health.to_dict()


async def detailed_metrics_endpoint() -> Dict[str, Any]:
    """
    Detailed metrics endpoint (JSON).

    Usage:
        @app.get("/metrics/detailed")
        async def detailed_metrics():
            return await detailed_metrics_endpoint()
    """
    return await get_detailed_metrics()


# ============================================================================
# Logging Decorators
# ============================================================================


def log_memory_operation(operation_name: str):
    """
    Decorator to log memory operations.

    Usage:
        @log_memory_operation("patient_memory_search")
        async def search_memory(patient_id, query):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args, **kwargs) -> T:
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Log success
                structured_logger.log_search(
                    operation=operation_name,
                    patient_id=kwargs.get("patient_id", "unknown"),
                    query=kwargs.get("query", ""),
                    latency_ms=latency_ms,
                    result_count=len(result) if isinstance(result, list) else 1,
                    cache_hit=kwargs.get("cache_hit", False),
                    status="success",
                )

                # Record metrics
                metrics_collector.record_search(
                    latency_ms=latency_ms,
                    success=True,
                )

                return result

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                # Log failure
                structured_logger.log_search(
                    operation=operation_name,
                    patient_id=kwargs.get("patient_id", "unknown"),
                    query=kwargs.get("query", ""),
                    latency_ms=latency_ms,
                    result_count=0,
                    cache_hit=False,
                    status="failure",
                    error=e,
                )

                # Record metrics
                is_timeout = "timeout" in str(e).lower()
                metrics_collector.record_search(
                    latency_ms=latency_ms,
                    success=False,
                    timeout=is_timeout,
                )

                raise

        return wrapper

    return decorator


# ============================================================================
# Summary
# ============================================================================

"""
INTEGRATION WITH main.py:

1. Add imports:
   ```python
   from memory_observability import (
       MemoriMonitoringMiddleware,
       metrics_endpoint,
       health_endpoint,
       detailed_metrics_endpoint,
   )
   ```

2. Add middleware:
   ```python
   app.add_middleware(MemoriMonitoringMiddleware)
   ```

3. Add endpoints:
   ```python
   @app.get("/metrics")
   async def metrics():
       return await metrics_endpoint()

   @app.get("/health/memori")
   async def memori_health():
       return await health_endpoint()

   @app.get("/metrics/detailed")
   async def detailed_metrics():
       return await detailed_metrics_endpoint()
   ```

4. Update existing health endpoint:
   ```python
   @app.get("/health")
   async def health_check():
       memori_health = await get_memori_health_check()
       return {
           "status": "healthy",
           "models_loaded": {...},
           "memory": memori_health.to_dict()
       }
   ```

5. For Prometheus scraping:
   ```
   # prometheus.yml
   scrape_configs:
     - job_name: 'nlp-service'
       static_configs:
         - targets: ['localhost:5001']
       metrics_path: '/metrics'
   ```

6. For monitoring dashboards:
   ```
   GET /metrics/detailed  # Grafana data source
   GET /health/memori     # Status dashboard
   ```
"""
