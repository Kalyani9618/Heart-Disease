"""
Performance Monitor

Real implementation backed by PrometheusMetrics.
Records reranking, embedding, and cache operation metrics.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded metrics singleton to avoid circular imports
_metrics = None


def _get_metrics():
    """Lazy-load PrometheusMetrics singleton."""
    global _metrics
    if _metrics is None:
        try:
            from core.monitoring.prometheus_metrics import get_metrics
            _metrics = get_metrics()
        except Exception as e:
            logger.warning(f"PrometheusMetrics unavailable, falling back to logging: {e}")
    return _metrics


def record_rerank_operation(elapsed_ms: float):
    """Record rerank operation duration to Prometheus metrics."""
    metrics = _get_metrics()
    if metrics:
        metrics.record_histogram("rag_rerank_duration_ms", elapsed_ms)
        metrics.increment_counter("rag_rerank_operations")
    logger.debug(f"Rerank operation: {elapsed_ms:.1f}ms")


def record_embedding_operation(elapsed_ms: float):
    """Record embedding operation duration to Prometheus metrics."""
    metrics = _get_metrics()
    if metrics:
        metrics.record_histogram("rag_embedding_duration_ms", elapsed_ms)
        metrics.increment_counter("rag_embedding_operations")
    logger.debug(f"Embedding operation: {elapsed_ms:.1f}ms")


def record_cache_operation(operation: str, hit: bool = False, latency_ms: float = 0.0):
    """Record cache operation metrics to Prometheus metrics."""
    metrics = _get_metrics()
    if metrics:
        if hit:
            metrics.record_memory_hit()
        else:
            metrics.record_memory_miss()
        metrics.record_histogram("rag_cache_operation_duration_ms", latency_ms,
                                  labels={"operation": operation})
    logger.debug(f"Cache {operation}: hit={hit}, latency={latency_ms:.1f}ms")


class CacheTimer:
    """Context manager for timing cache operations and recording to Prometheus."""

    def __init__(self, operation: str = "get"):
        self.operation = operation
        self._start: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000
            record_cache_operation(
                operation=self.operation,
                hit=exc_type is None,
                latency_ms=self.elapsed_ms,
            )