"""
Graceful Degradation Utilities

This module provides utilities for building fault-tolerant services
that continue operating even when dependencies fail.

Key patterns:
1. Tiered fallback execution
2. Partial response construction
3. Service status tracking


Usage:
    from core.graceful_degradation import fallback_chain, with_fallback

    @with_fallback(default=[])
    async def get_documents(query: str):
        return await vector_store.search(query)

    # Or use fallback_chain for multiple fallbacks:
    result = await fallback_chain(
        primary=lambda: postgres_db.query(drugs),
        secondary=lambda: json_fallback.query(drugs),
        tertiary=lambda: [],
        service_name="drug_interactions"
    )
"""

import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar, List, Dict
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Fallback Decorators
# ============================================================================


def with_fallback(
    default: Any = None,
    on_error: Optional[Callable[[Exception], Any]] = None,
    log_errors: bool = True,
):
    """
    Decorator that returns a default value if the function fails.

    Usage:
        @with_fallback(default=[], log_errors=True)
        async def get_documents():
            return await vector_store.search(query)

    Args:
        default: Value to return on failure
        on_error: Optional callback to transform error to result
        log_errors: Whether to log errors

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                if log_errors:
                    logger.warning(f"Graceful degradation in {func.__name__}: {e}")
                if on_error:
                    return on_error(e)
                return default

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"Graceful degradation in {func.__name__}: {e}")
                if on_error:
                    return on_error(e)
                return default

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def fallback_chain(
    primary: Callable[[], T],
    secondary: Optional[Callable[[], T]] = None,
    tertiary: Optional[Callable[[], T]] = None,
    default: T = None,
    service_name: str = "unknown",
    timeout_seconds: float = 10.0,
) -> T:
    """
    Execute functions in a fallback chain until one succeeds.

    Usage:
        result = await fallback_chain(
            primary=lambda: postgres_db.query(drugs),
            secondary=lambda: json_fallback.query(drugs),
            tertiary=lambda: [],
            service_name="drug_interactions"
        )

    Args:
        primary: Primary function to try first
        secondary: Secondary fallback function
        tertiary: Tertiary fallback function
        default: Default value if all fail
        service_name: Name for logging
        timeout_seconds: Timeout for each attempt

    Returns:
        Result from first successful function or default
    """
    functions = [f for f in [primary, secondary, tertiary] if f is not None]

    for i, func in enumerate(functions):
        tier = ["primary", "secondary", "tertiary"][i]
        try:
            result = func()

            # Handle async functions
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=timeout_seconds)

            logger.debug(f"{service_name}: {tier} succeeded")
            return result

        except asyncio.TimeoutError:
            logger.warning(f"{service_name}: {tier} timed out after {timeout_seconds}s")
        except Exception as e:
            logger.warning(f"{service_name}: {tier} failed with {type(e).__name__}: {e}")

    logger.error(f"{service_name}: all fallbacks exhausted, returning default")
    return default


# ============================================================================
# Service Status Tracking
# ============================================================================


class ServiceStatusTracker:
    """
    Track health status of multiple services for graceful degradation decisions.

    Usage:
        tracker = ServiceStatusTracker()
        tracker.record_success("postgres")
        tracker.record_failure("postgres", "Connection refused")

        status = tracker.get_status("postgres")
        if status["healthy"]:
            # Use postgres
        else:
            # Use fallback
    """

    def __init__(self):
        self._services: Dict[str, Dict[str, Any]] = {}
        self._failure_threshold = 3
        self._recovery_window_seconds = 60

    def record_success(self, service_name: str) -> None:
        """Record a successful service call."""
        if service_name not in self._services:
            self._services[service_name] = self._init_service_state()

        state = self._services[service_name]
        state["consecutive_failures"] = 0
        state["last_success"] = datetime.utcnow()
        state["healthy"] = True

    def record_failure(self, service_name: str, error: str = "") -> None:
        """Record a failed service call."""
        if service_name not in self._services:
            self._services[service_name] = self._init_service_state()

        state = self._services[service_name]
        state["consecutive_failures"] += 1
        state["last_failure"] = datetime.utcnow()
        state["last_error"] = error

        if state["consecutive_failures"] >= self._failure_threshold:
            state["healthy"] = False
            logger.warning(
                f"Service {service_name} marked unhealthy after "
                f"{state['consecutive_failures']} failures"
            )

    def get_status(self, service_name: str) -> Dict[str, Any]:
        """Get current status of a service."""
        if service_name not in self._services:
            return {"healthy": True, "unknown": True}

        state = self._services[service_name]

        # Check if enough time has passed for recovery attempt
        if not state["healthy"] and state.get("last_failure"):
            elapsed = (datetime.utcnow() - state["last_failure"]).total_seconds()
            if elapsed >= self._recovery_window_seconds:
                state["healthy"] = True  # Allow recovery attempt
                logger.info(f"Service {service_name} recovery window elapsed, allowing retry")

        return state

    def is_healthy(self, service_name: str) -> bool:
        """Quick check if service is healthy."""
        return self.get_status(service_name).get("healthy", True)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tracked services."""
        return {name: self.get_status(name) for name in self._services}

    def _init_service_state(self) -> Dict[str, Any]:
        """Initialize state for a new service."""
        return {
            "healthy": True,
            "consecutive_failures": 0,
            "last_success": None,
            "last_failure": None,
            "last_error": "",
        }


# Global service tracker instance
_service_tracker: Optional[ServiceStatusTracker] = None


def get_service_tracker() -> ServiceStatusTracker:
    """Get global service status tracker."""
    global _service_tracker
    if _service_tracker is None:
        _service_tracker = ServiceStatusTracker()
    return _service_tracker


# ============================================================================
# Partial Response Builder
# ============================================================================


class PartialResponseBuilder:
    """
    Build responses that may have missing components due to service failures.

    Usage:
        builder = PartialResponseBuilder()
        builder.add_component("documents", documents, source="vector_store")
        builder.add_component("interactions", interactions, source="postgresql")
        builder.add_failed_component("memories", "redis unavailable")

        response = builder.build()
        # Returns: {
        #     "documents": [...],
        #     "interactions": [...],
        #     "_partial": True,
        #     "_missing": ["memories"],
        #     "_sources": {"documents": "vector_store", ...}
        # }
    """

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._sources: Dict[str, str] = {}
        self._missing: List[str] = []
        self._errors: Dict[str, str] = {}

    def add_component(
        self,
        name: str,
        value: Any,
        source: str = "unknown",
    ) -> "PartialResponseBuilder":
        """Add a successful component to the response."""
        self._components[name] = value
        self._sources[name] = source
        return self

    def add_failed_component(
        self,
        name: str,
        error: str = "unavailable",
    ) -> "PartialResponseBuilder":
        """Record a component that failed to load."""
        self._missing.append(name)
        self._errors[name] = error
        return self

    def build(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Build the final response dictionary."""
        response = dict(self._components)

        if include_metadata:
            response["_partial"] = len(self._missing) > 0
            if self._missing:
                response["_missing"] = self._missing
            if self._sources:
                response["_sources"] = self._sources

        return response

    @property
    def is_complete(self) -> bool:
        """Check if all components loaded successfully."""
        return len(self._missing) == 0
