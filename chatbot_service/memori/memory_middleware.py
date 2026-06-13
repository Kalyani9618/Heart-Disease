"""
Memory-Aware Middleware and Endpoint Extensions

This module provides:
1. Request context middleware for correlation ID tracking
2. Memory context injection utilities
3. Enhanced endpoint decorators for memory operations
4. Streaming response support with memory updates

Features:
- Automatic correlation ID generation and tracking
- Request-scoped memory context management
- Automatic memory storage of conversations
- Streaming responses with incremental memory updates
- Error handling and graceful degradation

Integration Pattern:
    Request → CorrelationIDMiddleware → set request_id_context
              → @with_memory_context decorator → get patient memory
              → endpoint logic (can use memory directly)
              → Auto-store results before response

Complexity:
- Middleware: O(1) per request
- Memory context injection: O(1) lookup (cached)
- Automatic storage: O(1) async write
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar, Awaitable

from fastapi import Request, HTTPException, status
from fastapi.responses import StreamingResponse

# Import from local memori module
from .memory_manager import (
    MemoryManager,
    PatientMemory,
    MemoryManagerException,
)

logger = logging.getLogger(__name__)

# Context variables for request scope
request_id_context: ContextVar[str] = ContextVar("request_id", default="")
patient_id_context: ContextVar[str] = ContextVar("patient_id", default="")
session_id_context: ContextVar[str] = ContextVar("session_id", default="default")

T = TypeVar("T")


from starlette.middleware.base import BaseHTTPMiddleware

# ============================================================================
# Middleware for Request Context
# ============================================================================


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request correlation IDs for distributed tracing.

    Sets:
    - X-Request-ID header (request-wide correlation ID)
    - request_id_context for automatic injection into logs and memory

    Complexity: O(1) per request
    """

    async def dispatch(self, request: Request, call_next):
        # Get or generate correlation ID
        request_id = request.headers.get(
            "X-Request-ID", request.headers.get("x-request-id", str(uuid.uuid4()))
        )

        # Set context for this request
        request_id_context.set(request_id)

        # Extract patient ID if available (from path, query, or body)
        patient_id = _extract_patient_id(request)
        if patient_id:
            patient_id_context.set(patient_id)

        # Extract session ID if available
        session_id = request.headers.get("X-Session-ID", "default")
        session_id_context.set(session_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Request-Date"] = datetime.now(timezone.utc).isoformat()

        return response


def _extract_patient_id(request: Request) -> Optional[str]:
    """Extract patient ID from request if available."""
    # Try path parameters: /patients/{patient_id}/...
    if "patient_id" in request.path_params:
        return request.path_params["patient_id"]

    # Try query parameters: ?patient_id=...
    if "patient_id" in request.query_params:
        return request.query_params["patient_id"]

    # Try headers: X-Patient-ID
    if "X-Patient-ID" in request.headers:
        return request.headers["X-Patient-ID"]

    return None


# ============================================================================
# Memory Context Management
# ============================================================================


@asynccontextmanager
async def get_memory_context(
    patient_id: str,
    session_id: Optional[str] = None,
):
    """
    Context manager for request-scoped memory access.

    Usage:
        async with get_memory_context(patient_id) as memory:
            context = await memory.get_conversation_context()
            await memory.add_conversation(...)

    Args:
        patient_id: Patient identifier
        session_id: Optional session identifier

    Yields:
        PatientMemory instance (cached, reused)

    Complexity: O(1) for cached access, O(n) for first initialization
    """
    if not patient_id:
        raise ValueError("patient_id is required")

    session_id = session_id or session_id_context.get() or "default"

    try:
        memory_mgr = MemoryManager.get_instance()
        patient_memory = await memory_mgr.get_patient_memory(patient_id, session_id)
        yield patient_memory
    except MemoryManagerException as e:
        logger.warning(f"Memory context unavailable: {e}")
        # Graceful degradation: create a no-op memory context
        yield None
    except Exception as e:
        logger.error(f"Unexpected error in memory context: {e}", exc_info=True)
        yield None


class MemoryContextInjector:
    """
    Dependency injector for FastAPI endpoints.

    Allows endpoints to receive PatientMemory as a dependency:

        @app.post("/nlp/process")
        async def process(
            request: NLPProcessRequest,
            memory: PatientMemory = Depends(get_memory_injector("patient_id"))
        ):
            ...
    """

    def __init__(self, patient_id_param: str = "patient_id"):
        """
        Initialize injector.

        Args:
            patient_id_param: Name of patient ID parameter
        """
        self.patient_id_param = patient_id_param

    async def __call__(self, request: Request) -> Optional[PatientMemory]:
        """Get patient memory instance."""
        # Try to extract patient ID from various sources
        patient_id = request.path_params.get(self.patient_id_param)
        if not patient_id:
            patient_id = request.query_params.get(self.patient_id_param)
        if not patient_id:
            patient_id = request.headers.get("X-Patient-ID")

        if not patient_id:
            return None

        try:
            memory_mgr = MemoryManager.get_instance()
            return await memory_mgr.get_patient_memory(patient_id)
        except Exception as e:
            logger.warning(f"Could not get patient memory: {e}")
            return None


def get_memory_injector(patient_id_param: str = "patient_id") -> MemoryContextInjector:
    """Get memory injector for FastAPI Depends."""
    return MemoryContextInjector(patient_id_param)


# ============================================================================
# Automatic Memory Storage Decorator
# ============================================================================


def with_memory_context(
    memory_type: str = "conversation",
    include_metadata: bool = True,
):
    """
    Decorator to automatically store conversation results to memory.

    Usage:
        @app.post("/nlp/process")
        @with_memory_context(memory_type="conversation")
        async def process(request: NLPProcessRequest) -> NLPProcessResponse:
            # endpoint logic
            return response

    Args:
        memory_type: Type of memory to store (conversation, health_data, etc.)
        include_metadata: Whether to include request metadata

    Stores the response to patient memory automatically before returning.

    Complexity: O(1) async write after response generation
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            # Get patient ID from context or kwargs
            patient_id = patient_id_context.get()
            if not patient_id and "patient_id" in kwargs:
                patient_id = kwargs["patient_id"]
            if not patient_id and args:
                # Try to extract from request object if first arg
                if hasattr(args[0], "patient_id"):
                    patient_id = args[0].patient_id

            # Call the actual endpoint
            result = await func(*args, **kwargs)

            # Auto-store to memory if patient ID available
            if patient_id:
                try:
                    await _store_to_memory(
                        patient_id=patient_id,
                        memory_type=memory_type,
                        content=result,
                        include_metadata=include_metadata,
                    )
                except Exception as e:
                    # Don't fail the endpoint if storage fails
                    logger.warning(f"Failed to store to memory: {e}")

            return result

        return wrapper

    return decorator


async def _store_to_memory(
    patient_id: str,
    memory_type: str,
    content: Any,
    include_metadata: bool = True,
) -> None:
    """
    Store content to patient memory.

    Args:
        patient_id: Patient identifier
        memory_type: Type of memory
        content: Content to store (dict, str, or Pydantic model)
        include_metadata: Whether to include metadata
    """
    try:
        memory_mgr = MemoryManager.get_instance()

        # Serialize content
        if hasattr(content, "dict"):
            # Pydantic model
            content_str = json.dumps(content.dict())
        elif isinstance(content, dict):
            content_str = json.dumps(content)
        else:
            content_str = str(content)

        # Build metadata
        metadata = {}
        if include_metadata:
            metadata = {
                "correlation_id": request_id_context.get(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Store
        session_id = session_id_context.get() or "default"
        await memory_mgr.store_memory(
            patient_id=patient_id,
            memory_type=memory_type,
            content=content_str,
            session_id=session_id,
            metadata=metadata,
        )

    except Exception as e:
        logger.warning(f"Error storing to memory: {e}", exc_info=True)


# ============================================================================
# Streaming Response with Memory Updates
# ============================================================================


class MemoryAwareStreamingResponse(StreamingResponse):
    """
    Streaming response that can update memory as data streams.

    Usage:
        async def stream_generator():
            async for chunk in some_async_generator():
                await memory.add_memory(...)  # Update memory as we stream
                yield json.dumps(chunk).encode() + b"\n"

        return MemoryAwareStreamingResponse(
            stream_generator(),
            patient_id="p123",
            memory_type="conversation"
        )
    """

    def __init__(
        self,
        content,
        patient_id: Optional[str] = None,
        memory_type: str = "conversation",
        **kwargs,
    ):
        super().__init__(content, **kwargs)
        self.patient_id = patient_id or patient_id_context.get()
        self.memory_type = memory_type


# ============================================================================
# Response Models with Memory Context
# ============================================================================


class ResponseWithMemoryContext:
    """
    Base class for responses that include memory context.

    Endpoints can use this to return both results and retrieved context:

        return ResponseWithMemoryContext(
            result=intent_result,
            context=conversation_context,
            memory_used=True
        )
    """

    def __init__(
        self,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
        memory_used: bool = False,
        memory_latency_ms: float = 0.0,
    ):
        self.result = result
        self.context = context or {}
        self.memory_used = memory_used
        self.memory_latency_ms = memory_latency_ms

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response."""
        result_dict = (
            self.result.dict()
            if hasattr(self.result, "dict")
            else (
                self.result
                if isinstance(self.result, dict)
                else {"result": self.result}
            )
        )

        return {
            **result_dict,
            "_memory": {
                "used": self.memory_used,
                "latency_ms": self.memory_latency_ms,
                "context_available": bool(self.context),
            },
            **({"_context": self.context} if self.context else {}),
        }


# ============================================================================
# Enhanced Error Handling with Memory
# ============================================================================


async def handle_endpoint_error(
    patient_id: Optional[str],
    error: Exception,
    operation: str,
    store_to_memory: bool = True,
) -> HTTPException:
    """
    Handle endpoint errors with optional memory logging.

    Args:
        patient_id: Patient identifier
        error: Exception that occurred
        operation: Operation that failed
        store_to_memory: Whether to log error to memory

    Returns:
        HTTPException for response
    """
    logger.error(
        f"Endpoint error: operation={operation}, "
        f"patient_id={patient_id}, error={error}",
        exc_info=True,
    )

    # Optionally store error to memory for audit trail
    if patient_id and store_to_memory:
        try:
            await _store_to_memory(
                patient_id=patient_id,
                memory_type="error_log",
                content={
                    "operation": operation,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                include_metadata=True,
            )
        except Exception as e:
            logger.warning(f"Could not log error to memory: {e}")

    # Return appropriate HTTP exception
    if isinstance(error, MemoryManagerException):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory service temporarily unavailable",
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ============================================================================
# Utility Functions
# ============================================================================


async def fetch_and_merge_context(
    patient_id: str,
    query: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch conversation context and merge with response.

    Helper function for endpoints that want to include memory context
    in their responses.

    Args:
        patient_id: Patient identifier
        query: Search query for context
        session_id: Optional session ID

    Returns:
        Dictionary with context information

    Complexity: O(m) where m = number of memory searches
    """
    try:
        memory_mgr = MemoryManager.get_instance()
        session_id = session_id or session_id_context.get() or "default"

        patient_memory = await memory_mgr.get_patient_memory(patient_id, session_id)

        # Fetch context in parallel
        results = await asyncio.gather(
            patient_memory.get_conversation_context(limit=5),
            patient_memory.get_health_summary(limit=10),
            return_exceptions=True,
        )

        conversation_context = (
            results[0] if not isinstance(results[0], Exception) else {}
        )
        health_summary = results[1] if not isinstance(results[1], Exception) else {}

        return {
            "conversation": conversation_context,
            "health": health_summary,
            "available": True,
        }

    except Exception as e:
        logger.warning(f"Could not fetch context: {e}")
        return {
            "conversation": {},
            "health": {},
            "available": False,
            "error": str(e),
        }


def get_request_id() -> str:
    """Get current request correlation ID."""
    return request_id_context.get()


# Alias for compatibility
get_correlation_id = get_request_id


def get_patient_id() -> Optional[str]:
    """Get current patient ID from context."""
    return patient_id_context.get() or None


def get_session_id() -> str:
    """Get current session ID."""
    return session_id_context.get() or "default"


def get_structured_logger(name: str = "app") -> logging.Logger:
    """
    Get a logger that automatically includes correlation ID in all log entries.
    
    Usage:
        log = get_structured_logger("orchestrator")
        log.info("Processing query", extra={"query": query[:50]})
        
        # Output: 2026-01-10 15:00:00 | orchestrator | INFO | [corr_id=abc123] Processing query | query="What is..."
    
    Returns:
        Logger configured with request context
    """
    class CorrelationIDFilter(logging.Filter):
        """Filter that adds correlation ID to log records."""
        
        def filter(self, record):
            record.correlation_id = request_id_context.get() or "-"
            record.patient_id = patient_id_context.get() or "-"
            record.session_id = session_id_context.get() or "default"
            return True
    
    log = logging.getLogger(name)
    
    # Add filter if not already added
    filter_names = [f.name for f in log.filters if hasattr(f, 'name')]
    if "correlation_id_filter" not in filter_names:
        corr_filter = CorrelationIDFilter()
        corr_filter.name = "correlation_id_filter"
        log.addFilter(corr_filter)
    
    return log


def log_with_context(
    level: str,
    message: str,
    logger_name: str = "app",
    **extra_fields
) -> None:
    """
    Log a message with automatic correlation ID injection.
    
    Usage:
        log_with_context("info", "Processing query", query_hash="abc123")
        log_with_context("error", "Query failed", error=str(e))
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        logger_name: Logger name
        **extra_fields: Additional fields to include
    """
    log = get_structured_logger(logger_name)
    
    # Build context-aware message
    context = {
        "correlation_id": request_id_context.get() or "-",
        "patient_id": patient_id_context.get() or "-",
        **extra_fields
    }
    
    context_str = " | ".join(f"{k}={v}" for k, v in context.items() if v != "-")
    full_message = f"[{context_str}] {message}" if context_str else message
    
    log_method = getattr(log, level.lower(), log.info)
    log_method(full_message)

