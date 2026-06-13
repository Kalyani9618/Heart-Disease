"""
Agent Tracing - Observability for agent operations.

Provides:
- AgentTracer: Unified tracing for all operations
- Span context management
- Integration with Langfuse/OpenTelemetry
- Local fallback logging

Based on monitoring-and-evaluating-agents.ipynb patterns.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import json
import logging
import uuid
import threading
from collections import deque

logger = logging.getLogger(__name__)


class SpanType(Enum):
    """Types of traced operations."""
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_STEP = "agent_step"
    RAG_RETRIEVAL = "rag_retrieval"
    DB_QUERY = "db_query"
    WEB_REQUEST = "web_request"
    PLANNING = "planning"
    VALIDATION = "validation"
    CUSTOM = "custom"


class SpanStatus(Enum):
    """Status of a span."""
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class Span:
    """
    A single traced operation.
    
    Attributes:
        span_id: Unique span identifier
        name: Operation name
        span_type: Type of operation
        start_time: When operation started
        end_time: When operation ended (if finished)
        status: Current status
        metadata: Additional context
        parent_id: Parent span ID (for nested operations)
        events: List of events/logs within span
    """
    span_id: str
    name: str
    span_type: SpanType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.RUNNING
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def add_event(self, name: str, data: Optional[Dict] = None):
        """Add an event to this span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "type": self.span_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms(),
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "events": self.events,
            "error": self.error
        }


@dataclass
class Trace:
    """
    A complete trace consisting of multiple spans.
    
    Represents a full operation from start to finish.
    """
    trace_id: str
    name: str
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    def add_span(self, span: Span):
        """Add a span to this trace."""
        self.spans.append(span)
    
    def get_root_span(self) -> Optional[Span]:
        """Get the root span (no parent)."""
        for span in self.spans:
            if span.parent_id is None:
                return span
        return self.spans[0] if self.spans else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


class AgentTracer:
    """
    Unified tracing for all agent operations.
    
    Features:
    - Automatic span management
    - Nested span support
    - Multiple backend support (Langfuse, local)
    - Metrics aggregation
    
    Usage:
        tracer = AgentTracer()
        
        async with tracer.trace_operation(
            name="process_query",
            operation_type=SpanType.AGENT_STEP,
            metadata={"user_id": "123"}
        ) as span:
            # Do work
            result = await process(query)
            span.add_event("processed", {"tokens": 100})
    """
    
    def __init__(
        self,
        backend: str = "local",
        max_traces: int = 1000,
        service_name: str = "cardio-ai-agent"
    ):
        """
        Initialize the tracer.
        
        Args:
            backend: "langfuse", "opentelemetry", or "local"
            max_traces: Maximum traces to keep in memory
            service_name: Service name for external backends
        """
        self.backend = backend
        self.service_name = service_name
        self.traces: deque = deque(maxlen=max_traces)
        self._current_trace: Optional[Trace] = None
        self._current_span_id: Optional[str] = None
        self._lock = threading.Lock()
        
        # Setup external backend
        self._client = None
        self._setup_backend()
    
    def _setup_backend(self):
        """Initialize external tracing backend."""
        if self.backend == "langfuse":
            try:
                import os
                from langfuse import Langfuse
                
                # Check if Langfuse is explicitly enabled
                langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
                
                if not langfuse_enabled:
                    logger.info("⚠️ Langfuse disabled (LANGFUSE_ENABLED not set to 'true')")
                    self.backend = "local"
                    return
                
                # Get credentials from environment
                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                
                if not public_key or not secret_key:
                    logger.warning("❌ Langfuse credentials missing (LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set)")
                    self.backend = "local"
                    return
                
                # Initialize with credentials
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                logger.info(f"✅ Langfuse tracing enabled (host: {host})")
            except ImportError:
                logger.warning("Langfuse not installed, falling back to local tracing")
                self.backend = "local"
            except Exception as e:
                logger.error(f"❌ Failed to initialize Langfuse: {e}")
                self.backend = "local"
        elif self.backend == "opentelemetry":
            try:
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                trace.set_tracer_provider(TracerProvider())
                self._client = trace.get_tracer(self.service_name)
                logger.info("OpenTelemetry tracing enabled")
            except ImportError:
                logger.warning("OpenTelemetry not installed, falling back to local")
                self.backend = "local"
    
    def start_trace(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """
        Start a new trace.
        
        Args:
            name: Trace name
            metadata: Additional metadata
            
        Returns:
            New Trace object
        """
        trace = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._current_trace = trace
        
        return trace
    
    def end_trace(self, trace: Optional[Trace] = None):
        """End a trace and record it."""
        trace = trace or self._current_trace
        if trace:
            trace.end_time = datetime.utcnow()
            self.traces.append(trace)
            self._record_to_backend(trace)
            
            with self._lock:
                if self._current_trace == trace:
                    self._current_trace = None
    
    @asynccontextmanager
    async def trace_operation(
        self,
        name: str,
        operation_type: SpanType = SpanType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing an operation.
        
        Args:
            name: Operation name
            operation_type: Type of operation
            metadata: Additional context
            
        Yields:
            Span object for adding events
        """
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            span_type=operation_type,
            start_time=datetime.utcnow(),
            metadata=metadata or {},
            parent_id=self._current_span_id
        )
        
        # Set as current span
        previous_span_id = self._current_span_id
        self._current_span_id = span.span_id
        
        # Add to current trace
        if self._current_trace:
            self._current_trace.add_span(span)
        
        try:
            yield span
            span.status = SpanStatus.SUCCESS
        except Exception as e:
            span.status = SpanStatus.ERROR
            span.error = str(e)
            raise
        finally:
            span.end_time = datetime.utcnow()
            self._current_span_id = previous_span_id
            
            # Log locally
            self._log_span(span)
    
    def record_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None
    ):
        """Record an LLM call as a span."""
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=f"llm_call:{model}",
            span_type=SpanType.LLM_CALL,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            status=SpanStatus.SUCCESS,
            metadata={
                "model": model,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "tokens_used": tokens_used,
                "latency_ms": latency_ms
            }
        )
        
        if self._current_trace:
            self._current_trace.add_span(span)
        
        self._log_span(span)
    
    def record_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        success: bool,
        latency_ms: float
    ):
        """Record a tool call as a span."""
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=f"tool:{tool_name}",
            span_type=SpanType.TOOL_CALL,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            status=SpanStatus.SUCCESS if success else SpanStatus.ERROR,
            metadata={
                "tool_name": tool_name,
                "args": args,
                "success": success,
                "latency_ms": latency_ms
            }
        )
        
        if self._current_trace:
            self._current_trace.add_span(span)
        
        self._log_span(span)
    
    def _log_span(self, span: Span):
        """Log span to local logger."""
        duration = span.duration_ms() or 0
        status = "✅" if span.status == SpanStatus.SUCCESS else "❌"
        
        logger.info(
            f"TRACE {status} [{span.span_type.value}] {span.name} "
            f"({duration:.1f}ms) {json.dumps(span.metadata)}"
        )
    
    def _record_to_backend(self, trace: Trace):
        """Record trace to external backend."""
        if self._client and self.backend == "langfuse":
            try:
                self._client.trace(
                    name=trace.name,
                    id=trace.trace_id,
                    metadata=trace.to_dict()
                )
            except Exception as e:
                logger.error(f"Failed to record to Langfuse: {e}")
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces as dictionaries."""
        traces = list(self.traces)[-limit:]
        return [t.to_dict() for t in traces]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from traces."""
        if not self.traces:
            return {"total_traces": 0}
        
        total_spans = sum(len(t.spans) for t in self.traces)
        success_spans = sum(
            1 for t in self.traces 
            for s in t.spans 
            if s.status == SpanStatus.SUCCESS
        )
        
        # Latency stats
        latencies = [
            s.duration_ms() for t in self.traces 
            for s in t.spans 
            if s.duration_ms() is not None
        ]
        
        return {
            "total_traces": len(self.traces),
            "total_spans": total_spans,
            "success_rate": (success_spans / total_spans * 100) if total_spans else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "span_types": self._count_span_types()
        }
    
    def _count_span_types(self) -> Dict[str, int]:
        """Count spans by type."""
        counts = {}
        for trace in self.traces:
            for span in trace.spans:
                key = span.span_type.value
                counts[key] = counts.get(key, 0) + 1
        return counts


# Global tracer instance
_tracer: Optional[AgentTracer] = None


def get_tracer() -> AgentTracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = AgentTracer()
    return _tracer


def init_tracer(backend: str = "local", **kwargs) -> AgentTracer:
    """Initialize global tracer with specific backend."""
    global _tracer
    _tracer = AgentTracer(backend=backend, **kwargs)
    return _tracer
