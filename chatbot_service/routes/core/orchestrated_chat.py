"""
Orchestrated Chat Routes - Async Job Pattern

This module provides chat endpoints that leverage an async worker pattern:
- API returns immediately with a job_id (<100ms response time)
- Heavy processing happens in ARQ workers
- Real-time updates via WebSocket, SSE, or webhooks

Migration from sync to async pattern enables:
- 10k+ concurrent users
- Horizontal scaling of workers
- Fault tolerance with automatic retries
- HIPAA-compliant audit trail
"""


import logging
import uuid
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status, Request
from pydantic import BaseModel, Field, validator
import re

# PERFORMANCE: Lazy import - LangGraphOrchestrator has heavy dependencies
# that slow down route loading. Import moved inside get_orchestrator().
if TYPE_CHECKING:
    from agents.langgraph_orchestrator import LangGraphOrchestrator
    from arq import ArqRedis

from core.security import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Orchestrated Chat"])

# Global orchestrator instance (lazy loaded)
_orchestrator: Optional["LangGraphOrchestrator"] = None

# ARQ pool reference - set during app lifespan startup
_arq_pool: Optional["ArqRedis"] = None


def set_arq_pool(pool: "ArqRedis") -> None:
    """Set the ARQ pool for job enqueueing (called from app_lifespan)."""
    global _arq_pool
    _arq_pool = pool
    logger.info("✅ ARQ pool configured for orchestrated chat routes")


def get_arq_pool() -> "ArqRedis":
    """Get the ARQ pool, raising an error if not configured."""
    if _arq_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Job queue not available. Service is starting up."
        )
    return _arq_pool

def reset_orchestrator() -> None:
    """Reset the orchestrator singleton to force re-initialization."""
    global _orchestrator
    _orchestrator = None
    logger.info("🔄 Orchestrator reset - will re-initialize on next request")

def get_orchestrator() -> "LangGraphOrchestrator":
    """Get or initialize the orchestrator singleton (lazy loaded)."""
    global _orchestrator
    if _orchestrator is None:
        try:
            # Try to use the orchestrator initialized in app_lifespan first
            from app_lifespan import get_orchestrator as get_lifespan_orchestrator
            lifespan_orch = get_lifespan_orchestrator()
            if lifespan_orch is not None:
                _orchestrator = lifespan_orch
                logger.info("✅ Using orchestrator from app_lifespan")
                return _orchestrator
        except Exception:
            pass
        
        try:
            # LAZY IMPORT: Only load heavy dependencies when first request comes in
            from agents.langgraph_orchestrator import LangGraphOrchestrator
            logger.info("Initializing LangGraph Orchestrator for the first time...")
            _orchestrator = LangGraphOrchestrator()
            logger.info("✅ LangGraph Orchestrator initialized successfully")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize orchestrator: {e}")
            raise HTTPException(
                status_code=503,
                detail="Chat service is currently unavailable. Please try again later."
            )
    return _orchestrator

# --- Data Models ---

class ChatRequest(BaseModel):
    """Request model for chat messages."""
    user_id: str = Field(..., min_length=1, max_length=100, description="User ID")
    message: str = Field(..., min_length=1, max_length=10000, description="User message (max 10KB)")
    session_id: Optional[str] = Field(None, max_length=100, description="Session ID")
    
    # Async pattern options
    sync: bool = Field(False, description="If True, wait for result (backward compat). Default: async")
    webhook_url: Optional[str] = Field(None, description="URL for result delivery via webhook")
    priority: int = Field(0, ge=-10, le=10, description="Job priority (-10 to 10, higher = more urgent)")
    
    # Feature flags
    thinking: bool = Field(False, description="Enable thinking agent")
    web_search: bool = Field(False, description="Enable web search")
    deep_search: bool = Field(False, description="Enable deep search")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs to process")
    
    @validator('message')
    def sanitize_message(cls, v):
        """Remove potentially dangerous control characters."""
        # Remove control characters except newlines and tabs
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        """Validate webhook URL if provided."""
        if v is not None:
            if not v.startswith('https://'):
                raise ValueError("webhook_url must be a valid HTTPS URL")
            # Block internal/private IP targets
            from urllib.parse import urlparse
            host = urlparse(v).hostname or ''
            blocked_prefixes = ['localhost', '127.', '0.0.0.0', '::1', '169.254.', '10.', '192.168.']
            blocked_ranges = ['172.16.', '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.']
            all_blocked = blocked_prefixes + blocked_ranges
            if any(host.startswith(b) or host == b for b in all_blocked):
                raise ValueError("webhook_url cannot target internal/private addresses")
        return v


class JobAcceptedResponse(BaseModel):
    """Response model for async job acceptance."""
    job_id: str
    status: str = "accepted"
    message: str = "Job queued for processing"
    estimated_wait_seconds: Optional[int] = None
    poll_url: str
    websocket_url: str
    sse_url: str

class ChatResponse(BaseModel):
    """Response model for chat messages."""
    response: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    session_id: str
    success: bool = True

class ResearchRequest(BaseModel):
    """Request model for deep research."""
    query: str = Field(..., min_length=5, max_length=500)
    user_id: str
    
    # Async pattern options
    webhook_url: Optional[str] = Field(None, description="URL for result delivery via webhook")
    priority: int = Field(0, ge=-10, le=10, description="Job priority (-10 to 10, higher = more urgent)")
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        """Validate webhook URL if provided."""
        if v is not None:
            if not v.startswith(('http://', 'https://')):
                raise ValueError("webhook_url must be a valid HTTP/HTTPS URL")
        return v


class ResearchAcceptedResponse(BaseModel):
    """Response model for async research job acceptance."""
    job_id: str
    status: str = "accepted"
    message: str = "Deep research job queued for processing"
    estimated_wait_seconds: Optional[int] = None
    poll_url: str
    websocket_url: str
    sse_url: str

# --- Routes ---

@router.post("/message", response_model=None)
async def orchestrated_chat(
    request: ChatRequest,
    fastapi_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Process a chat message through the LangGraph agentic workflow.
    
    Default behavior (async):
    - Returns immediately with job_id (<100ms)
    - Processing happens in background workers
    - Get results via WebSocket, SSE, polling, or webhook
    
    Sync mode (request.sync=True):
    - Blocks until result is ready
    - For backward compatibility only
    - Not recommended for production
    
    Requires Authentication.
    """
    # 1. Security: Validate User ID matches Authenticated User
    if str(request.user_id) != str(current_user.get("user_id")):
        logger.warning("Auth mismatch: Request user does not match token user")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only post messages for your own user ID."
        )
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # 2. Check if sync mode requested (backward compatibility)
    if request.sync:
        return await _process_sync_chat(request, session_id)
    
    # 3. Async mode (default): Enqueue job and return immediately
    return await _enqueue_chat_job(request, session_id, fastapi_request)


async def _process_sync_chat(request: ChatRequest, session_id: str) -> ChatResponse:
    """
    Process chat synchronously (blocking) - for backward compatibility.
    
    WARNING: This blocks the API worker. Use only for testing or low-traffic scenarios.
    """
    orchestrator = get_orchestrator()
    
    try:
        logger.info("[SYNC] Processing message")
        
        result = await orchestrator.execute(
            query=request.message,
            user_id=request.user_id,
            thinking=request.thinking,
            web_search=request.web_search,
            deep_search=request.deep_search,
            file_ids=request.file_ids
        )
        
        is_success = bool(result.get("response")) and result.get("confidence", 0) > 0.3
        
        return ChatResponse(
            response=result.get("response", "I apologize, but I couldn't generate a response."),
            sources=result.get("sources", []),
            metadata={
                "processing_time": result.get("processing_time"),
                "steps": result.get("steps", []),
                "confidence": result.get("confidence", 0.0),
                "source": result.get("metadata", {}).get("source", "unknown"),
                "model": result.get("metadata", {}).get("model", "unknown"),
                "intent": result.get("intent", "unknown"),
                "pii_scrubbed": result.get("pii_scrubbed", False),
                "sync_mode": True  # Indicate this was processed synchronously
            },
            session_id=session_id,
            success=is_success
        )
        
    except Exception as e:
        logger.error(f"[SYNC] Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _enqueue_chat_job(
    request: ChatRequest,
    session_id: str,
    fastapi_request: Request
) -> JobAcceptedResponse:
    """
    Enqueue chat job for async processing and return immediately.
    
    This is the scalable pattern:
    - API returns in <100ms
    - Heavy processing in ARQ workers
    - Real-time updates via WebSocket/SSE
    """
    from core.dependencies import DIContainer
    
    pool = get_arq_pool()
    
    # Get JobStore from DIContainer (initialized at startup)
    container = DIContainer.get_instance()
    job_store = container.get_service('job_store')
    if not job_store:
        raise HTTPException(status_code=503, detail="Job store not available")
    
    # Generate unique job ID
    job_id = f"chat_{uuid.uuid4().hex[:16]}"
    
    try:
        # 1. Create job record in Redis
        # Note: JobStore.create_job() generates its own job_id internally,
        # so we pass metadata to track additional info
        job = await job_store.create_job(
            user_id=str(request.user_id),
            query=request.message,
            session_id=session_id,
            priority=request.priority if request.priority is not None else "normal",
            metadata={
                "job_type": "chat",
                "thinking": request.thinking,
                "web_search": request.web_search,
                "deep_search": request.deep_search,
                "file_ids": request.file_ids,
                "webhook_url": request.webhook_url,
            }
        )
        job_id = job.id  # Use the job ID generated by JobStore
        
        # 2. Enqueue to ARQ worker queue
        await pool.enqueue_job(
            "process_chat_message",
            job_id=job_id,
            user_id=request.user_id,
            message=request.message,
            session_id=session_id,
            thinking=request.thinking,
            web_search=request.web_search,
            deep_search=request.deep_search,
            file_ids=request.file_ids,
            webhook_url=request.webhook_url,
            _job_id=job_id  # Use our job_id as ARQ's job ID for tracking
        )
        
        logger.info(f"[ASYNC] Job {job_id} enqueued")
        
        # 3. Build response URLs
        base_url = str(fastapi_request.base_url).rstrip('/')
        
        return JobAcceptedResponse(
            job_id=job_id,
            status="accepted",
            message="Chat job queued for processing",
            estimated_wait_seconds=5,  # TODO: Estimate based on queue depth
            poll_url=f"{base_url}/api/v2/jobs/{job_id}",
            websocket_url=f"ws://{fastapi_request.url.netloc}/ws/jobs/{job_id}",
            sse_url=f"{base_url}/sse/jobs/{job_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}", exc_info=True)
        
        # Cleanup job record if enqueue failed
        try:
            await job_store.fail_job(job_id, str(e))
        except Exception:
            pass
        
        raise HTTPException(
            status_code=503,
            detail="Failed to queue job. Please try again."
        )

@router.post("/research", response_model=ResearchAcceptedResponse)
async def deep_research(
    request: ResearchRequest,
    fastapi_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ResearchAcceptedResponse:
    """
    Trigger a deep research task (Async).
    
    Deep research typically takes 30-120 seconds. This endpoint:
    - Returns immediately with job_id
    - Processing happens in dedicated worker
    - Get results via WebSocket, SSE, polling, or webhook
    """
    # Validate User
    if str(request.user_id) != str(current_user.get("user_id")):
        raise HTTPException(status_code=403, detail="Access denied")
    
    from core.dependencies import DIContainer
    
    pool = get_arq_pool()
    
    # Get JobStore from DIContainer (initialized at startup)
    container = DIContainer.get_instance()
    job_store = container.get_service('job_store')
    if not job_store:
        raise HTTPException(status_code=503, detail="Job store not available")
    
    # Generate unique job ID
    job_id = f"research_{uuid.uuid4().hex[:16]}"
    
    try:
        # 1. Create job record in Redis
        job = await job_store.create_job(
            user_id=str(request.user_id),
            query=request.query,
            priority=request.priority if request.priority is not None else "normal",
            metadata={
                "job_type": "deep_research",
                "webhook_url": request.webhook_url,
            }
        )
        job_id = job.id  # Use the job ID generated by JobStore
        
        # 2. Enqueue to ARQ worker queue
        await pool.enqueue_job(
            "process_deep_research",
            job_id=job_id,
            user_id=request.user_id,
            query=request.query,
            webhook_url=request.webhook_url,
            _job_id=job_id
        )
        
        logger.info(f"[ASYNC] Research job {job_id} enqueued")
        
        # 3. Build response URLs
        base_url = str(fastapi_request.base_url).rstrip('/')
        
        return ResearchAcceptedResponse(
            job_id=job_id,
            status="accepted",
            message="Deep research job queued for processing",
            estimated_wait_seconds=60,  # Research typically takes longer
            poll_url=f"{base_url}/api/v2/jobs/{job_id}",
            websocket_url=f"ws://{fastapi_request.url.netloc}/ws/jobs/{job_id}",
            sse_url=f"{base_url}/sse/jobs/{job_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue research job: {e}", exc_info=True)
        
        # Cleanup job record if enqueue failed
        try:
            await job_store.fail_job(job_id, str(e))
        except Exception:
            pass
        
        raise HTTPException(
            status_code=503,
            detail="Failed to queue research job. Please try again."
        )

@router.get("/health")
async def health_check():
    """Check if the orchestrator and job queue are ready."""
    global _orchestrator, _arq_pool
    
    orchestrator_status = "healthy" if _orchestrator else "uninitialized"
    arq_status = "healthy" if _arq_pool else "uninitialized"
    
    # Overall health
    if _arq_pool:
        overall = "healthy"
    elif _orchestrator:
        overall = "degraded"  # Can do sync but not async
    else:
        overall = "uninitialized"
    
    return {
        "status": overall, 
        "service": "langgraph_orchestrator",
        "components": {
            "orchestrator": orchestrator_status,
            "job_queue": arq_status
        },
        "mode": "async" if _arq_pool else "sync"
    }

@router.post("/reset")
async def reset_orchestrator_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Reset the orchestrator to force re-initialization with updated code."""
    reset_orchestrator()
    return {"status": "reset", "message": "Orchestrator will re-initialize on next request"}


# --- Queue Statistics Endpoint ---

@router.get("/queue/stats")
async def queue_stats():
    """
    Get current queue statistics.
    
    Useful for monitoring and auto-scaling decisions.
    """
    from core.services.job_store import JobStatus
    from core.dependencies import DIContainer
    
    try:
        pool = get_arq_pool()
        
        # Get JobStore from DIContainer (initialized at startup)
        container = DIContainer.get_instance()
        job_store = container.get_service('job_store')
        if not job_store:
            return {
                "status": "error",
                "error": "Job store not available",
                "stats": None
            }
        
        stats = await job_store.get_queue_stats()
        
        return {
            "status": "ok",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "stats": None
        }