"""
Server-Sent Events (SSE) Routes for Real-Time Updates

SSE is simpler than WebSocket and works through proxies and load balancers.
Good choice for clients that only need to receive updates (one-way).

Advantages over WebSocket:
- Simpler client-side code (just EventSource)
- Works with HTTP/2
- Automatic reconnection in browsers
- No need for ping/pong

Usage:
    GET /sse/job/{job_id} - Stream updates for a single job
    GET /sse/user/{user_id} - Stream updates for all user's jobs
"""


import logging
import json
import asyncio
from typing import Optional, AsyncGenerator
from datetime import datetime
from fastapi import APIRouter, Request, Depends, Query, HTTPException, status
from fastapi.responses import StreamingResponse

from core.security import get_current_user
from core.services.job_store import get_job_store, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Server-Sent Events"])


# ============================================================================
# SSE Response Helpers
# ============================================================================

def format_sse_message(
    event: str,
    data: dict,
    event_id: Optional[str] = None
) -> str:
    """
    Format a message for SSE.
    
    SSE format:
        event: <event_type>
        id: <event_id>
        data: <json_data>
        
        (blank line to end message)
    
    Args:
        event: Event type (e.g., "progress", "result", "heartbeat")
        data: Data dictionary to send as JSON
        event_id: Optional event ID for client-side tracking
    
    Returns:
        Formatted SSE message string
    """
    lines = []
    
    if event:
        lines.append(f"event: {event}")
    
    if event_id:
        lines.append(f"id: {event_id}")
    
    # Data must be single line, so we JSON encode
    json_data = json.dumps(data)
    lines.append(f"data: {json_data}")
    
    # SSE messages end with double newline
    lines.append("")
    lines.append("")
    
    return "\n".join(lines)


def format_sse_comment(comment: str) -> str:
    """Format an SSE comment (keep-alive)."""
    return f": {comment}\n\n"


# ============================================================================
# SSE Generators
# ============================================================================

async def job_event_generator(
    request: Request,
    job_id: str,
    user_id: str,
    timeout: float = 300
) -> AsyncGenerator[str, None]:
    """
    Generator that yields SSE events for a job.
    
    Polls job status and yields:
    - progress: When job progress updates
    - heartbeat: Every 15 seconds to keep connection alive
    - result: When job completes or fails
    
    Args:
        request: FastAPI request (for disconnect detection)
        job_id: Job ID to monitor
        user_id: User ID for verification
        timeout: Maximum time to stream (seconds)
    
    Yields:
        SSE formatted messages
    """
    job_store = await get_job_store()
    start_time = asyncio.get_event_loop().time()
    last_progress = None
    event_counter = 0
    
    # Send initial connection event
    yield format_sse_message("connected", {
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat()
    }, event_id=str(event_counter))
    event_counter += 1
    
    try:
        while True:
            # Check for client disconnect
            if await request.is_disconnected():
                logger.info(f"SSE client disconnected: job={job_id}")
                break
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                yield format_sse_message("timeout", {
                    "message": "Stream timeout reached",
                    "elapsed_seconds": int(elapsed)
                }, event_id=str(event_counter))
                break
            
            # Get current job status
            job = await job_store.get_job(job_id)
            
            if not job:
                yield format_sse_message("error", {
                    "message": "Job not found"
                }, event_id=str(event_counter))
                break
            
            # Verify ownership
            if str(job.user_id) != str(user_id):
                yield format_sse_message("error", {
                    "message": "Access denied"
                }, event_id=str(event_counter))
                break
            
            # Check for progress update
            if job.progress and job.progress != last_progress:
                yield format_sse_message("progress", {
                    "job_id": job_id,
                    "status": job.status,
                    **job.progress
                }, event_id=str(event_counter))
                event_counter += 1
                last_progress = dict(job.progress)  # Deep copy to avoid identity comparison
            
            # Check for completion
            if job.status == JobStatus.COMPLETED.value:
                result = await job_store.get_job_result(job_id)
                yield format_sse_message("result", {
                    "job_id": job_id,
                    "status": "completed",
                    **(result or {})
                }, event_id=str(event_counter))
                break
            
            elif job.status == JobStatus.FAILED.value:
                yield format_sse_message("result", {
                    "job_id": job_id,
                    "status": "failed",
                    "error": job.error,
                    "error_type": job.error_type
                }, event_id=str(event_counter))
                break
            
            elif job.status == JobStatus.CANCELLED.value:
                yield format_sse_message("result", {
                    "job_id": job_id,
                    "status": "cancelled"
                }, event_id=str(event_counter))
                break
            
            # Send heartbeat comment (not a real event, just keeps connection alive)
            yield format_sse_comment(f"heartbeat {datetime.utcnow().isoformat()}")
            
            # Poll interval
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled: job={job_id}")
    except Exception as e:
        logger.error(f"SSE error: {e}")
        yield format_sse_message("error", {
            "message": str(e)
        }, event_id=str(event_counter))


async def user_event_generator(
    request: Request,
    user_id: str,
    timeout: float = 300
) -> AsyncGenerator[str, None]:
    """
    Generator that yields SSE events for all user's jobs.
    
    Args:
        request: FastAPI request
        user_id: User ID
        timeout: Maximum stream time
    
    Yields:
        SSE formatted messages for all user's active jobs
    """
    job_store = await get_job_store()
    start_time = asyncio.get_event_loop().time()
    event_counter = 0
    known_jobs = {}  # job_id -> last known state
    
    # Send initial connection
    yield format_sse_message("connected", {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }, event_id=str(event_counter))
    event_counter += 1
    
    try:
        while True:
            if await request.is_disconnected():
                break
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                yield format_sse_message("timeout", {
                    "message": "Stream timeout reached"
                }, event_id=str(event_counter))
                break
            
            # Get user's active jobs
            jobs = await job_store.get_user_jobs(
                user_id=user_id,
                limit=50
            )
            
            for job in jobs:
                job_key = job.id
                last_state = known_jobs.get(job_key)
                
                current_state = {
                    "status": job.status,
                    "progress": job.progress
                }
                
                # Detect changes
                if last_state != current_state:
                    if job.status == JobStatus.COMPLETED.value:
                        result = await job_store.get_job_result(job.id)
                        yield format_sse_message("job_completed", {
                            "job_id": job.id,
                            **(result or {})
                        }, event_id=str(event_counter))
                        event_counter += 1
                        
                    elif job.status == JobStatus.FAILED.value:
                        yield format_sse_message("job_failed", {
                            "job_id": job.id,
                            "error": job.error
                        }, event_id=str(event_counter))
                        event_counter += 1
                        
                    elif job.progress and last_state and last_state.get("progress") != job.progress:
                        yield format_sse_message("job_progress", {
                            "job_id": job.id,
                            **job.progress
                        }, event_id=str(event_counter))
                        event_counter += 1
                    
                    known_jobs[job_key] = current_state
            
            # Clean up completed/failed jobs from known_jobs to prevent unbounded growth
            completed_statuses = {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
            active_job_ids = {job.id for job in jobs}
            stale_keys = [k for k in known_jobs if k not in active_job_ids or known_jobs[k].get("status") in completed_statuses]
            for k in stale_keys:
                del known_jobs[k]
            
            # Heartbeat
            yield format_sse_comment(f"heartbeat {datetime.utcnow().isoformat()}")
            
            await asyncio.sleep(2)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"SSE error: {e}")
        yield format_sse_message("error", {
            "message": str(e)
        })


# ============================================================================
# SSE Endpoints
# ============================================================================

@router.get("/job/{job_id}")
async def sse_job_updates(
    request: Request,
    job_id: str,
    timeout: Optional[float] = Query(300, description="Stream timeout in seconds"),
    current_user: dict = Depends(get_current_user)
):
    """
    Stream Server-Sent Events for a job.
    
    Provides real-time updates for a single job:
    - progress: Processing progress updates
    - result: Final result when complete
    - error: If job fails
    
    Args:
        job_id: Job ID to monitor
        timeout: Maximum stream duration (default 5 minutes)
    
    Returns:
        SSE stream
    
    Example client (JavaScript):
    ```javascript
    const eventSource = new EventSource('/sse/job/123?token=xxx');
    
    eventSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        console.log('Progress:', data.current_step, '/', data.total_steps);
    });
    
    eventSource.addEventListener('result', (e) => {
        const data = JSON.parse(e.data);
        console.log('Result:', data.response);
        eventSource.close();
    });
    ```
    """
    user_id = str(current_user.get("user_id"))
    
    # Verify job exists and user owns it
    job_store = await get_job_store()
    job = await job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if str(job.user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return StreamingResponse(
        job_event_generator(request, job_id, user_id, timeout),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/user/{user_id}")
async def sse_user_updates(
    request: Request,
    user_id: str,
    timeout: Optional[float] = Query(300, description="Stream timeout in seconds"),
    current_user: dict = Depends(get_current_user)
):
    """
    Stream Server-Sent Events for all user's jobs.
    
    Provides updates for all jobs belonging to the user.
    Useful for dashboard-style UIs.
    
    Events:
    - job_completed: A job finished successfully
    - job_failed: A job failed
    - job_progress: Progress update for any job
    
    Args:
        user_id: User ID
        timeout: Maximum stream duration
    
    Returns:
        SSE stream
    """
    # Verify user_id matches authenticated user
    if str(current_user.get("user_id")) != str(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return StreamingResponse(
        user_event_generator(request, user_id, timeout),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def sse_health():
    """Check SSE service health."""
    return {
        "status": "healthy",
        "service": "sse",
        "message": "SSE endpoints available"
    }
