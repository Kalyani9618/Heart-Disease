"""
Job Management API Routes

Provides endpoints for:
- Querying job status
- Listing user jobs
- Cancelling jobs
- Retrying failed jobs
- Viewing dead letter queue
- Queue statistics

Usage:
    GET /jobs/{job_id} - Get job status and result
    GET /jobs - List user's jobs
    DELETE /jobs/{job_id} - Cancel a job
    POST /jobs/{job_id}/retry - Retry a failed job
    GET /jobs/stats/queue - Get queue statistics
"""


import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field

from core.security import get_current_user
from core.services.job_store import get_job_store, JobStatus, Job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Job Management"])


# ============================================================================
# Response Models
# ============================================================================

class JobProgressResponse(BaseModel):
    """Job progress details."""
    current_step: int = 0
    total_steps: int = 0
    current_node: str = ""
    message: str = ""


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_ms: Optional[int] = None
    progress: Optional[JobProgressResponse] = None
    retry_count: int = 0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    """Response for job listing."""
    jobs: List[JobStatusResponse]
    total: int
    offset: int
    limit: int


class QueueStatsResponse(BaseModel):
    """Response for queue statistics."""
    pending: int
    processing: int
    completed: int
    failed: int
    cancelled: int
    total_active: int


class JobSubmitResponse(BaseModel):
    """Response after submitting a job."""
    job_id: str
    status: str
    created_at: str
    status_url: str
    ws_url: str
    sse_url: str


# ============================================================================
# Routes
# ============================================================================

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    include_result: bool = Query(False, description="Include full result if available"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the status of a job.
    
    Returns job status, progress, timing information, and optionally the result.
    
    Args:
        job_id: The job ID to query
        include_result: Whether to include the full result
    
    Returns:
        Job status response
    
    Raises:
        404: Job not found
        403: Job belongs to different user
    """
    job_store = await get_job_store()
    job = await job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Verify ownership
    if str(job.user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only view your own jobs"
        )
    
    # Calculate processing time
    processing_time_ms = None
    if job.started_at and job.completed_at:
        from datetime import datetime
        try:
            started = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
            completed = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00"))
            processing_time_ms = int((completed - started).total_seconds() * 1000)
        except ValueError:
            pass
    
    # Get result if requested and available
    result = None
    if include_result and job.status == JobStatus.COMPLETED.value:
        result = await job_store.get_job_result(job_id)
    
    # Build progress response
    progress = None
    if job.progress:
        progress = JobProgressResponse(**job.progress)
    
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        processing_time_ms=processing_time_ms,
        progress=progress,
        retry_count=job.retry_count,
        error=job.error,
        result=result
    )


@router.get("/{job_id}/result")
async def get_job_result(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the result of a completed job.
    
    Args:
        job_id: The job ID
    
    Returns:
        Full job result
    
    Raises:
        404: Job not found or not completed
        403: Job belongs to different user
    """
    job_store = await get_job_store()
    job = await job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Verify ownership
    if str(job.user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} is not completed (status: {job.status})"
        )
    
    result = await job_store.get_job_result(job_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result for job {job_id} not found"
        )
    
    return result


@router.get("/", response_model=JobListResponse)
async def list_user_jobs(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum jobs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List jobs for the current user.
    
    Args:
        status_filter: Optional filter by status (pending, processing, completed, failed, cancelled)
        limit: Maximum number of jobs to return (1-100)
        offset: Offset for pagination
    
    Returns:
        List of jobs with pagination info
    """
    job_store = await get_job_store()
    user_id = str(current_user.get("user_id"))
    
    jobs = await job_store.get_user_jobs(
        user_id=user_id,
        limit=limit,
        offset=offset,
        status_filter=status_filter
    )
    
    # Convert to response models
    job_responses = []
    for job in jobs:
        progress = None
        if job.progress:
            progress = JobProgressResponse(**job.progress)
        
        job_responses.append(JobStatusResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            progress=progress,
            retry_count=job.retry_count,
            error=job.error
        ))
    
    return JobListResponse(
        jobs=job_responses,
        total=len(job_responses),  # Note: page count only, use count query for real total
        offset=offset,
        limit=limit
    )


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    reason: Optional[str] = Query(None, description="Cancellation reason"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Cancel a pending or processing job.
    
    Args:
        job_id: The job ID to cancel
        reason: Optional reason for cancellation
    
    Returns:
        Cancellation confirmation
    
    Raises:
        404: Job not found
        403: Job belongs to different user
        400: Job cannot be cancelled (already completed/failed)
    """
    job_store = await get_job_store()
    job = await job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Verify ownership
    if str(job.user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check if job can be cancelled
    if job.status not in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    cancelled_job = await job_store.cancel_job(job_id)
    
    if not cancelled_job:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job"
        )
    
    logger.info(f"❌ Job {job_id} cancelled by user {current_user.get('user_id')}: {reason}")
    
    return {
        "status": "cancelled",
        "job_id": job_id,
        "message": f"Job cancelled: {reason}" if reason else "Job cancelled"
    }


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Retry a failed job.
    
    Requeues the job for processing with reset retry count.
    
    Args:
        job_id: The job ID to retry
    
    Returns:
        Retry confirmation with new job status
    
    Raises:
        404: Job not found
        403: Job belongs to different user
        400: Job cannot be retried (not failed)
    """
    job_store = await get_job_store()
    job = await job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Verify ownership
    if str(job.user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Only failed jobs can be retried
    if job.status != JobStatus.FAILED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retry job with status: {job.status}"
        )
    
    # Queue for processing via ARQ first (if available)
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        
        redis_settings = RedisSettings(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
        )
        
        pool = await create_pool(redis_settings)
        await pool.enqueue_job(
            "process_chat_message",
            job_id,
            job.user_id,
            job.query,
            session_id=job.session_id,
            priority=job.priority,
            metadata=job.metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue retry job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue job for retry"
        )
    
    # Only update status to PENDING after successful enqueue
    updated_job = await job_store.update_job_status(
        job_id,
        JobStatus.PENDING.value,
        metadata={"manual_retry": True}
    )
    
    logger.info(f"🔄 Job {job_id} queued for retry by user {current_user.get('user_id')}")
    
    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Job queued for retry"
    }


@router.get("/stats/queue", response_model=QueueStatsResponse)
async def get_queue_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get queue statistics.
    
    Returns counts of jobs in various states.
    
    Returns:
        Queue statistics
    """
    job_store = await get_job_store()
    stats = await job_store.get_queue_stats()
    
    total_active = stats.get("pending", 0) + stats.get("processing", 0)
    
    return QueueStatsResponse(
        pending=stats.get("pending", 0),
        processing=stats.get("processing", 0),
        completed=stats.get("completed", 0),
        failed=stats.get("failed", 0),
        cancelled=stats.get("cancelled", 0),
        total_active=total_active
    )


@router.get("/stats/dead-letter")
async def list_dead_letter_jobs(
    limit: int = Query(20, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List jobs in the dead letter queue.
    
    These are jobs that have exceeded maximum retry attempts.
    
    Args:
        limit: Maximum jobs to return
    
    Returns:
        List of dead letter jobs
    """
    # This requires admin access in production
    # For now, return user's dead jobs only
    
    job_store = await get_job_store()
    user_id = str(current_user.get("user_id"))
    
    # Get failed jobs for this user
    jobs = await job_store.get_user_jobs(
        user_id=user_id,
        limit=limit,
        status_filter=JobStatus.FAILED.value
    )
    
    # Filter to those with max retries exceeded
    dead_jobs = [j for j in jobs if j.retry_count >= j.max_retries]
    
    return {
        "jobs": [
            {
                "job_id": j.id,
                "query": j.query[:100] + "..." if len(j.query) > 100 else j.query,
                "created_at": j.created_at,
                "failed_at": j.completed_at,
                "retry_count": j.retry_count,
                "error": j.error,
                "error_type": j.error_type
            }
            for j in dead_jobs
        ],
        "total": len(dead_jobs)
    }
