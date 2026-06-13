"""
Job Store Service

Manages job metadata in Redis for the async worker pattern.
Works in conjunction with ARQ for comprehensive job tracking.

Features:
- Job creation and status updates
- Progress tracking for long-running jobs
- User job listing with pagination
- Automatic TTL-based cleanup
- Priority queue support

Usage:
    job_store = await get_job_store()
    job = await job_store.create_job(user_id, query)
    await job_store.update_job_status(job.id, JobStatus.PROCESSING)
"""


import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "24"))

# Redis key prefixes
JOB_PREFIX = "chatbot:job:"
USER_JOBS_PREFIX = "chatbot:user_jobs:"
JOB_RESULT_PREFIX = "chatbot:job_result:"


# ============================================================================
# Data Models
# ============================================================================

class JobStatus(str, Enum):
    """Job lifecycle status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    """Job priority levels for queue routing."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class JobProgress:
    """Track job progress for long-running tasks."""
    current_step: int = 0
    total_steps: int = 0
    current_node: str = ""
    message: str = ""
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Job:
    """Job metadata stored in Redis."""
    id: str
    user_id: str
    query: str
    session_id: Optional[str] = None
    priority: str = JobPriority.NORMAL.value
    status: str = JobStatus.PENDING.value
    worker_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create Job from dictionary."""
        return cls(**data)


# ============================================================================
# Job Store Service
# ============================================================================

class JobStore:
    """
    Redis-based job store for async worker pattern.
    
    Provides:
    - Job creation with unique IDs
    - Status transitions with timestamps
    - Progress tracking for real-time updates
    - User job history with pagination
    - Automatic cleanup of old jobs
    """
    
    def __init__(self, redis_url: str = REDIS_URL):
        """Initialize job store with Redis connection."""
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return
        
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis.ping()
            self._initialized = True
            logger.info(f"✅ JobStore connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.aclose()
            self._initialized = False
            logger.info("🔌 JobStore Redis connection closed")
    
    # ========================================================================
    # Job CRUD Operations
    # ========================================================================
    
    async def create_job(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        priority: str = JobPriority.NORMAL.value,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Job:
        """
        Create a new job and store it in Redis.
        
        Args:
            user_id: User who submitted the job
            query: The chat message/query
            session_id: Optional session ID for conversation context
            priority: Job priority for queue routing
            metadata: Additional job metadata
        
        Returns:
            Created Job instance
        """
        if not self._initialized:
            await self.initialize()
        
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            user_id=user_id,
            query=query,
            session_id=session_id,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store job in Redis
        job_key = f"{JOB_PREFIX}{job_id}"
        await self.redis.set(
            job_key,
            json.dumps(job.to_dict()),
            ex=JOB_TTL_HOURS * 3600  # TTL in seconds
        )
        
        # Add to user's job list (sorted set by creation time)
        user_jobs_key = f"{USER_JOBS_PREFIX}{user_id}"
        await self.redis.zadd(
            user_jobs_key,
            {job_id: datetime.utcnow().timestamp()}
        )
        await self.redis.expire(user_jobs_key, JOB_TTL_HOURS * 3600)
        
        logger.info(f"📝 Job {job_id} created for user {user_id} (priority: {priority})")
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        if not self._initialized:
            await self.initialize()
        
        job_key = f"{JOB_PREFIX}{job_id}"
        job_data = await self.redis.get(job_key)
        
        if job_data:
            return Job.from_dict(json.loads(job_data))
        return None
    
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        worker_id: Optional[str] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Job]:
        """
        Update job status and related fields.
        
        Args:
            job_id: Job ID to update
            status: New status
            worker_id: ID of worker processing the job
            error: Error message if failed
            error_type: Type of error
            metadata: Additional metadata to merge
        
        Returns:
            Updated Job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            logger.warning(f"⚠️ Job {job_id} not found for status update")
            return None
        
        job.status = status
        
        if worker_id:
            job.worker_id = worker_id
        
        if status == JobStatus.PROCESSING.value:
            job.started_at = datetime.utcnow().isoformat()
        elif status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
            job.completed_at = datetime.utcnow().isoformat()
        
        if error:
            job.error = error
            job.error_type = error_type
        
        if metadata:
            job.metadata.update(metadata)
        
        # Save updated job
        job_key = f"{JOB_PREFIX}{job_id}"
        await self.redis.set(
            job_key,
            json.dumps(job.to_dict()),
            ex=JOB_TTL_HOURS * 3600
        )
        
        logger.info(f"📊 Job {job_id} status: {status}")
        
        return job
    
    async def update_job_progress(
        self,
        job_id: str,
        current_step: int,
        total_steps: int,
        current_node: str,
        message: str = ""
    ) -> Optional[Job]:
        """Update job progress for real-time tracking."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.progress = {
            "current_step": current_step,
            "total_steps": total_steps,
            "current_node": current_node,
            "message": message,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        job_key = f"{JOB_PREFIX}{job_id}"
        await self.redis.set(
            job_key,
            json.dumps(job.to_dict()),
            ex=JOB_TTL_HOURS * 3600
        )
        
        return job
    
    async def complete_job(
        self,
        job_id: str,
        result: Dict[str, Any]
    ) -> Optional[Job]:
        """Mark job as completed and store result."""
        # Store result separately (may be large)
        result_key = f"{JOB_RESULT_PREFIX}{job_id}"
        await self.redis.set(
            result_key,
            json.dumps(result),
            ex=JOB_TTL_HOURS * 3600
        )
        
        # Update job status
        return await self.update_job_status(
            job_id,
            JobStatus.COMPLETED.value,
            metadata={"has_result": True}
        )
    
    async def fail_job(
        self,
        job_id: str,
        error_result: Dict[str, Any]
    ) -> Optional[Job]:
        """Mark job as failed with error details."""
        return await self.update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error=error_result.get("error"),
            error_type=error_result.get("error_type")
        )
    
    async def cancel_job(self, job_id: str) -> Optional[Job]:
        """Cancel a pending or processing job."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        # Can only cancel pending or processing jobs
        if job.status not in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
            logger.warning(f"⚠️ Cannot cancel job {job_id} with status {job.status}")
            return None
        
        return await self.update_job_status(job_id, JobStatus.CANCELLED.value)
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get stored job result."""
        if not self._initialized:
            await self.initialize()
        
        result_key = f"{JOB_RESULT_PREFIX}{job_id}"
        result_data = await self.redis.get(result_key)
        
        if result_data:
            return json.loads(result_data)
        return None
    
    # ========================================================================
    # Job Listing and Queries
    # ========================================================================
    
    async def get_user_jobs(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Job]:
        """
        Get jobs for a user with pagination.
        
        Args:
            user_id: User ID
            limit: Maximum jobs to return
            offset: Offset for pagination
            status_filter: Optional filter by status
        
        Returns:
            List of Job instances
        """
        if not self._initialized:
            await self.initialize()
        
        user_jobs_key = f"{USER_JOBS_PREFIX}{user_id}"
        
        # Get job IDs (sorted by creation time, most recent first)
        job_ids = await self.redis.zrevrange(
            user_jobs_key,
            offset,
            offset + limit - 1
        )
        
        jobs = []
        for job_id in job_ids:
            job = await self.get_job(job_id)
            if job:
                if status_filter is None or job.status == status_filter:
                    jobs.append(job)
        
        return jobs
    
    async def get_jobs_by_status(
        self,
        status: str,
        limit: int = 100
    ) -> List[Job]:
        """Get jobs by status (for admin/monitoring)."""
        if not self._initialized:
            await self.initialize()
        
        # Scan for job keys (use carefully in production)
        jobs = []
        async for key in self.redis.scan_iter(f"{JOB_PREFIX}*", count=limit):
            job = await self.get_job(key.replace(JOB_PREFIX, ""))
            if job and job.status == status:
                jobs.append(job)
                if len(jobs) >= limit:
                    break
        
        return jobs
    
    # ========================================================================
    # Cleanup Operations
    # ========================================================================
    
    async def cleanup_old_jobs(self, hours: int = 24) -> int:
        """
        Clean up jobs older than specified hours.
        
        Returns:
            Number of jobs cleaned up
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
        deleted_count = 0
        
        # Scan all user job lists
        async for key in self.redis.scan_iter(f"{USER_JOBS_PREFIX}*"):
            # Remove old entries from sorted set
            removed = await self.redis.zremrangebyscore(key, "-inf", cutoff_time)
            deleted_count += removed
        
        logger.info(f"🧹 Cleaned up {deleted_count} old job references")
        return deleted_count
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics for monitoring."""
        if not self._initialized:
            await self.initialize()
        
        stats = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }
        
        # Count jobs by status (sample-based for performance)
        async for key in self.redis.scan_iter(f"{JOB_PREFIX}*", count=1000):
            job_data = await self.redis.get(key)
            if job_data:
                job = json.loads(job_data)
                status = job.get("status", "unknown")
                if status in stats:
                    stats[status] += 1
        
        return stats


# ============================================================================
# Singleton Instance
# ============================================================================

_job_store: Optional[JobStore] = None


async def get_job_store() -> JobStore:
    """Get or create JobStore singleton."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
        await _job_store.initialize()
    return _job_store


async def shutdown_job_store() -> None:
    """Shutdown JobStore singleton."""
    global _job_store
    if _job_store:
        await _job_store.shutdown()
        _job_store = None
