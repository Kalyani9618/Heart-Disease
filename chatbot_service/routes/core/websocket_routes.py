"""
WebSocket Routes for Real-Time Updates

Provides WebSocket endpoints for:
- Subscribing to job results
- Receiving progress updates
- Multi-job subscriptions
- User-level update streams

Usage:
    WebSocket /ws/job/{job_id} - Subscribe to single job updates
    WebSocket /ws/jobs - Subscribe to multiple jobs (send job IDs via messages)
    WebSocket /ws/user/{user_id} - Subscribe to all user's job updates
"""


import logging
import json
from typing import Optional, Set, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from starlette.websockets import WebSocketState

from core.services.websocket_manager import get_ws_manager, WebSocketConnectionManager, WebSocketManager
from core.services.job_store import get_job_store, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


# ============================================================================
# Authentication Helper
# ============================================================================

async def authenticate_websocket(
    websocket: WebSocket,
    token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Authenticate a WebSocket connection.
    
    In production, validate the JWT token.
    For development, allow connections with user_id query param.
    
    Args:
        websocket: The WebSocket connection
        token: Optional JWT token
    
    Returns:
        User info if authenticated, None otherwise
    """
    # In production, validate JWT token
    if token:
        try:
            from core.security import decode_token
            payload = decode_token(token)
            return {"user_id": payload.get("sub"), "token_valid": True}
        except Exception as e:
            logger.warning(f"WebSocket auth failed: {e}")
            return None
    
    # Development fallback removed for security - require token auth
    return None


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@router.websocket("/job/{job_id}")
async def websocket_job_updates(
    websocket: WebSocket,
    job_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for single job updates.
    
    Connects and immediately subscribes to the specified job's updates.
    Receives:
    - progress: Step-by-step progress updates
    - heartbeat: Keep-alive signals during processing
    - result: Final job result or error
    
    Client can send:
    - {"type": "ping"}: Respond with pong
    - {"type": "unsubscribe"}: Stop receiving updates
    
    Args:
        job_id: Job ID to subscribe to
        token: Optional authentication token
    """
    # Authenticate
    user_info = await authenticate_websocket(websocket, token)
    if not user_info:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    user_id = user_info["user_id"]
    
    # Verify user owns this job
    job_store = await get_job_store()
    job = await job_store.get_job(job_id)
    
    if not job:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Job not found")
        return
    
    if str(job.user_id) != str(user_id):
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Access denied")
        return
    
    # Get WebSocket manager and connect
    ws_manager = await get_ws_manager()
    connection = await ws_manager.connect(websocket, user_id)
    
    try:
        # Subscribe to job updates
        await ws_manager.subscribe_to_job(connection, job_id)
        
        # Send current job status
        await connection.send_json({
            "type": "subscribed",
            "job_id": job_id,
            "current_status": job.status,
            "progress": job.progress
        })
        
        # If job is already completed, send result and close
        if job.status == JobStatus.COMPLETED.value:
            result = await job_store.get_job_result(job_id)
            await connection.send_json({
                "type": "result",
                "job_id": job_id,
                "status": "completed",
                **(result or {})
            })
        elif job.status == JobStatus.FAILED.value:
            await connection.send_json({
                "type": "result",
                "job_id": job_id,
                "status": "failed",
                "error": job.error,
                "error_type": job.error_type
            })
        
        # Listen for client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    await connection.send_json({"type": "pong"})
                    
                elif msg_type == "unsubscribe":
                    await ws_manager.unsubscribe_from_job(connection, job_id)
                    await connection.send_json({
                        "type": "unsubscribed",
                        "job_id": job_id
                    })
                    break
                    
                elif msg_type == "get_status":
                    job = await job_store.get_job(job_id)
                    if job:
                        await connection.send_json({
                            "type": "status",
                            "job_id": job_id,
                            "status": job.status,
                            "progress": job.progress
                        })
                        
            except json.JSONDecodeError:
                await connection.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}, job={job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(connection)


@router.websocket("/jobs")
async def websocket_multi_job_updates(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for multiple job subscriptions.
    
    Connects without initial subscriptions. Client can dynamically
    subscribe/unsubscribe to jobs by sending messages.
    
    Client can send:
    - {"type": "subscribe", "job_ids": ["id1", "id2"]}
    - {"type": "unsubscribe", "job_ids": ["id1"]}
    - {"type": "ping"}: Respond with pong
    - {"type": "list"}: List current subscriptions
    
    Receives same updates as single-job endpoint for each subscribed job.
    
    Args:
        token: Optional authentication token
    """
    # Authenticate
    user_info = await authenticate_websocket(websocket, token)
    if not user_info:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    user_id = user_info["user_id"]
    
    ws_manager = await get_ws_manager()
    job_store = await get_job_store()
    connection = await ws_manager.connect(websocket, user_id)
    
    try:
        await connection.send_json({
            "type": "connected",
            "message": "Send subscribe messages to watch jobs"
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    await connection.send_json({"type": "pong"})
                    
                elif msg_type == "subscribe":
                    job_ids = message.get("job_ids", [])
                    subscribed = []
                    
                    for job_id in job_ids:
                        # Verify ownership
                        job = await job_store.get_job(job_id)
                        if job and str(job.user_id) == str(user_id):
                            await ws_manager.subscribe_to_job(connection, job_id)
                            subscribed.append({
                                "job_id": job_id,
                                "status": job.status
                            })
                    
                    await connection.send_json({
                        "type": "subscribed",
                        "jobs": subscribed
                    })
                    
                elif msg_type == "unsubscribe":
                    job_ids = message.get("job_ids", [])
                    
                    for job_id in job_ids:
                        await ws_manager.unsubscribe_from_job(connection, job_id)
                    
                    await connection.send_json({
                        "type": "unsubscribed",
                        "job_ids": job_ids
                    })
                    
                elif msg_type == "list":
                    await connection.send_json({
                        "type": "subscriptions",
                        "job_ids": list(connection.subscribed_jobs)
                    })
                    
            except json.JSONDecodeError:
                await connection.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(connection)


@router.websocket("/user/{user_id}")
async def websocket_user_updates(
    websocket: WebSocket,
    user_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for all user updates.
    
    Receives updates for all jobs belonging to the user.
    Useful for dashboard-style UIs that show all active jobs.
    
    Client receives:
    - Job submitted notifications
    - Progress updates for all jobs
    - Completion/failure notifications
    
    Args:
        user_id: User ID to subscribe to
        token: Optional authentication token
    """
    # Authenticate
    user_info = await authenticate_websocket(websocket, token)
    if not user_info:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Verify user_id matches authenticated user
    if str(user_info["user_id"]) != str(user_id):
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Access denied")
        return
    
    ws_manager = await get_ws_manager()
    job_store = await get_job_store()
    connection = await ws_manager.connect(websocket, user_id)
    
    try:
        # Send current active jobs
        active_jobs = await job_store.get_user_jobs(
            user_id=user_id,
            limit=50,
            status_filter=None  # All statuses
        )
        
        # Subscribe to all active jobs
        active_count = 0
        for job in active_jobs:
            if job.status in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
                await ws_manager.subscribe_to_job(connection, job.id)
                active_count += 1
        
        await connection.send_json({
            "type": "connected",
            "user_id": user_id,
            "active_jobs": active_count,
            "message": f"Subscribed to {active_count} active jobs"
        })
        
        # Listen for client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    await connection.send_json({"type": "pong"})
                    
                elif msg_type == "refresh":
                    # Re-subscribe to active jobs
                    active_jobs = await job_store.get_user_jobs(
                        user_id=user_id,
                        limit=50
                    )
                    
                    # Clear old subscriptions
                    for job_id in list(connection.subscribed_jobs):
                        await ws_manager.unsubscribe_from_job(connection, job_id)
                    
                    # Subscribe to active
                    active_count = 0
                    for job in active_jobs:
                        if job.status in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
                            await ws_manager.subscribe_to_job(connection, job.id)
                            active_count += 1
                    
                    await connection.send_json({
                        "type": "refreshed",
                        "active_jobs": active_count
                    })
                    
            except json.JSONDecodeError:
                await connection.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(connection)


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def websocket_health():
    """Check WebSocket service health."""
    try:
        ws_manager = await get_ws_manager()
        return {
            "status": "healthy",
            "service": "websocket",
            "initialized": ws_manager._initialized
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "websocket",
            "error": str(e)
        }
