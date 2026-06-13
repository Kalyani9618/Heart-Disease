"""
WebSocket Connection Manager with Heartbeat Support

Manages WebSocket connections for real-time result delivery.
Includes heartbeat mechanism to prevent load balancer connection termination.

CRITICAL: WebSocket Heartbeats Required
Load Balancers (AWS ALB, Nginx, GCP Load Balancer) will silently kill idle 
WebSocket connections after 30-60 seconds of inactivity. If an agent is 
"thinking" for 45 seconds without sending any data, the connection will be 
terminated, and the client will never receive the result.

Solution: Send a "processing..." keep-alive signal every 15 seconds during job execution.


Features:
- Per-user connections
- Per-job subscriptions
- Broadcast to user's devices
- Graceful disconnection handling
- Heartbeat keep-alive during long operations
- Redis pub/sub for multi-instance coordination

Usage:
    ws_manager = await get_ws_manager()
    await ws_manager.connect(websocket, user_id)
    await ws_manager.broadcast_job_result(job_id, user_id, result)
"""

import os
import json
import asyncio
import logging
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from weakref import WeakSet
from dataclasses import dataclass, field
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
HEARTBEAT_INTERVAL = int(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", "15"))  # seconds
PUBSUB_CHANNEL = "chatbot:websocket:broadcast"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection."""
    websocket: WebSocket
    user_id: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    subscribed_jobs: Set[str] = field(default_factory=set)
    
    async def send_json(self, data: Dict[str, Any]) -> bool:
        """Send JSON data to client. Returns False if disconnected."""
        try:
            await self.websocket.send_json(data)
            return True
        except Exception as e:
            logger.debug(f"Failed to send to {self.user_id}: {e}")
            return False


# ============================================================================
# Heartbeat Manager
# ============================================================================

class WebSocketHeartbeatManager:
    """
    Manages heartbeat signals for active jobs.
    
    Prevents load balancer connection termination during long agent processing.
    Sends "processing..." messages every 15 seconds to all subscribers of a job.
    """
    
    def __init__(self, ws_manager: "WebSocketConnectionManager"):
        self.ws_manager = ws_manager
        self._active_jobs: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start_heartbeat(self, job_id: str) -> None:
        """Start sending heartbeats for a job."""
        if job_id in self._active_jobs:
            return  # Already running
        
        self._running = True
        task = asyncio.create_task(self._heartbeat_loop(job_id))
        self._active_jobs[job_id] = task
        logger.debug(f"💓 Started heartbeat for job {job_id}")
    
    async def stop_heartbeat(self, job_id: str) -> None:
        """Stop sending heartbeats for a job."""
        if job_id in self._active_jobs:
            task = self._active_jobs.pop(job_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug(f"💔 Stopped heartbeat for job {job_id}")
    
    async def _heartbeat_loop(self, job_id: str) -> None:
        """Send periodic heartbeat messages."""
        try:
            while self._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                
                heartbeat_message = {
                    "type": "heartbeat",
                    "job_id": job_id,
                    "status": "processing",
                    "message": "Agent is working...",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self.ws_manager.broadcast_to_job(job_id, heartbeat_message)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error for job {job_id}: {e}")
    
    async def shutdown(self) -> None:
        """Stop all heartbeats."""
        self._running = False
        for job_id in list(self._active_jobs.keys()):
            await self.stop_heartbeat(job_id)


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class WebSocketConnectionManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Features:
    - Track connections per user
    - Subscribe connections to job updates
    - Broadcast results to subscribers
    - Multi-instance coordination via Redis pub/sub
    - Heartbeat keep-alive during processing
    """
    
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Connection tracking
        self._connections: Dict[str, List[WebSocketConnection]] = {}  # user_id -> connections
        self._job_subscribers: Dict[str, Set[str]] = {}  # job_id -> user_ids
        
        # Heartbeat manager
        self.heartbeat_manager: Optional[WebSocketHeartbeatManager] = None
        
        self._initialized = False
        self._pubsub_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection and pub/sub listener."""
        if self._initialized:
            return
        
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            
            # Initialize heartbeat manager
            self.heartbeat_manager = WebSocketHeartbeatManager(self)
            
            # Start pub/sub listener for multi-instance coordination
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(PUBSUB_CHANNEL)
            self._pubsub_task = asyncio.create_task(self._pubsub_listener())
            
            self._initialized = True
            logger.info(f"✅ WebSocketConnectionManager initialized with heartbeat support")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebSocketConnectionManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown manager and close all connections."""
        logger.info("🛑 Shutting down WebSocketConnectionManager...")
        
        # Stop heartbeat manager
        if self.heartbeat_manager:
            await self.heartbeat_manager.shutdown()
        
        # Stop pub/sub listener
        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for user_id, connections in list(self._connections.items()):
            for conn in connections:
                try:
                    await conn.websocket.close()
                except Exception:
                    pass
        
        self._connections.clear()
        self._job_subscribers.clear()
        
        # Close Redis
        if self.pubsub:
            await self.pubsub.unsubscribe(PUBSUB_CHANNEL)
            await self.pubsub.aclose()
        
        if self.redis:
            await self.redis.aclose()
        
        self._initialized = False
        logger.info("✅ WebSocketConnectionManager shutdown complete")
    
    async def close_all(self) -> None:
        """Alias for shutdown() - closes all connections gracefully."""
        await self.shutdown()
    
    # ========================================================================
    # Connection Management
    # ========================================================================
    
    async def connect(self, websocket: WebSocket, user_id: str) -> WebSocketConnection:
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        
        connection = WebSocketConnection(websocket=websocket, user_id=user_id)
        
        if user_id not in self._connections:
            self._connections[user_id] = []
        self._connections[user_id].append(connection)
        
        logger.info(f"🔌 WebSocket connected: user={user_id}")
        
        # Send welcome message
        await connection.send_json({
            "type": "connected",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return connection
    
    async def disconnect(self, connection: WebSocketConnection) -> None:
        """Remove a disconnected WebSocket connection."""
        user_id = connection.user_id
        
        if user_id in self._connections:
            self._connections[user_id] = [
                c for c in self._connections[user_id] 
                if c is not connection
            ]
            if not self._connections[user_id]:
                del self._connections[user_id]
        
        # Remove from job subscriptions
        for job_id in list(connection.subscribed_jobs):
            await self.unsubscribe_from_job(connection, job_id)
        
        logger.info(f"🔌 WebSocket disconnected: user={user_id}")
    
    def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
        """Get all connections for a user."""
        return self._connections.get(user_id, [])
    
    # ========================================================================
    # Job Subscriptions
    # ========================================================================
    
    async def subscribe_to_job(
        self, 
        connection: WebSocketConnection, 
        job_id: str
    ) -> None:
        """Subscribe a connection to job updates."""
        connection.subscribed_jobs.add(job_id)
        
        if job_id not in self._job_subscribers:
            self._job_subscribers[job_id] = set()
        self._job_subscribers[job_id].add(connection.user_id)
        
        logger.debug(f"📡 User {connection.user_id} subscribed to job {job_id}")
    
    async def unsubscribe_from_job(
        self, 
        connection: WebSocketConnection, 
        job_id: str
    ) -> None:
        """Unsubscribe a connection from job updates."""
        connection.subscribed_jobs.discard(job_id)
        
        if job_id in self._job_subscribers:
            self._job_subscribers[job_id].discard(connection.user_id)
            if not self._job_subscribers[job_id]:
                del self._job_subscribers[job_id]
    
    # ========================================================================
    # Broadcasting
    # ========================================================================
    
    async def broadcast_to_user(
        self, 
        user_id: str, 
        message: Dict[str, Any]
    ) -> int:
        """Broadcast message to all connections of a user."""
        connections = self.get_user_connections(user_id)
        sent_count = 0
        
        for conn in connections:
            if await conn.send_json(message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_job(
        self, 
        job_id: str, 
        message: Dict[str, Any]
    ) -> int:
        """Broadcast message to all subscribers of a job."""
        user_ids = self._job_subscribers.get(job_id, set())
        sent_count = 0
        
        for user_id in user_ids:
            sent_count += await self.broadcast_to_user(user_id, message)
        
        return sent_count
    
    async def broadcast_job_progress(
        self,
        job_id: str,
        step: int,
        total: int,
        node: str,
        message: str = ""
    ) -> None:
        """Broadcast job progress update."""
        progress_message = {
            "type": "progress",
            "job_id": job_id,
            "step": step,
            "total": total,
            "node": node,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Local broadcast
        await self.broadcast_to_job(job_id, progress_message)
        
        # Publish to Redis for multi-instance coordination
        if self.redis:
            await self.redis.publish(
                PUBSUB_CHANNEL,
                json.dumps({"job_id": job_id, "data": progress_message})
            )
    
    async def broadcast_job_result(
        self,
        job_id: str,
        user_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Broadcast job result to subscribers."""
        result_message = {
            "type": "result",
            "job_id": job_id,
            **result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to job subscribers
        await self.broadcast_to_job(job_id, result_message)
        
        # Also broadcast to user (in case they're not subscribed to job specifically)
        await self.broadcast_to_user(user_id, result_message)
        
        # Publish to Redis for multi-instance coordination
        if self.redis:
            await self.redis.publish(
                PUBSUB_CHANNEL,
                json.dumps({"job_id": job_id, "user_id": user_id, "data": result_message})
            )
    
    # ========================================================================
    # Pub/Sub Listener (Multi-Instance Coordination)
    # ========================================================================
    
    async def _pubsub_listener(self) -> None:
        """Listen for messages from other instances."""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        job_id = data.get("job_id")
                        user_id = data.get("user_id")
                        payload = data.get("data", {})
                        
                        # Broadcast to local subscribers
                        if job_id:
                            await self.broadcast_to_job(job_id, payload)
                        if user_id:
                            await self.broadcast_to_user(user_id, payload)
                            
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        logger.error(f"Pub/sub message handling error: {e}")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Pub/sub listener error: {e}")


# ============================================================================
# Singleton Instance
# ============================================================================

_ws_manager: Optional[WebSocketConnectionManager] = None


async def get_ws_manager() -> WebSocketConnectionManager:
    """Get WebSocketConnectionManager singleton."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketConnectionManager()
        await _ws_manager.initialize()
    return _ws_manager


async def initialize_ws_manager() -> WebSocketConnectionManager:
    """Initialize WebSocket manager during app startup."""
    global _ws_manager
    _ws_manager = WebSocketConnectionManager()
    await _ws_manager.initialize()
    return _ws_manager


async def shutdown_ws_manager() -> None:
    """Shutdown WebSocket manager."""
    global _ws_manager
    if _ws_manager:
        await _ws_manager.shutdown()
        _ws_manager = None


# Alias for backward compatibility
WebSocketManager = WebSocketConnectionManager
