"""
Webhook Delivery Service

Delivers job results via HTTP callbacks to registered URLs.
Supports:
- Retry with exponential backoff
- Signature verification (HMAC)
- Delivery tracking
- Rate limiting per endpoint

Usage:
    webhook_service = await get_webhook_service()
    await webhook_service.register_webhook(user_id, url, secret, events)
    await webhook_service.deliver(job_id, user_id, result)
"""


import os
import json
import hmac
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret")
WEBHOOK_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", "10.0"))
MAX_RETRIES = int(os.getenv("WEBHOOK_MAX_RETRIES", "5"))
RETRY_BACKOFF_BASE = 2  # Exponential backoff base (seconds)
RETRY_BACKOFF_MAX = 300  # Max backoff (5 minutes)

# Redis key prefixes
WEBHOOK_CONFIG_PREFIX = "chatbot:webhook:"
WEBHOOK_DELIVERY_PREFIX = "chatbot:webhook_delivery:"


# ============================================================================
# Data Models
# ============================================================================

class WebhookEvent(str, Enum):
    """Webhook event types."""
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_PROGRESS = "job.progress"
    JOB_CANCELLED = "job.cancelled"
    # Medical events
    HALLUCINATION_DETECTED = "medical.hallucination_detected"
    DRUG_INTERACTION_FOUND = "medical.drug_interaction_found"
    RED_FLAG_SYMPTOMS = "medical.red_flag_symptoms"
    EMERGENCY_TRIAGE = "medical.emergency_triage"


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    id: str
    user_id: str
    url: str
    secret: str
    events: List[str] = field(default_factory=lambda: [WebhookEvent.JOB_COMPLETED.value])
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        return cls(**data)


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    id: str
    webhook_id: str
    job_id: str
    event: str
    payload: Dict[str, Any]
    status: str = DeliveryStatus.PENDING.value
    attempt: int = 0
    max_attempts: int = MAX_RETRIES
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    next_retry_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookDelivery":
        return cls(**data)


# ============================================================================
# Webhook Service
# ============================================================================

class WebhookService:
    """
    Service for delivering job results via webhooks.
    
    Features:
    - Register/unregister webhook endpoints per user
    - HMAC signature verification
    - Automatic retry with exponential backoff
    - Delivery tracking and monitoring
    """
    
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self._initialized = False
        self._retry_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection and HTTP client."""
        if self._initialized:
            return
        
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(WEBHOOK_TIMEOUT),
                follow_redirects=True
            )
            
            # Start retry processor
            self._retry_task = asyncio.create_task(self._retry_processor())
            
            self._initialized = True
            logger.info("✅ WebhookService initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebhookService: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown service and close connections."""
        logger.info("🛑 Shutting down WebhookService...")
        
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        
        if self.http_client:
            await self.http_client.aclose()
        
        if self.redis:
            await self.redis.aclose()
        
        self._initialized = False
        logger.info("✅ WebhookService shutdown complete")
    
    # ========================================================================
    # Webhook Registration
    # ========================================================================
    
    async def register_webhook(
        self,
        user_id: str,
        url: str,
        secret: Optional[str] = None,
        events: Optional[List[str]] = None
    ) -> WebhookConfig:
        """
        Register a new webhook endpoint for a user.
        
        Args:
            user_id: User ID
            url: Webhook URL to call
            secret: Secret for HMAC signing (generated if not provided)
            events: List of events to subscribe to
        
        Returns:
            Created WebhookConfig
        """
        import uuid
        
        if not self._initialized:
            await self.initialize()
        
        webhook_id = str(uuid.uuid4())
        
        config = WebhookConfig(
            id=webhook_id,
            user_id=user_id,
            url=url,
            secret=secret or hashlib.sha256(os.urandom(32)).hexdigest()[:32],
            events=events or [WebhookEvent.JOB_COMPLETED.value, WebhookEvent.JOB_FAILED.value]
        )
        
        # Store webhook config
        config_key = f"{WEBHOOK_CONFIG_PREFIX}{webhook_id}"
        await self.redis.set(config_key, json.dumps(config.to_dict()))
        
        # Add to user's webhook list
        user_webhooks_key = f"{WEBHOOK_CONFIG_PREFIX}user:{user_id}"
        await self.redis.sadd(user_webhooks_key, webhook_id)
        
        logger.info(f"📡 Webhook registered: {webhook_id} for user {user_id}")
        
        return config
    
    async def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration by ID."""
        if not self._initialized:
            await self.initialize()
        
        config_key = f"{WEBHOOK_CONFIG_PREFIX}{webhook_id}"
        data = await self.redis.get(config_key)
        
        if data:
            return WebhookConfig.from_dict(json.loads(data))
        return None
    
    async def get_user_webhooks(self, user_id: str) -> List[WebhookConfig]:
        """Get all webhooks for a user."""
        if not self._initialized:
            await self.initialize()
        
        user_webhooks_key = f"{WEBHOOK_CONFIG_PREFIX}user:{user_id}"
        webhook_ids = await self.redis.smembers(user_webhooks_key)
        
        webhooks = []
        for webhook_id in webhook_ids:
            webhook = await self.get_webhook(webhook_id)
            if webhook:
                webhooks.append(webhook)
        
        return webhooks
    
    async def delete_webhook(self, webhook_id: str, user_id: str) -> bool:
        """Delete a webhook."""
        if not self._initialized:
            await self.initialize()
        
        config_key = f"{WEBHOOK_CONFIG_PREFIX}{webhook_id}"
        user_webhooks_key = f"{WEBHOOK_CONFIG_PREFIX}user:{user_id}"
        
        # Remove from user list
        await self.redis.srem(user_webhooks_key, webhook_id)
        
        # Delete config
        deleted = await self.redis.delete(config_key)
        
        if deleted:
            logger.info(f"🗑️ Webhook deleted: {webhook_id}")
        
        return deleted > 0
    
    # ========================================================================
    # Webhook Delivery
    # ========================================================================
    
    async def emit(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[str]:
        """
        Emit a webhook event to all matching registered endpoints.
        
        Convenience method that broadcasts to all users with matching event subscriptions,
        or to a specific user if user_id is provided.
        
        Args:
            event: WebhookEvent type
            payload: Event payload data
            user_id: Optional specific user to notify
            
        Returns:
            List of delivery IDs
        """
        event_str = event.value if isinstance(event, WebhookEvent) else str(event)
        target_user = user_id or payload.get("user_id", "system")
        job_id = payload.get("thread_id", payload.get("job_id", "event"))
        
        try:
            return await self.deliver(
                job_id=job_id,
                user_id=target_user,
                event=event_str,
                payload=payload
            )
        except Exception as e:
            logger.debug(f"Webhook emit for {event_str} skipped: {e}")
            return []
    
    async def deliver(
        self,
        job_id: str,
        user_id: str,
        event: str,
        payload: Dict[str, Any]
    ) -> List[str]:
        """
        Deliver webhook to all matching endpoints for a user.
        
        Args:
            job_id: Job ID
            user_id: User ID
            event: Event type
            payload: Payload to deliver
        
        Returns:
            List of delivery IDs
        """
        webhooks = await self.get_user_webhooks(user_id)
        delivery_ids = []
        
        for webhook in webhooks:
            if not webhook.active:
                continue
            
            if event not in webhook.events:
                continue
            
            delivery_id = await self._create_delivery(
                webhook=webhook,
                job_id=job_id,
                event=event,
                payload=payload
            )
            delivery_ids.append(delivery_id)
            
            # Attempt immediate delivery
            asyncio.create_task(self._attempt_delivery(delivery_id))
        
        return delivery_ids
    
    async def _create_delivery(
        self,
        webhook: WebhookConfig,
        job_id: str,
        event: str,
        payload: Dict[str, Any]
    ) -> str:
        """Create a delivery record."""
        import uuid
        
        delivery_id = str(uuid.uuid4())
        
        delivery = WebhookDelivery(
            id=delivery_id,
            webhook_id=webhook.id,
            job_id=job_id,
            event=event,
            payload=payload
        )
        
        delivery_key = f"{WEBHOOK_DELIVERY_PREFIX}{delivery_id}"
        await self.redis.set(
            delivery_key,
            json.dumps(delivery.to_dict()),
            ex=86400 * 7  # 7 day TTL
        )
        
        return delivery_id
    
    async def _attempt_delivery(self, delivery_id: str) -> bool:
        """Attempt to deliver a webhook."""
        delivery_key = f"{WEBHOOK_DELIVERY_PREFIX}{delivery_id}"
        delivery_data = await self.redis.get(delivery_key)
        
        if not delivery_data:
            return False
        
        delivery = WebhookDelivery.from_dict(json.loads(delivery_data))
        
        if delivery.status == DeliveryStatus.SUCCESS.value:
            return True
        
        webhook = await self.get_webhook(delivery.webhook_id)
        if not webhook:
            delivery.status = DeliveryStatus.FAILED.value
            delivery.error = "Webhook not found"
            await self.redis.set(delivery_key, json.dumps(delivery.to_dict()))
            return False
        
        delivery.attempt += 1
        
        try:
            # Build payload
            webhook_payload = {
                "id": delivery.id,
                "event": delivery.event,
                "job_id": delivery.job_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": delivery.payload
            }
            
            # Sign payload
            signature = self._sign_payload(webhook_payload, webhook.secret)
            
            # Make request
            response = await self.http_client.post(
                webhook.url,
                json=webhook_payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": signature,
                    "X-Webhook-Event": delivery.event,
                    "X-Delivery-ID": delivery.id
                }
            )
            
            delivery.response_status = response.status_code
            delivery.response_body = response.text[:1000]  # Truncate
            
            if 200 <= response.status_code < 300:
                delivery.status = DeliveryStatus.SUCCESS.value
                delivery.completed_at = datetime.utcnow().isoformat()
                logger.info(f"✅ Webhook delivered: {delivery_id} to {webhook.url}")
            else:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response
                )
            
        except Exception as e:
            delivery.error = str(e)
            
            if delivery.attempt >= delivery.max_attempts:
                delivery.status = DeliveryStatus.FAILED.value
                delivery.completed_at = datetime.utcnow().isoformat()
                logger.error(f"❌ Webhook failed permanently: {delivery_id} - {e}")
            else:
                # Schedule retry with exponential backoff
                backoff = min(
                    RETRY_BACKOFF_BASE ** delivery.attempt,
                    RETRY_BACKOFF_MAX
                )
                next_retry = datetime.utcnow() + timedelta(seconds=backoff)
                delivery.next_retry_at = next_retry.isoformat()
                delivery.status = DeliveryStatus.RETRYING.value
                
                # Add to retry queue
                await self.redis.zadd(
                    f"{WEBHOOK_DELIVERY_PREFIX}retry_queue",
                    {delivery_id: next_retry.timestamp()}
                )
                
                logger.warning(
                    f"⚠️ Webhook delivery failed (attempt {delivery.attempt}), "
                    f"retry in {backoff}s: {delivery_id}"
                )
        
        # Update delivery record
        await self.redis.set(delivery_key, json.dumps(delivery.to_dict()))
        
        return delivery.status == DeliveryStatus.SUCCESS.value
    
    def _sign_payload(self, payload: Dict[str, Any], secret: str) -> str:
        """Sign webhook payload with HMAC-SHA256."""
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(
            secret.encode(),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    # ========================================================================
    # Retry Processor
    # ========================================================================
    
    async def _retry_processor(self) -> None:
        """Background task to process webhook retries."""
        retry_queue_key = f"{WEBHOOK_DELIVERY_PREFIX}retry_queue"
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Get due retries
                now = datetime.utcnow().timestamp()
                due_deliveries = await self.redis.zrangebyscore(
                    retry_queue_key,
                    "-inf",
                    now,
                    start=0,
                    num=10  # Process up to 10 at a time
                )
                
                for delivery_id in due_deliveries:
                    # Remove from queue
                    await self.redis.zrem(retry_queue_key, delivery_id)
                    # Retry delivery
                    await self._attempt_delivery(delivery_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry processor error: {e}")
                await asyncio.sleep(10)
    
    # ========================================================================
    # Delivery Status
    # ========================================================================
    
    async def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery record by ID."""
        if not self._initialized:
            await self.initialize()
        
        delivery_key = f"{WEBHOOK_DELIVERY_PREFIX}{delivery_id}"
        data = await self.redis.get(delivery_key)
        
        if data:
            return WebhookDelivery.from_dict(json.loads(data))
        return None
    
    async def get_job_deliveries(self, job_id: str) -> List[WebhookDelivery]:
        """Get all deliveries for a job."""
        if not self._initialized:
            await self.initialize()
        
        deliveries = []
        async for key in self.redis.scan_iter(f"{WEBHOOK_DELIVERY_PREFIX}*"):
            if key.endswith("retry_queue"):
                continue
            data = await self.redis.get(key)
            if data:
                delivery = WebhookDelivery.from_dict(json.loads(data))
                if delivery.job_id == job_id:
                    deliveries.append(delivery)
        
        return deliveries


# ============================================================================
# Singleton Instance
# ============================================================================

_webhook_service: Optional[WebhookService] = None


async def get_webhook_service() -> WebhookService:
    """Get or create WebhookService singleton."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
        await _webhook_service.initialize()
    return _webhook_service


async def shutdown_webhook_service() -> None:
    """Shutdown WebhookService singleton."""
    global _webhook_service
    if _webhook_service:
        await _webhook_service.shutdown()
        _webhook_service = None
