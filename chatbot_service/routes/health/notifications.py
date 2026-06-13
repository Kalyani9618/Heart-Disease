"""
Notifications Routes
====================
Multi-channel notification delivery (WhatsApp, Email, Push).
Endpoints:
    POST /notifications/whatsapp
    POST /notifications/email
    POST /notifications/register-device
    POST /notifications/push
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.database.postgres_db import get_database

logger = logging.getLogger("notifications")

router = APIRouter()

MAX_DEVICES_PER_USER = 10


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WhatsAppRequest(BaseModel):
    to: str = Field(..., description="Phone number in E.164 format")
    message: str
    template: Optional[str] = None


class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    html: Optional[bool] = False


class DeviceRegistration(BaseModel):
    user_id: str
    device_token: str
    platform: str = Field(..., description="ios, android, or web")


class PushRequest(BaseModel):
    user_id: str
    title: str
    body: str
    data: Optional[Dict[str, Any]] = None


class NotificationResponse(BaseModel):
    id: str
    status: str
    channel: str
    sent_at: str
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/whatsapp", response_model=NotificationResponse)
async def send_whatsapp(request: WhatsAppRequest):
    """Send a WhatsApp notification via Twilio/WhatsApp Business API."""
    notification_id = str(uuid.uuid4())

    # In production, integrate with Twilio WhatsApp API
    # Mask phone number in logs
    masked_phone = f"***{request.to[-4:]}" if len(request.to) > 4 else "****"
    logger.info(f"WhatsApp notification queued to {masked_phone}")

    return NotificationResponse(
        id=notification_id,
        status="queued",
        channel="whatsapp",
        sent_at=datetime.utcnow().isoformat() + "Z",
        message="WhatsApp notification queued for delivery",
    )


@router.post("/email", response_model=NotificationResponse)
async def send_email(request: EmailRequest):
    """Send an email notification."""
    notification_id = str(uuid.uuid4())

    # In production, integrate with SMTP / SendGrid / SES
    # Mask email in logs
    at_idx = request.to.find("@")
    masked_email = f"***{request.to[at_idx:]}" if at_idx > 0 else "***@***"
    logger.info(f"Email notification queued to {masked_email}: {request.subject}")

    return NotificationResponse(
        id=notification_id,
        status="queued",
        channel="email",
        sent_at=datetime.utcnow().isoformat() + "Z",
        message="Email notification queued for delivery",
    )


@router.post("/register-device")
async def register_device(request: DeviceRegistration):
    """Register a device for push notifications."""
    if request.platform not in ("ios", "android", "web"):
        raise HTTPException(status_code=400, detail="Platform must be ios, android, or web")

    db = await get_database()

    # Limit devices per user
    count = await db.count_user_push_devices(request.user_id)
    if count >= MAX_DEVICES_PER_USER:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_DEVICES_PER_USER} devices per user")

    await db.register_push_device(request.user_id, request.device_token, request.platform)

    masked_user = f"***{request.user_id[-4:]}" if len(request.user_id) > 4 else "****"
    logger.info(f"Device registered for user {masked_user}: {request.platform}")
    return {"message": "Device registered successfully", "platform": request.platform}


@router.post("/push", response_model=NotificationResponse)
async def send_push(request: PushRequest):
    """Send a push notification to registered devices."""
    notification_id = str(uuid.uuid4())

    db = await get_database()
    devices = await db.get_user_push_devices(request.user_id)
    masked_user = f"***{request.user_id[-4:]}" if len(request.user_id) > 4 else "****"
    if not devices:
        logger.warning(f"No devices registered for user {masked_user}")

    # In production, integrate with FCM / APNs
    logger.info(f"Push notification queued for user {masked_user}")

    return NotificationResponse(
        id=notification_id,
        status="queued",
        channel="push",
        sent_at=datetime.utcnow().isoformat() + "Z",
        message=f"Push notification queued for {len(devices)} device(s)",
    )
