"""
App Settings Routes (PostgreSQL-backed)
========================================
CRUD endpoints for user app settings and connected devices.

Endpoints:
    GET    /settings/{user_id}
    PUT    /settings/{user_id}
    GET    /settings/{user_id}/devices
    POST   /settings/{user_id}/devices
    DELETE /settings/{user_id}/devices/{device_id}
"""

import logging
import json
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("settings")

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NotificationSettings(BaseModel):
    all: bool = True
    meds: bool = True
    insights: bool = True


class PreferencesSettings(BaseModel):
    units: str = "Metric"
    language: str = "en"
    theme: str = "system"


class AppSettingsResponse(BaseModel):
    user_id: str
    notifications: NotificationSettings = NotificationSettings()
    preferences: PreferencesSettings = PreferencesSettings()


class AppSettingsUpdate(BaseModel):
    notifications: Optional[NotificationSettings] = None
    preferences: Optional[PreferencesSettings] = None


class DeviceModel(BaseModel):
    id: str = ""
    name: str
    type: str = "watch"
    lastSync: str = ""
    status: str = "connected"
    battery: int = 100


class DeviceCreate(BaseModel):
    id: str
    name: str
    type: str = "watch"
    status: str = "connected"
    battery: int = 100


class DeviceResponse(BaseModel):
    id: str
    name: str
    type: str = "watch"
    lastSync: str = ""
    status: str = "connected"
    battery: int = 100


# ---------------------------------------------------------------------------
# Database Helper
# ---------------------------------------------------------------------------

async def _get_db():
    """Get the PostgreSQL database instance."""
    from core.database.postgres_db import get_database
    db = await get_database()
    if not db or not db.pool:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


# ---------------------------------------------------------------------------
# Settings Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}", response_model=AppSettingsResponse)
async def get_settings(user_id: str):
    """Get app settings for a user."""
    db = await _get_db()
    try:
        row = await db.fetch_one(
            "SELECT * FROM user_app_settings WHERE user_id = $1",
            (user_id,)
        )

        if not row:
            return AppSettingsResponse(user_id=user_id)

        # Parse notification settings
        notif = row.get("notification_settings") or {}
        if isinstance(notif, str):
            notif = json.loads(notif)

        # Parse preference settings
        prefs = row.get("preference_settings") or {}
        if isinstance(prefs, str):
            prefs = json.loads(prefs)

        return AppSettingsResponse(
            user_id=user_id,
            notifications=NotificationSettings(
                all=notif.get("all", True),
                meds=notif.get("meds", True),
                insights=notif.get("insights", True),
            ),
            preferences=PreferencesSettings(
                units=prefs.get("units", "Metric"),
                language=prefs.get("language", "en"),
                theme=prefs.get("theme", "system"),
            ),
        )
    except Exception as e:
        logger.error(f"Failed to get settings for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")


@router.put("/{user_id}", response_model=AppSettingsResponse)
async def update_settings(user_id: str, update: AppSettingsUpdate):
    """Create or update app settings for a user (upsert)."""
    db = await _get_db()
    try:
        existing = await db.fetch_one(
            "SELECT * FROM user_app_settings WHERE user_id = $1",
            (user_id,)
        )

        # Merge with existing or defaults
        current_notif = {}
        current_prefs = {}

        if existing:
            cn = existing.get("notification_settings") or {}
            if isinstance(cn, str):
                cn = json.loads(cn)
            current_notif = cn

            cp = existing.get("preference_settings") or {}
            if isinstance(cp, str):
                cp = json.loads(cp)
            current_prefs = cp

        if update.notifications:
            current_notif.update(update.notifications.dict())
        if update.preferences:
            current_prefs.update(update.preferences.dict())

        notif_json = json.dumps(current_notif)
        prefs_json = json.dumps(current_prefs)

        if existing:
            await db.execute_query(
                """UPDATE user_app_settings
                   SET notification_settings = $1, preference_settings = $2, updated_at = $3
                   WHERE user_id = $4""",
                (notif_json, prefs_json, datetime.utcnow(), user_id),
            )
        else:
            await db.execute_query(
                """INSERT INTO user_app_settings (user_id, notification_settings, preference_settings)
                   VALUES ($1, $2, $3)""",
                (user_id, notif_json, prefs_json),
            )

        return await get_settings(user_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update settings for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update settings")


# ---------------------------------------------------------------------------
# Devices Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}/devices", response_model=List[DeviceResponse])
async def get_devices(user_id: str):
    """Get all connected devices for a user."""
    db = await _get_db()
    try:
        rows = await db.fetch_all(
            "SELECT * FROM user_devices WHERE user_id = $1 ORDER BY created_at DESC",
            (user_id,)
        )
        result = []
        for row in rows:
            result.append(DeviceResponse(
                id=row.get("device_id") or str(row.get("id", "")),
                name=row.get("device_name") or row.get("device_type") or "Unknown Device",
                type=row.get("device_type") or "watch",
                lastSync=row["last_sync"].strftime("%Y-%m-%d %H:%M") if row.get("last_sync") else "",
                status=row.get("status") or "connected",
                battery=row.get("battery") or 100,
            ))
        return result
    except Exception as e:
        logger.error(f"Failed to get devices for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get devices")


@router.post("/{user_id}/devices", response_model=DeviceResponse, status_code=201)
async def add_device(user_id: str, device: DeviceCreate):
    """Add a connected device for a user."""
    db = await _get_db()
    try:
        # Check if device already exists
        existing = await db.fetch_one(
            "SELECT id FROM user_devices WHERE user_id = $1 AND device_id = $2",
            (user_id, device.id),
        )

        if existing:
            # Update existing device
            row = await db.execute_query(
                """UPDATE user_devices
                   SET device_name = $1, device_type = $2, status = $3, battery = $4,
                       last_sync = $5, updated_at = $5
                   WHERE user_id = $6 AND device_id = $7
                   RETURNING *""",
                (device.name, device.type, device.status, device.battery,
                 datetime.utcnow(), user_id, device.id),
                fetch_one=True,
            )
        else:
            row = await db.execute_query(
                """INSERT INTO user_devices
                       (user_id, device_id, device_name, device_type, status, battery, last_sync)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)
                   RETURNING *""",
                (user_id, device.id, device.name, device.type, device.status,
                 device.battery, datetime.utcnow()),
                fetch_one=True,
            )

        return DeviceResponse(
            id=row.get("device_id") or str(row.get("id", "")),
            name=row.get("device_name") or device.name,
            type=row.get("device_type") or device.type,
            lastSync="Now",
            status=row.get("status") or device.status,
            battery=row.get("battery") or device.battery,
        )
    except Exception as e:
        logger.error(f"Failed to add device for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add device")


@router.delete("/{user_id}/devices/{device_id}")
async def remove_device(user_id: str, device_id: str):
    """Remove a connected device."""
    db = await _get_db()
    try:
        result = await db.execute_query(
            "DELETE FROM user_devices WHERE user_id = $1 AND device_id = $2 RETURNING id",
            (user_id, device_id),
            fetch_one=True,
        )

        if not result:
            # Also try by integer id
            did = int(device_id) if device_id.isdigit() else -1
            result = await db.execute_query(
                "DELETE FROM user_devices WHERE user_id = $1 AND id = $2 RETURNING id",
                (user_id, did),
                fetch_one=True,
            )

        if not result:
            raise HTTPException(status_code=404, detail="Device not found")

        return {"message": "Device removed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove device for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove device")
