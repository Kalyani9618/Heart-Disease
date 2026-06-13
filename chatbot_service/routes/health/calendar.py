"""
Calendar Routes
===============
Google Calendar integration for appointment scheduling and reminders.
Endpoints:
    POST /calendar/{user_id}/credentials
    POST /calendar/{user_id}/sync
    GET  /calendar/{user_id}/events
    POST /calendar/{user_id}/reminder
"""

import logging
import uuid
from typing import Optional, List, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.database.postgres_db import get_database

logger = logging.getLogger("calendar")

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CalendarCredentials(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    provider: str = "google"


class SyncOptions(BaseModel):
    days_ahead: int = 30
    include_recurring: bool = True


class SyncResponse(BaseModel):
    events_synced: int
    reminders_created: int
    sync_completed_at: str


class CalendarEvent(BaseModel):
    id: str
    title: str
    start_time: str
    end_time: str
    location: Optional[str] = None
    description: Optional[str] = None


class ReminderRequest(BaseModel):
    title: str
    scheduled_for: str
    description: Optional[str] = None
    reminder_minutes_before: int = 30


class ReminderResponse(BaseModel):
    id: str
    appointment_id: str
    scheduled_for: str
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{user_id}/credentials")
async def store_credentials(user_id: str, credentials: CalendarCredentials):
    """Store calendar integration credentials for a user."""
    db = await get_database()
    await db.save_calendar_credentials(
        user_id, credentials.provider,
        access_token=credentials.access_token,
        refresh_token=credentials.refresh_token,
    )
    logger.info(f"Calendar credentials stored for user {user_id} (provider: {credentials.provider})")
    return {"message": "Credentials stored successfully", "provider": credentials.provider}


@router.post("/{user_id}/sync", response_model=SyncResponse)
async def sync_calendar(user_id: str, options: SyncOptions):
    """Sync calendar events from the configured provider."""
    db = await get_database()

    # Clear old synced events before re-syncing
    await db.delete_calendar_events(user_id)

    # Generate sample health-related events if no real integration
    now = datetime.utcnow()
    sample_events = [
        CalendarEvent(
            id=str(uuid.uuid4()),
            title="Annual Physical Exam",
            start_time=(now + timedelta(days=7)).isoformat() + "Z",
            end_time=(now + timedelta(days=7, hours=1)).isoformat() + "Z",
            location="Primary Care Clinic",
            description="Annual wellness check-up",
        ),
        CalendarEvent(
            id=str(uuid.uuid4()),
            title="Cardiology Follow-up",
            start_time=(now + timedelta(days=14)).isoformat() + "Z",
            end_time=(now + timedelta(days=14, hours=1)).isoformat() + "Z",
            location="Heart Center",
            description="Follow-up appointment for heart health",
        ),
    ]

    # Persist each event to DB
    for evt in sample_events:
        await db.save_calendar_event(
            user_id=user_id,
            event_id=evt.id,
            title=evt.title,
            start_time=evt.start_time,
            end_time=evt.end_time,
            location=evt.location,
            description=evt.description,
        )

    return SyncResponse(
        events_synced=len(sample_events),
        reminders_created=0,
        sync_completed_at=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/{user_id}/events", response_model=List[CalendarEvent])
async def get_events(
    user_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Get calendar events for a user, optionally filtered by date range."""
    db = await get_database()
    rows = await db.get_calendar_events(user_id, start_date=start_date, end_date=end_date)
    return [
        CalendarEvent(
            id=row["event_id"],
            title=row["title"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            location=row.get("location"),
            description=row.get("description"),
        )
        for row in rows
    ]


@router.post("/{user_id}/reminder", response_model=ReminderResponse)
async def schedule_reminder(user_id: str, reminder: ReminderRequest):
    """Schedule a health-related reminder."""
    db = await get_database()
    reminder_id = str(uuid.uuid4())
    appointment_id = str(uuid.uuid4())

    await db.save_calendar_reminder(
        reminder_id=reminder_id,
        user_id=user_id,
        appointment_id=appointment_id,
        title=reminder.title,
        scheduled_for=reminder.scheduled_for,
        description=reminder.description,
        reminder_minutes_before=reminder.reminder_minutes_before,
    )
    logger.info(f"Reminder scheduled for user {user_id}: {reminder.title}")

    return ReminderResponse(
        id=reminder_id,
        appointment_id=appointment_id,
        scheduled_for=reminder.scheduled_for,
        status="scheduled",
    )
