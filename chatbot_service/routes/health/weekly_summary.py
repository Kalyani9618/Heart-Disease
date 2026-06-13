"""
Weekly Summary Trigger Route
=============================
Separate prefix for triggering weekly summary generation.
Endpoint:
    POST /weekly-summary/trigger
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("weekly-summary")

router = APIRouter()


@router.get("/health")
async def weekly_summary_health():
    """Health check for weekly summary service."""
    return {
        "status": "healthy",
        "service": "Weekly Summary",
        "message": "Weekly summary trigger available via POST /weekly-summary/trigger",
    }


class TriggerRequest(BaseModel):
    user_id: str


class TriggerResponse(BaseModel):
    status: str
    user_id: str
    message: str
    triggered_at: str


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_weekly_summary(request: TriggerRequest):
    """Trigger generation of a weekly health summary for a user."""
    masked_user = f"***{request.user_id[-4:]}" if len(request.user_id) > 4 else "****"
    logger.info(f"Weekly summary triggered for user {masked_user}")

    # TODO: Integrate with actual background job system (ARQ) to generate summary
    return TriggerResponse(
        status="pending",
        user_id=request.user_id,
        message="Weekly summary generation request received. Note: Background job integration pending.",
        triggered_at=datetime.utcnow().isoformat() + "Z",
    )
