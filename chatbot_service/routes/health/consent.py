"""
Consent Management Routes
=========================
User consent tracking for data processing, sharing, and analytics.
Persisted to PostgreSQL (user_consents table).
Endpoints:
    GET    /consent/{user_id}
    PUT    /consent/{user_id}
    DELETE /consent/{user_id}/{consent_type}
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.database.postgres_db import get_database

logger = logging.getLogger("consent")

router = APIRouter()


# ---------------------------------------------------------------------------
# Default consent template (used to seed initial consents for new users)
# ---------------------------------------------------------------------------

DEFAULT_CONSENTS = {
    "data_processing": {
        "description": "Allow processing of health data for AI-powered analysis",
        "required": True,
    },
    "data_sharing_doctors": {
        "description": "Share health data with your healthcare providers",
        "required": False,
    },
    "data_sharing_research": {
        "description": "Allow anonymized data to be used for medical research",
        "required": False,
    },
    "analytics": {
        "description": "Allow anonymous usage analytics to improve the service",
        "required": False,
    },
    "notifications": {
        "description": "Receive health-related notifications and reminders",
        "required": False,
    },
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ConsentStatus(BaseModel):
    user_id: str
    consents: Dict[str, Any]
    last_updated: Optional[str] = None


class ConsentUpdate(BaseModel):
    consents: Dict[str, bool]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ensure_user_consents(user_id: str):
    """Ensure the user has all default consent types in the DB."""
    db = await get_database()
    existing = await db.get_user_consents(user_id)
    existing_types = {r['consent_type'] for r in existing}

    for consent_type, info in DEFAULT_CONSENTS.items():
        if consent_type not in existing_types:
            await db.upsert_consent(
                user_id, consent_type, False,
                description=info["description"],
                required=info["required"]
            )


def _rows_to_consents_dict(rows: list) -> Dict[str, Any]:
    """Convert DB rows to the nested consents dict for the response."""
    consents = {}
    for r in rows:
        consents[r['consent_type']] = {
            "granted": r.get('granted', False),
            "description": r.get('description', ''),
            "required": r.get('required', False),
            "updated_at": r['updated_at'].isoformat() + "Z" if r.get('updated_at') else None,
        }
    return consents


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}", response_model=ConsentStatus)
async def get_consent(user_id: str):
    """Get all consent statuses for a user."""
    await _ensure_user_consents(user_id)
    db = await get_database()
    rows = await db.get_user_consents(user_id)
    consents = _rows_to_consents_dict(rows)

    updated_times = [c.get("updated_at") for c in consents.values() if c.get("updated_at")]
    last_updated = max(updated_times) if updated_times else None

    return ConsentStatus(user_id=user_id, consents=consents, last_updated=last_updated)


@router.put("/{user_id}", response_model=ConsentStatus)
async def update_consent(user_id: str, update: ConsentUpdate):
    """Update consent statuses for a user."""
    await _ensure_user_consents(user_id)
    db = await get_database()

    for consent_type, granted in update.consents.items():
        desc = DEFAULT_CONSENTS.get(consent_type, {}).get("description", f"Custom consent: {consent_type}")
        req = DEFAULT_CONSENTS.get(consent_type, {}).get("required", False)
        await db.upsert_consent(user_id, consent_type, granted, description=desc, required=req)

    logger.info(f"Consent updated for user ***{user_id[-4:] if len(user_id) > 4 else '****'}: {list(update.consents.keys())}")

    rows = await db.get_user_consents(user_id)
    consents = _rows_to_consents_dict(rows)
    updated_times = [c.get("updated_at") for c in consents.values() if c.get("updated_at")]
    last_updated = max(updated_times) if updated_times else None

    return ConsentStatus(user_id=user_id, consents=consents, last_updated=last_updated)


@router.delete("/{user_id}/{consent_type}")
async def revoke_consent(user_id: str, consent_type: str):
    """Revoke a specific consent type."""
    await _ensure_user_consents(user_id)
    db = await get_database()

    rows = await db.get_user_consents(user_id)
    consent_map = {r['consent_type']: r for r in rows}

    if consent_type not in consent_map:
        raise HTTPException(status_code=404, detail=f"Consent type '{consent_type}' not found")

    if consent_map[consent_type].get("required"):
        raise HTTPException(status_code=400, detail=f"Cannot revoke required consent: {consent_type}")

    await db.revoke_consent(user_id, consent_type)

    logger.info(f"Consent revoked for user ***{user_id[-4:] if len(user_id) > 4 else '****'}: {consent_type}")
    return {"message": f"Consent '{consent_type}' revoked successfully", "user_id": user_id}
