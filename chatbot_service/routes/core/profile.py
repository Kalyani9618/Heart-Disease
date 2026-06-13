"""
User Profile Routes (PostgreSQL-backed)
========================================
CRUD endpoints for user profile, emergency contacts, conditions, allergies,
and family members.

Endpoints:
    GET    /profile/{user_id}
    PUT    /profile/{user_id}
    PUT    /profile/{user_id}/avatar
    GET    /profile/{user_id}/emergency-contact
    PUT    /profile/{user_id}/emergency-contact
    POST   /profile/{user_id}/conditions
    DELETE /profile/{user_id}/conditions/{condition}
    POST   /profile/{user_id}/allergies
    DELETE /profile/{user_id}/allergies/{allergy}
    GET    /profile/{user_id}/family
    POST   /profile/{user_id}/family
    DELETE /profile/{user_id}/family/{member_id}
"""

import logging
import json
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("profile")

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EmergencyContactModel(BaseModel):
    name: str = ""
    relation: str = ""
    phone: str = ""


class ProfileResponse(BaseModel):
    id: str
    name: str
    email: str
    phone: str = ""
    dob: str = ""
    gender: str = ""
    conditions: List[str] = []
    allergies: List[str] = []
    medications: List[str] = []
    emergencyContact: EmergencyContactModel = EmergencyContactModel()
    avatar: str = ""


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None


class AvatarUpdate(BaseModel):
    avatar: str


class ItemAdd(BaseModel):
    value: str


class FamilyMemberCreate(BaseModel):
    name: str
    relation: str
    avatar: str = ""
    accessLevel: str = "read-only"
    status: str = "Stable"


class FamilyMemberResponse(BaseModel):
    id: str
    name: str
    relation: str
    avatar: str = ""
    accessLevel: str = "read-only"
    status: str = "Stable"
    lastActive: str = ""


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
# Profile Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """Get a user's full profile from PostgreSQL."""
    db = await _get_db()
    try:
        # Fetch user base info
        user_row = await db.fetch_one(
            "SELECT * FROM users WHERE user_id = $1",
            (user_id,)
        )

        if not user_row:
            # Return a default profile if user not found yet
            return ProfileResponse(
                id=user_id,
                name="",
                email="",
                phone="",
                dob="",
                gender="",
                conditions=[],
                allergies=[],
                medications=[],
                emergencyContact=EmergencyContactModel(),
                avatar=""
            )

        # Fetch emergency contact
        ec_row = await db.fetch_one(
            "SELECT * FROM emergency_contacts WHERE user_id = $1 LIMIT 1",
            (user_id,)
        )

        emergency_contact = EmergencyContactModel()
        if ec_row:
            emergency_contact = EmergencyContactModel(
                name=ec_row.get("contact_name") or "",
                relation=ec_row.get("relation") or "",
                phone=ec_row.get("contact_phone") or ""
            )

        # Parse conditions and allergies from user_preferences or a JSON column
        conditions = []
        allergies = []
        medications_list = []

        # Try to get from user_preferences table
        prefs_row = await db.fetch_one(
            "SELECT preferences FROM user_preferences WHERE user_id = $1",
            (user_id,)
        )
        if prefs_row and prefs_row.get("preferences"):
            prefs = prefs_row["preferences"]
            if isinstance(prefs, str):
                prefs = json.loads(prefs)
            conditions = prefs.get("conditions", [])
            allergies = prefs.get("allergies", [])
            medications_list = prefs.get("medications_list", [])

        # Build DOB string
        dob = ""
        if user_row.get("date_of_birth"):
            dob = user_row["date_of_birth"].strftime("%Y-%m-%d") if hasattr(user_row["date_of_birth"], "strftime") else str(user_row["date_of_birth"])
        elif user_row.get("dob"):
            dob = str(user_row["dob"])

        return ProfileResponse(
            id=user_id,
            name=user_row.get("name") or user_row.get("username") or "",
            email=user_row.get("email") or "",
            phone=user_row.get("phone") or "",
            dob=dob,
            gender=user_row.get("gender") or "",
            conditions=conditions,
            allergies=allergies,
            medications=medications_list,
            emergencyContact=emergency_contact,
            avatar=user_row.get("avatar") or ""
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.put("/{user_id}", response_model=ProfileResponse)
async def update_profile(user_id: str, update: ProfileUpdate):
    """Update a user's basic profile info in PostgreSQL."""
    db = await _get_db()

    set_clauses = []
    params = []
    param_idx = 1

    field_map = {
        "name": "name",
        "email": "email",
        "phone": "phone",
        "gender": "gender",
    }

    update_data = update.dict(exclude_unset=True)

    for field, column in field_map.items():
        if field in update_data and update_data[field] is not None:
            set_clauses.append(f"{column} = ${param_idx}")
            params.append(update_data[field])
            param_idx += 1

    if "dob" in update_data and update_data["dob"] is not None:
        set_clauses.append(f"date_of_birth = ${param_idx}")
        try:
            params.append(datetime.strptime(update_data["dob"], "%Y-%m-%d"))
        except ValueError:
            params.append(None)
        param_idx += 1

    # Always update updated_at
    set_clauses.append(f"updated_at = ${param_idx}")
    params.append(datetime.utcnow())
    param_idx += 1

    if not set_clauses:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(user_id)

    try:
        await db.execute_query(
            f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = ${param_idx}",
            tuple(params),
        )
    except Exception as e:
        logger.error(f"Failed to update profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

    # Return updated profile
    return await get_profile(user_id)


@router.put("/{user_id}/avatar")
async def update_avatar(user_id: str, body: AvatarUpdate):
    """Update user's avatar URL."""
    db = await _get_db()
    try:
        await db.execute_query(
            "UPDATE users SET avatar = $1, updated_at = $2 WHERE user_id = $3",
            (body.avatar, datetime.utcnow(), user_id),
        )
        return {"message": "Avatar updated", "avatar": body.avatar}
    except Exception as e:
        logger.error(f"Failed to update avatar for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update avatar")


# ---------------------------------------------------------------------------
# Emergency Contact Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}/emergency-contact", response_model=EmergencyContactModel)
async def get_emergency_contact(user_id: str):
    """Get the user's emergency contact."""
    db = await _get_db()
    try:
        row = await db.fetch_one(
            "SELECT * FROM emergency_contacts WHERE user_id = $1 LIMIT 1",
            (user_id,)
        )
        if not row:
            return EmergencyContactModel()
        return EmergencyContactModel(
            name=row.get("contact_name") or "",
            relation=row.get("relation") or "",
            phone=row.get("contact_phone") or ""
        )
    except Exception as e:
        logger.error(f"Failed to get emergency contact for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get emergency contact")


@router.put("/{user_id}/emergency-contact", response_model=EmergencyContactModel)
async def update_emergency_contact(user_id: str, contact: EmergencyContactModel):
    """Create or update the user's emergency contact (upsert)."""
    db = await _get_db()
    try:
        existing = await db.fetch_one(
            "SELECT id FROM emergency_contacts WHERE user_id = $1 LIMIT 1",
            (user_id,)
        )

        if existing:
            await db.execute_query(
                """UPDATE emergency_contacts
                   SET contact_name = $1, relation = $2, contact_phone = $3, updated_at = $4
                   WHERE user_id = $5""",
                (contact.name, contact.relation, contact.phone, datetime.utcnow(), user_id),
            )
        else:
            await db.execute_query(
                """INSERT INTO emergency_contacts (user_id, contact_name, relation, contact_phone)
                   VALUES ($1, $2, $3, $4)""",
                (user_id, contact.name, contact.relation, contact.phone),
            )

        return contact
    except Exception as e:
        logger.error(f"Failed to update emergency contact for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update emergency contact")


# ---------------------------------------------------------------------------
# Conditions / Allergies Endpoints
# ---------------------------------------------------------------------------

async def _get_medical_lists(db, user_id: str) -> dict:
    """Get conditions, allergies, medications_list from user_preferences."""
    row = await db.fetch_one(
        "SELECT preferences FROM user_preferences WHERE user_id = $1",
        (user_id,)
    )
    if row and row.get("preferences"):
        prefs = row["preferences"]
        if isinstance(prefs, str):
            prefs = json.loads(prefs)
        return prefs
    return {}


async def _save_medical_lists(db, user_id: str, prefs: dict):
    """Save conditions, allergies, medications_list to user_preferences."""
    existing = await db.fetch_one(
        "SELECT id FROM user_preferences WHERE user_id = $1",
        (user_id,)
    )

    prefs_json = json.dumps(prefs)

    if existing:
        await db.execute_query(
            "UPDATE user_preferences SET preferences = $1, updated_at = $2 WHERE user_id = $3",
            (prefs_json, datetime.utcnow(), user_id),
        )
    else:
        await db.execute_query(
            "INSERT INTO user_preferences (user_id, preferences) VALUES ($1, $2)",
            (user_id, prefs_json),
        )


@router.post("/{user_id}/conditions")
async def add_condition(user_id: str, item: ItemAdd):
    """Add a medical condition to the user's profile."""
    db = await _get_db()
    try:
        prefs = await _get_medical_lists(db, user_id)
        conditions = prefs.get("conditions", [])
        if item.value not in conditions:
            conditions.append(item.value)
        prefs["conditions"] = conditions
        await _save_medical_lists(db, user_id, prefs)
        return {"conditions": conditions}
    except Exception as e:
        logger.error(f"Failed to add condition for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add condition")


@router.delete("/{user_id}/conditions/{condition}")
async def remove_condition(user_id: str, condition: str):
    """Remove a medical condition from the user's profile."""
    db = await _get_db()
    try:
        prefs = await _get_medical_lists(db, user_id)
        conditions = prefs.get("conditions", [])
        conditions = [c for c in conditions if c != condition]
        prefs["conditions"] = conditions
        await _save_medical_lists(db, user_id, prefs)
        return {"conditions": conditions}
    except Exception as e:
        logger.error(f"Failed to remove condition for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove condition")


@router.post("/{user_id}/allergies")
async def add_allergy(user_id: str, item: ItemAdd):
    """Add an allergy to the user's profile."""
    db = await _get_db()
    try:
        prefs = await _get_medical_lists(db, user_id)
        allergies = prefs.get("allergies", [])
        if item.value not in allergies:
            allergies.append(item.value)
        prefs["allergies"] = allergies
        await _save_medical_lists(db, user_id, prefs)
        return {"allergies": allergies}
    except Exception as e:
        logger.error(f"Failed to add allergy for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add allergy")


@router.delete("/{user_id}/allergies/{allergy}")
async def remove_allergy(user_id: str, allergy: str):
    """Remove an allergy from the user's profile."""
    db = await _get_db()
    try:
        prefs = await _get_medical_lists(db, user_id)
        allergies = prefs.get("allergies", [])
        allergies = [a for a in allergies if a != allergy]
        prefs["allergies"] = allergies
        await _save_medical_lists(db, user_id, prefs)
        return {"allergies": allergies}
    except Exception as e:
        logger.error(f"Failed to remove allergy for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove allergy")


# ---------------------------------------------------------------------------
# Family Members Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}/family", response_model=List[FamilyMemberResponse])
async def get_family_members(user_id: str):
    """Get all family members for a user."""
    db = await _get_db()
    try:
        rows = await db.fetch_all(
            "SELECT * FROM family_members WHERE user_id = $1 ORDER BY created_at DESC",
            (user_id,)
        )
        result = []
        for row in rows:
            result.append(FamilyMemberResponse(
                id=str(row["id"]),
                name=row.get("name") or "",
                relation=row.get("relation") or "",
                avatar=row.get("avatar") or "",
                accessLevel=row.get("access_level") or "read-only",
                status=row.get("status") or "Stable",
                lastActive=row["updated_at"].strftime("%Y-%m-%d %H:%M") if row.get("updated_at") else ""
            ))
        return result
    except Exception as e:
        logger.error(f"Failed to get family members for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get family members")


@router.post("/{user_id}/family", response_model=FamilyMemberResponse, status_code=201)
async def add_family_member(user_id: str, member: FamilyMemberCreate):
    """Add a family member."""
    db = await _get_db()
    try:
        row = await db.execute_query(
            """INSERT INTO family_members (user_id, name, relation, avatar, access_level, status)
               VALUES ($1, $2, $3, $4, $5, $6)
               RETURNING *""",
            (user_id, member.name, member.relation, member.avatar, member.accessLevel, member.status),
            fetch_one=True,
        )
        return FamilyMemberResponse(
            id=str(row["id"]),
            name=row.get("name") or "",
            relation=row.get("relation") or "",
            avatar=row.get("avatar") or "",
            accessLevel=row.get("access_level") or "read-only",
            status=row.get("status") or "Stable",
            lastActive=""
        )
    except Exception as e:
        logger.error(f"Failed to add family member for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add family member")


@router.delete("/{user_id}/family/{member_id}")
async def remove_family_member(user_id: str, member_id: str):
    """Remove a family member."""
    db = await _get_db()
    mid = int(member_id) if member_id.isdigit() else -1
    try:
        result = await db.execute_query(
            "DELETE FROM family_members WHERE user_id = $1 AND id = $2 RETURNING id",
            (user_id, mid),
            fetch_one=True,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Family member not found")
        return {"message": "Family member removed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove family member for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove family member")
