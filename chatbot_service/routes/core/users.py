"""
User Medications Routes (PostgreSQL-backed)
============================================
CRUD endpoints for user medication management backed by PostgreSQL.

Endpoints:
    GET    /users/{user_id}/medications
    POST   /users/{user_id}/medications
    PUT    /users/{user_id}/medications/{medication_id}
    DELETE /users/{user_id}/medications/{medication_id}
"""

import logging
import json
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("users")

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MedicationBase(BaseModel):
    name: str
    dosage: str
    schedule: List[str] = Field(default_factory=list, description="e.g. ['08:00', '20:00']")
    frequency: str = "daily"
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    notes: Optional[str] = None
    quantity: Optional[int] = 30
    instructions: Optional[str] = None
    times: List[str] = Field(default_factory=lambda: ["08:00"])
    takenToday: List[bool] = Field(default_factory=list)


class MedicationCreate(MedicationBase):
    pass


class MedicationUpdate(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    schedule: Optional[List[str]] = None
    frequency: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    notes: Optional[str] = None
    quantity: Optional[int] = None
    instructions: Optional[str] = None
    times: Optional[List[str]] = None
    takenToday: Optional[List[bool]] = None


class MedicationResponse(MedicationBase):
    id: str


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


def _row_to_response(row: dict) -> dict:
    """Convert a database row to a MedicationResponse dict."""
    schedule = row.get("schedule") or []
    if isinstance(schedule, str):
        schedule = json.loads(schedule)

    times = row.get("times") or ["08:00"]
    if isinstance(times, str):
        times = json.loads(times)

    taken_today = row.get("taken_today") or []
    if isinstance(taken_today, str):
        taken_today = json.loads(taken_today)

    # Ensure takenToday has same length as times
    while len(taken_today) < len(times):
        taken_today.append(False)

    return {
        "id": str(row["id"]),
        "name": row["drug_name"],
        "dosage": row.get("dosage") or "",
        "schedule": schedule,
        "frequency": row.get("frequency") or "daily",
        "startDate": row["start_date"].strftime("%Y-%m-%d") if row.get("start_date") else None,
        "endDate": row["end_date"].strftime("%Y-%m-%d") if row.get("end_date") else None,
        "notes": row.get("notes"),
        "quantity": row.get("quantity") or 30,
        "instructions": row.get("instructions"),
        "times": times,
        "takenToday": taken_today,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}/medications", response_model=List[MedicationResponse])
async def get_medications(user_id: str):
    """Get all medications for a user from PostgreSQL."""
    db = await _get_db()
    try:
        rows = await db.fetch_all(
            "SELECT * FROM medications WHERE user_id = $1 AND is_active = TRUE ORDER BY created_at DESC",
            (user_id,)
        )
        return [_row_to_response(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get medications for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve medications")


@router.post("/{user_id}/medications", response_model=MedicationResponse, status_code=201)
async def add_medication(user_id: str, medication: MedicationCreate):
    """Add a new medication for a user in PostgreSQL."""
    db = await _get_db()

    start_date = None
    if medication.startDate:
        try:
            start_date = datetime.strptime(medication.startDate, "%Y-%m-%d")
        except ValueError:
            start_date = datetime.utcnow()
    else:
        start_date = datetime.utcnow()

    end_date = None
    if medication.endDate:
        try:
            end_date = datetime.strptime(medication.endDate, "%Y-%m-%d")
        except ValueError:
            pass

    # Ensure takenToday has same length as times
    taken_today = medication.takenToday or []
    while len(taken_today) < len(medication.times):
        taken_today.append(False)

    try:
        row = await db.execute_query(
            """
            INSERT INTO medications
                (user_id, drug_name, dosage, frequency, start_date, end_date,
                 is_active, schedule, notes, quantity, instructions, times, taken_today)
            VALUES ($1, $2, $3, $4, $5, $6, TRUE, $7, $8, $9, $10, $11, $12)
            RETURNING *
            """,
            (
                user_id,
                medication.name,
                medication.dosage,
                medication.frequency,
                start_date,
                end_date,
                json.dumps(medication.schedule),
                medication.notes,
                medication.quantity or 30,
                medication.instructions,
                json.dumps(medication.times),
                json.dumps(taken_today),
            ),
            fetch_one=True,
        )

        logger.info(f"Medication added for user {user_id}: {medication.name}")
        return _row_to_response(row)
    except Exception as e:
        logger.error(f"Failed to add medication for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add medication")


@router.put("/{user_id}/medications/{medication_id}", response_model=MedicationResponse)
async def update_medication(user_id: str, medication_id: str, update: MedicationUpdate):
    """Update an existing medication in PostgreSQL."""
    db = await _get_db()

    # Build dynamic update query
    set_clauses = []
    params = []
    param_idx = 1

    field_map = {
        "name": "drug_name",
        "dosage": "dosage",
        "frequency": "frequency",
        "notes": "notes",
        "quantity": "quantity",
        "instructions": "instructions",
    }

    update_data = update.dict(exclude_unset=True)

    for field, column in field_map.items():
        if field in update_data and update_data[field] is not None:
            set_clauses.append(f"{column} = ${param_idx}")
            params.append(update_data[field])
            param_idx += 1

    # Handle date fields
    if "startDate" in update_data and update_data["startDate"] is not None:
        set_clauses.append(f"start_date = ${param_idx}")
        try:
            params.append(datetime.strptime(update_data["startDate"], "%Y-%m-%d"))
        except ValueError:
            params.append(None)
        param_idx += 1

    if "endDate" in update_data and update_data["endDate"] is not None:
        set_clauses.append(f"end_date = ${param_idx}")
        try:
            params.append(datetime.strptime(update_data["endDate"], "%Y-%m-%d"))
        except ValueError:
            params.append(None)
        param_idx += 1

    # Handle JSON fields
    json_fields = {"schedule": "schedule", "times": "times", "takenToday": "taken_today"}
    for field, column in json_fields.items():
        if field in update_data and update_data[field] is not None:
            set_clauses.append(f"{column} = ${param_idx}")
            params.append(json.dumps(update_data[field]))
            param_idx += 1

    if not set_clauses:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Add WHERE clause params
    params.append(user_id)
    params.append(int(medication_id) if medication_id.isdigit() else -1)

    query = f"""
        UPDATE medications
        SET {', '.join(set_clauses)}
        WHERE user_id = ${param_idx} AND id = ${param_idx + 1}
        RETURNING *
    """

    try:
        row = await db.execute_query(query, tuple(params), fetch_one=True)
        if not row:
            raise HTTPException(status_code=404, detail=f"Medication {medication_id} not found")

        logger.info(f"Medication updated for user {user_id}: {medication_id}")
        return _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update medication {medication_id} for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update medication")


@router.delete("/{user_id}/medications/{medication_id}")
async def delete_medication(user_id: str, medication_id: str):
    """Soft-delete a medication (set is_active=FALSE)."""
    db = await _get_db()

    med_id = int(medication_id) if medication_id.isdigit() else -1

    try:
        result = await db.execute_query(
            "UPDATE medications SET is_active = FALSE WHERE user_id = $1 AND id = $2 RETURNING id",
            (user_id, med_id),
            fetch_one=True,
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"Medication {medication_id} not found")

        logger.info(f"Medication deleted for user {user_id}: {medication_id}")
        return {"message": f"Medication {medication_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete medication {medication_id} for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete medication")
