"""
Appointment Routes
==================
Full CRUD backend for the appointment booking system.

Endpoints:
    --- Providers ---
    GET    /appointments/providers                    List/search providers
    GET    /appointments/providers/specialties         Get distinct specialties
    GET    /appointments/providers/{provider_id}       Get single provider
    GET    /appointments/providers/{provider_id}/availability  Get available slots

    --- Appointments ---
    GET    /appointments/{user_id}                     List user appointments
    GET    /appointments/{user_id}/{appointment_id}    Get single appointment
    POST   /appointments/{user_id}                     Book a new appointment
    PUT    /appointments/{user_id}/{appointment_id}    Update an appointment
    POST   /appointments/{user_id}/{appointment_id}/cancel   Cancel appointment
    POST   /appointments/{user_id}/{appointment_id}/complete Mark completed

    --- Insurance ---
    GET    /appointments/{user_id}/insurance            Get user insurance info
    POST   /appointments/{user_id}/insurance            Save/update insurance

    --- Intake / Triage ---
    POST   /appointments/intake/analyze                AI symptom triage
"""

import logging
import uuid
import json
from typing import Optional, List, Any, Dict
from datetime import datetime, date

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("appointments")

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers — database access
# ---------------------------------------------------------------------------

async def _get_db():
    """Lazy-import to avoid circular dependency at import time."""
    from core.database.postgres_db import get_database
    return await get_database()


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

# --- Provider models ---

class ProviderResponse(BaseModel):
    id: Optional[int] = None
    provider_id: str
    name: str
    specialty: str
    qualifications: Optional[str] = None
    rating: float = 0.0
    review_count: int = 0
    photo_url: Optional[str] = Field(None, alias="photoUrl")
    clinic_name: Optional[str] = Field(None, alias="clinicName")
    address: Optional[str] = None
    languages: Optional[List[str]] = None
    telehealth_available: bool = Field(False, alias="telehealthAvailable")
    accepted_insurances: Optional[List[str]] = Field(None, alias="acceptedInsurances")
    bio: Optional[str] = None
    experience_years: int = Field(0, alias="experienceYears")
    accepts_new_patients: bool = Field(True, alias="acceptsNewPatients")

    class Config:
        populate_by_name = True
        from_attributes = True


class AvailabilityResponse(BaseModel):
    provider_id: str
    date: str
    slots: List[str]


# --- Appointment models ---

class AppointmentCreate(BaseModel):
    provider_id: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    appointment_type: str = "in-person"  # 'in-person' | 'video'
    reason: Optional[str] = None
    intake_summary: Optional[str] = None
    shared_chart_data: Optional[Dict[str, Any]] = None
    insurance_provider: Optional[str] = None
    insurance_member_id: Optional[str] = None
    insurance_group_id: Optional[str] = None
    duration_minutes: int = 30
    estimated_cost: float = 150.0


class AppointmentUpdate(BaseModel):
    appointment_type: Optional[str] = None
    reason: Optional[str] = None
    intake_summary: Optional[str] = None
    consultation_summary: Optional[str] = None
    insurance_provider: Optional[str] = None
    insurance_member_id: Optional[str] = None
    insurance_group_id: Optional[str] = None
    status: Optional[str] = None
    actual_cost: Optional[float] = None
    virtual_link: Optional[str] = None
    location: Optional[str] = None


class AppointmentResponse(BaseModel):
    appointment_id: str
    user_id: str
    provider_id: str
    doctor_name: str = Field(alias="doctorName")
    specialty: Optional[str] = None
    doctor_rating: Optional[float] = None
    date: str
    time: str
    duration_minutes: int = 30
    appointment_type: str = Field("in-person", alias="type")
    location: Optional[str] = None
    virtual_link: Optional[str] = None
    reason: Optional[str] = None
    intake_summary: Optional[str] = None
    consultation_summary: Optional[str] = Field(None, alias="summary")
    shared_chart_data: Optional[Dict[str, Any]] = None
    insurance_provider: Optional[str] = None
    insurance_member_id: Optional[str] = None
    insurance_group_id: Optional[str] = None
    status: str = "scheduled"
    cancellation_reason: Optional[str] = None
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        populate_by_name = True
        from_attributes = True


class CancelRequest(BaseModel):
    reason: Optional[str] = None


class CompleteRequest(BaseModel):
    consultation_summary: Optional[str] = None


# --- Insurance models ---

class InsuranceCreate(BaseModel):
    insurance_provider: str
    member_id: str
    group_id: Optional[str] = None
    plan_type: Optional[str] = None


class InsuranceResponse(BaseModel):
    id: Optional[int] = None
    user_id: str
    insurance_provider: str
    member_id: str
    group_id: Optional[str] = None
    plan_type: Optional[str] = None
    is_primary: bool = True
    is_verified: bool = False

    class Config:
        from_attributes = True


# --- Intake models ---

class IntakeRequest(BaseModel):
    symptoms: str
    user_name: Optional[str] = "Patient"


class IntakeResponse(BaseModel):
    urgency: str  # 'emergency' | 'urgent' | 'routine'
    reason: str
    summary: str
    recommendation: str


# ---------------------------------------------------------------------------
# Helper to convert DB row -> response-friendly dict
# ---------------------------------------------------------------------------

def _provider_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a DB row into a dict suitable for ProviderResponse."""
    data = dict(row)
    # Parse JSON strings if needed (handles double-encoded JSON too)
    for field in ('languages', 'accepted_insurances'):
        val = data.get(field)
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                # Handle double-encoded JSON: '"[\"a\",\"b\"]"' -> '["a","b"]' -> ["a","b"]
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                data[field] = parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                data[field] = []
        elif val is None:
            data[field] = []
    # Map snake_case to camelCase aliases
    data['photoUrl'] = data.pop('photo_url', None)
    data['clinicName'] = data.pop('clinic_name', None)
    data['telehealthAvailable'] = data.pop('telehealth_available', False)
    data['acceptedInsurances'] = data.pop('accepted_insurances', [])
    data['experienceYears'] = data.pop('experience_years', 0)
    data['acceptsNewPatients'] = data.pop('accepts_new_patients', True)
    data['reviewCount'] = data.pop('review_count', 0)
    return data


def _appointment_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a DB row into a dict suitable for AppointmentResponse."""
    data = dict(row)
    # Map to frontend aliases
    data['doctorName'] = data.pop('doctor_name', '')
    data['type'] = data.pop('appointment_type', 'in-person')
    data['summary'] = data.get('consultation_summary') or data.get('intake_summary', '')
    # Stringify timestamps
    for ts_field in ('created_at', 'updated_at'):
        if data.get(ts_field) and hasattr(data[ts_field], 'isoformat'):
            data[ts_field] = data[ts_field].isoformat()
    # Parse shared_chart_data
    if isinstance(data.get('shared_chart_data'), str):
        try:
            data['shared_chart_data'] = json.loads(data['shared_chart_data'])
        except (json.JSONDecodeError, TypeError):
            data['shared_chart_data'] = None
    return data


# ===========================================================================
# PROVIDER ENDPOINTS
# ===========================================================================

@router.get("/providers", response_model=List[ProviderResponse])
async def list_providers(
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    search: Optional[str] = Query(None, description="Search by name or specialty"),
    telehealth: Optional[bool] = Query(None, description="Filter telehealth-available"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List healthcare providers with optional filters."""
    db = await _get_db()
    rows = await db.get_providers(
        specialty=specialty,
        search=search,
        telehealth=telehealth,
        limit=limit,
        offset=offset,
    )
    return [_provider_row_to_dict(r) for r in rows]


@router.get("/providers/specialties")
async def list_specialties():
    """Get distinct provider specialties."""
    db = await _get_db()
    specialties = await db.get_provider_specialties()
    return {"specialties": ["All"] + specialties}


@router.get("/providers/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: str):
    """Get a single provider by ID."""
    db = await _get_db()
    row = await db.get_provider_by_id(provider_id)
    if not row:
        raise HTTPException(status_code=404, detail="Provider not found")
    return _provider_row_to_dict(row)


@router.get("/providers/{provider_id}/availability", response_model=AvailabilityResponse)
async def get_availability(
    provider_id: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
):
    """Get available time slots for a provider on a specific date."""
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    db = await _get_db()
    # Verify provider exists
    provider = await db.get_provider_by_id(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    slots = await db.get_provider_availability(provider_id, date)
    return AvailabilityResponse(provider_id=provider_id, date=date, slots=slots)


# ===========================================================================
# APPOINTMENT ENDPOINTS
# ===========================================================================

# ===========================================================================
# INSURANCE ENDPOINTS (must be before /{user_id}/{appointment_id} to avoid
# "insurance" being matched as appointment_id)
# ===========================================================================

@router.get("/{user_id}/insurance", response_model=List[InsuranceResponse])
async def get_insurance(user_id: str):
    """Get insurance information for a user."""
    db = await _get_db()
    rows = await db.get_user_insurance(user_id)
    return rows


@router.post("/{user_id}/insurance", response_model=InsuranceResponse, status_code=201)
async def save_insurance(user_id: str, body: InsuranceCreate):
    """Save or update insurance information."""
    db = await _get_db()
    data = {
        'user_id': user_id,
        'insurance_provider': body.insurance_provider,
        'member_id': body.member_id,
        'group_id': body.group_id,
        'plan_type': body.plan_type,
        'is_verified': False,
    }
    result = await db.save_insurance(data)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to save insurance info")
    return result


@router.get("/{user_id}", response_model=List[AppointmentResponse])
async def list_appointments(
    user_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    upcoming: bool = Query(False, description="Only upcoming appointments"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get all appointments for a user."""
    db = await _get_db()
    rows = await db.get_user_appointments(
        user_id=user_id,
        status=status,
        upcoming_only=upcoming,
        limit=limit,
        offset=offset,
    )
    return [_appointment_row_to_dict(r) for r in rows]


@router.get("/{user_id}/{appointment_id}", response_model=AppointmentResponse)
async def get_appointment(user_id: str, appointment_id: str):
    """Get a single appointment."""
    db = await _get_db()
    row = await db.get_appointment_by_id(appointment_id)
    if not row or row.get('user_id') != user_id:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return _appointment_row_to_dict(row)


@router.post("/{user_id}", response_model=AppointmentResponse, status_code=201)
async def create_appointment(user_id: str, body: AppointmentCreate):
    """Book a new appointment."""
    db = await _get_db()

    # 1. Verify provider exists
    provider = await db.get_provider_by_id(body.provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # 2. Check video type is allowed for this provider
    if body.appointment_type == 'video' and not provider.get('telehealth_available', False):
        raise HTTPException(status_code=400, detail="This provider does not offer video consultations")

    # 3. Check time slot availability
    available_slots = await db.get_provider_availability(body.provider_id, body.date)
    if body.time not in available_slots:
        raise HTTPException(status_code=409, detail=f"Time slot {body.time} is not available on {body.date}")

    # 4. Generate unique appointment ID
    appointment_id = f"apt_{int(datetime.now().timestamp() * 1000)}"

    # 5. Build virtual link for video appointments
    virtual_link = None
    if body.appointment_type == 'video':
        virtual_link = f"https://meet.cardioai.com/{appointment_id}"

    # 6. Create appointment record
    appt_data = {
        'appointment_id': appointment_id,
        'user_id': user_id,
        'provider_id': body.provider_id,
        'doctor_name': provider['name'],
        'specialty': provider.get('specialty'),
        'doctor_rating': provider.get('rating'),
        'date': body.date,
        'time': body.time,
        'duration_minutes': body.duration_minutes,
        'appointment_type': body.appointment_type,
        'location': provider.get('clinic_name', '') + ', ' + (provider.get('address', '') or ''),
        'virtual_link': virtual_link,
        'reason': body.reason,
        'intake_summary': body.intake_summary,
        'shared_chart_data': body.shared_chart_data,
        'insurance_provider': body.insurance_provider,
        'insurance_member_id': body.insurance_member_id,
        'insurance_group_id': body.insurance_group_id,
        'estimated_cost': body.estimated_cost,
        'status': 'scheduled',
    }

    result = await db.create_appointment(appt_data)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create appointment")

    logger.info(f"Appointment {appointment_id} created for user {user_id} with provider {body.provider_id}")
    return _appointment_row_to_dict(result)


@router.put("/{user_id}/{appointment_id}", response_model=AppointmentResponse)
async def update_appointment(user_id: str, appointment_id: str, body: AppointmentUpdate):
    """Update an existing appointment."""
    db = await _get_db()

    # Verify ownership
    existing = await db.get_appointment_by_id(appointment_id)
    if not existing or existing.get('user_id') != user_id:
        raise HTTPException(status_code=404, detail="Appointment not found")

    if existing.get('status') in ('cancelled', 'completed'):
        raise HTTPException(status_code=400, detail=f"Cannot update a {existing['status']} appointment")

    updates = body.dict(exclude_none=True)
    result = await db.update_appointment(appointment_id, updates)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to update appointment")

    logger.info(f"Appointment {appointment_id} updated")
    return _appointment_row_to_dict(result)


@router.post("/{user_id}/{appointment_id}/cancel", response_model=AppointmentResponse)
async def cancel_appointment(user_id: str, appointment_id: str, body: CancelRequest = None):
    """Cancel an appointment and release the time slot."""
    db = await _get_db()

    existing = await db.get_appointment_by_id(appointment_id)
    if not existing or existing.get('user_id') != user_id:
        raise HTTPException(status_code=404, detail="Appointment not found")

    if existing.get('status') == 'cancelled':
        raise HTTPException(status_code=400, detail="Appointment is already cancelled")

    if existing.get('status') == 'completed':
        raise HTTPException(status_code=400, detail="Cannot cancel a completed appointment")

    reason = body.reason if body else None
    result = await db.cancel_appointment(appointment_id, reason)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to cancel appointment")

    logger.info(f"Appointment {appointment_id} cancelled")
    return _appointment_row_to_dict(result)


@router.post("/{user_id}/{appointment_id}/complete", response_model=AppointmentResponse)
async def complete_appointment(user_id: str, appointment_id: str, body: CompleteRequest = None):
    """Mark an appointment as completed with optional clinical summary."""
    db = await _get_db()

    existing = await db.get_appointment_by_id(appointment_id)
    if not existing or existing.get('user_id') != user_id:
        raise HTTPException(status_code=404, detail="Appointment not found")

    if existing.get('status') == 'cancelled':
        raise HTTPException(status_code=400, detail="Cannot complete a cancelled appointment")

    summary = body.consultation_summary if body else None
    result = await db.complete_appointment(appointment_id, summary)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to complete appointment")

    logger.info(f"Appointment {appointment_id} completed")
    return _appointment_row_to_dict(result)


# ===========================================================================
# INTAKE / TRIAGE ENDPOINT
# ===========================================================================

@router.post("/intake/analyze", response_model=IntakeResponse)
async def analyze_intake(body: IntakeRequest):
    """AI-powered symptom triage before booking.

    Analyzes symptoms to determine urgency level and provide recommendations.
    In production, this would use a medical NLP model. Currently uses keyword matching.
    """
    symptoms_lower = body.symptoms.lower()

    # Emergency keywords
    emergency_keywords = [
        'chest pain', 'heart attack', 'stroke', 'can\'t breathe',
        'difficulty breathing', 'severe bleeding', 'unconscious',
        'seizure', 'loss of consciousness', 'crushing chest',
        'sudden numbness', 'sudden confusion', 'severe headache',
    ]

    # Urgent keywords
    urgent_keywords = [
        'high fever', 'blood in', 'severe pain', 'persistent vomiting',
        'shortness of breath', 'rapid heartbeat', 'palpitations',
        'fainting', 'dizziness severe', 'swelling legs',
    ]

    urgency = 'routine'
    recommendation = 'You can proceed to book a regular appointment.'

    for kw in emergency_keywords:
        if kw in symptoms_lower:
            urgency = 'emergency'
            recommendation = (
                'Your symptoms suggest a potentially life-threatening condition. '
                'Please call emergency services (911) immediately. '
                'Do not wait for an appointment.'
            )
            break

    if urgency == 'routine':
        for kw in urgent_keywords:
            if kw in symptoms_lower:
                urgency = 'urgent'
                recommendation = (
                    'Your symptoms suggest you should be seen soon. '
                    'Consider booking the earliest available slot or visiting urgent care.'
                )
                break

    return IntakeResponse(
        urgency=urgency,
        reason=body.symptoms,
        summary=f"Patient reports: {body.symptoms}",
        recommendation=recommendation,
    )
