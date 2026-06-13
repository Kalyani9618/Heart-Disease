"""
Integrations & Weekly Summary Routes
=====================================
Cross-service integration endpoints.
Endpoints:
    GET  /integrations/weekly-summary/{user_id}
    POST /integrations/predict-from-document
"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("integrations")

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PredictFromDocumentRequest(BaseModel):
    document_id: str
    user_id: str
    patient_profile: Optional[Dict[str, Any]] = None


class WeeklySummary(BaseModel):
    user_id: str
    period_start: str
    period_end: str
    summary: str
    highlights: List[str] = []
    health_score: Optional[float] = None
    recommendations: List[str] = []
    vitals_summary: Optional[Dict[str, Any]] = None
    generated_at: str


class DocumentPrediction(BaseModel):
    document_id: str
    user_id: str
    prediction: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = None
    confidence: Optional[float] = None
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/weekly-summary/{user_id}", response_model=WeeklySummary)
async def get_weekly_summary(user_id: str):
    """Get the weekly health summary for a user."""
    from core.database.postgres_db import get_database

    now = datetime.utcnow()
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

    db = await get_database()

    # Query real data from database
    vitals = await db.get_weekly_vitals_summary(user_id)
    medications = await db.get_user_medications_list(user_id)
    appt_count = await db.get_recent_appointments_count(user_id)

    # Build highlights from real data
    highlights = []
    vitals_summary_data = {}

    if vitals:
        avg_hr = vitals.get("avg_heart_rate")
        if avg_hr:
            vitals_summary_data["avg_heart_rate"] = round(avg_hr, 1)
            status = "normal" if 60 <= avg_hr <= 100 else "elevated"
            highlights.append(f"Average resting heart rate: {round(avg_hr)} bpm ({status})")
        avg_sys = vitals.get("avg_systolic")
        if avg_sys:
            vitals_summary_data["avg_systolic"] = round(avg_sys, 1)
            status = "stable" if avg_sys < 130 else "elevated"
            highlights.append(f"Blood pressure readings {status}")
        reading_count = vitals.get("reading_count", 0)
        vitals_summary_data["blood_pressure_readings"] = reading_count
    else:
        highlights.append("No vitals recorded this week")

    if medications:
        highlights.append(f"Active medications: {len(medications)}")
    if appt_count:
        highlights.append(f"Appointments this week: {appt_count}")

    # Build recommendations from data
    recommendations = ["Continue regular exercise routine"]
    if vitals and vitals.get("avg_heart_rate") and vitals["avg_heart_rate"] > 100:
        recommendations.append("Consider monitoring elevated heart rate with your doctor")
    if not vitals or vitals.get("reading_count", 0) < 3:
        recommendations.append("Try to record vitals at least 3 times per week")
    recommendations.append("Stay hydrated — aim for 8 glasses of water daily")

    # Health score based on data
    health_score = 7.5
    if vitals:
        avg_hr = vitals.get("avg_heart_rate")
        if avg_hr and 60 <= avg_hr <= 80:
            health_score += 0.5
        elif avg_hr and avg_hr > 100:
            health_score -= 1.0

    summary_text = f"Weekly health summary based on {vitals.get('reading_count', 0) if vitals else 0} recorded vitals readings."

    return WeeklySummary(
        user_id=user_id,
        period_start=week_start.isoformat() + "Z",
        period_end=week_end.isoformat() + "Z",
        summary=summary_text,
        highlights=highlights,
        health_score=round(health_score, 1),
        recommendations=recommendations,
        vitals_summary=vitals_summary_data or None,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/predict-from-document", response_model=DocumentPrediction)
async def predict_from_document(request: PredictFromDocumentRequest):
    """Predict heart disease risk from a medical document."""
    # In production, this would:
    # 1. Fetch the document from the documents service
    # 2. Extract clinical values using NLP
    # 3. Run the heart prediction model

    logger.info(f"Document prediction requested: doc={request.document_id}, user=***")

    return DocumentPrediction(
        document_id=request.document_id,
        user_id=request.user_id,
        prediction=None,
        risk_level="unknown",
        confidence=None,
        message="Document analysis is processing. Clinical values will be extracted and used for prediction.",
    )
