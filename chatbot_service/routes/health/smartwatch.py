"""
Smartwatch Routes
=================
Wearable device integration for vitals ingestion and health analysis.
Persisted to PostgreSQL (user_devices, vitals, device_timeseries tables).
Endpoints:
    POST /smartwatch/register
    POST /smartwatch/vitals/ingest
    GET  /smartwatch/vitals/{device_id}/aggregated
    POST /smartwatch/analyze
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.database.postgres_db import get_database

logger = logging.getLogger("smartwatch")

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DeviceRegistration(BaseModel):
    user_id: str
    device_type: str = Field(..., description="e.g. apple_watch, fitbit, garmin")
    device_name: Optional[str] = None
    firmware_version: Optional[str] = None


class VitalsPayload(BaseModel):
    device_id: str
    user_id: str
    metrics: List[Dict[str, Any]] = Field(..., description="List of {metric_type, value, unit, timestamp}")


class AggregatedVitals(BaseModel):
    device_id: str
    metric_type: str
    interval: str
    data_points: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    latest_value: Optional[float] = None
    unit: Optional[str] = None


class HealthAnalysisRequest(BaseModel):
    user_id: str
    device_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    time_range_hours: int = 24


class HealthAnalysisResponse(BaseModel):
    user_id: str
    summary: str
    risk_indicators: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    analyzed_at: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register")
async def register_device(device: DeviceRegistration):
    """Register a new smartwatch/wearable device."""
    device_id = str(uuid.uuid4())
    db = await get_database()
    await db.register_device(
        device_id=device_id,
        user_id=device.user_id,
        device_type=device.device_type,
        device_name=device.device_name or device.device_type,
        firmware_version=device.firmware_version,
    )
    masked_user = f"***{device.user_id[-4:]}" if len(device.user_id) > 4 else "****"
    logger.info(f"Smartwatch registered: {device_id} ({device.device_type}) for user {masked_user}")
    return {
        "device_id": device_id,
        "status": "registered",
        "message": f"{device.device_type} registered successfully",
    }


@router.post("/vitals/ingest")
async def ingest_vitals(payload: VitalsPayload):
    """Ingest vitals data from a smartwatch."""
    if not payload.metrics:
        raise HTTPException(status_code=400, detail="No metrics provided")

    db = await get_database()
    count = 0
    for metric in payload.metrics:
        metric_type = metric.get("metric_type", "unknown")
        value = metric.get("value")
        if value is None:
            continue
        try:
            value_float = float(value)
        except (ValueError, TypeError):
            continue
        unit = metric.get("unit", "")
        timestamp = metric.get("timestamp")
        await db.store_device_timeseries(
            device_id=payload.device_id,
            user_id=payload.user_id,
            metric_type=metric_type,
            value=value_float,
            unit=unit,
            timestamp=timestamp,
        )
        count += 1

    logger.info(f"Ingested {count} vitals from device {payload.device_id}")
    return {"ingested": count, "device_id": payload.device_id, "status": "ok"}


@router.get("/vitals/{device_id}/aggregated", response_model=AggregatedVitals)
async def get_aggregated_vitals(
    device_id: str,
    metric_type: str = Query(..., description="e.g. heart_rate, spo2, steps"),
    interval: str = Query("1h", description="Aggregation interval: 1h, 6h, 24h, 7d"),
):
    """Get aggregated vitals for a device and metric type."""
    interval_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}
    hours = interval_map.get(interval, 24)

    db = await get_database()
    records = await db.get_device_timeseries(device_id, metric_type, hours)

    if not records:
        return AggregatedVitals(
            device_id=device_id,
            metric_type=metric_type,
            interval=interval,
            data_points=0,
        )

    values = []
    unit = ""
    for r in records:
        if r.get("value") is not None:
            try:
                values.append(float(r["value"]))
            except (ValueError, TypeError):
                pass
        if not unit and r.get("unit"):
            unit = r["unit"]

    return AggregatedVitals(
        device_id=device_id,
        metric_type=metric_type,
        interval=interval,
        data_points=len(values),
        min_value=round(min(values), 2) if values else None,
        max_value=round(max(values), 2) if values else None,
        avg_value=round(sum(values) / len(values), 2) if values else None,
        latest_value=values[-1] if values else None,
        unit=unit,
    )


@router.post("/analyze", response_model=HealthAnalysisResponse)
async def analyze_health(request: HealthAnalysisRequest):
    """Analyze smartwatch health data for a user."""
    risk_indicators = []
    recommendations = []

    # Check provided metrics
    if request.metrics:
        hr = request.metrics.get("heart_rate")
        if hr is not None:
            if hr > 100:
                risk_indicators.append({"metric": "heart_rate", "value": hr, "status": "elevated", "severity": "moderate"})
                recommendations.append("Your resting heart rate is elevated. Consider relaxation techniques and consult your doctor.")
            elif hr < 50:
                risk_indicators.append({"metric": "heart_rate", "value": hr, "status": "low", "severity": "moderate"})
                recommendations.append("Your heart rate is below normal. If you're not an athlete, consult your doctor.")
            elif 50 <= hr < 60:
                recommendations.append(f"Heart rate of {hr} BPM is on the lower end of normal. Fine if you're athletic.")
            elif 60 <= hr <= 80:
                recommendations.append(f"Heart rate of {hr} BPM is in the healthy range.")
            elif 80 < hr <= 100:
                recommendations.append(f"Heart rate of {hr} BPM is in the normal range but slightly elevated.")

        spo2 = request.metrics.get("spo2")
        if spo2 is not None:
            if spo2 < 90:
                risk_indicators.append({"metric": "spo2", "value": spo2, "status": "critical", "severity": "critical"})
                recommendations.append("⚠️ Blood oxygen below 90% is critically low. Seek immediate medical attention.")
            elif spo2 < 95:
                risk_indicators.append({"metric": "spo2", "value": spo2, "status": "low", "severity": "high"})
                recommendations.append("Your blood oxygen level is below 95%. Seek medical attention if symptoms persist.")
            elif spo2 >= 95:
                recommendations.append(f"SpO2 of {spo2}% is within normal range.")

        steps = request.metrics.get("steps")
        if steps is not None:
            if steps < 3000:
                recommendations.append(f"You've logged {steps} steps today. Try to be more active — aim for at least 7,000-10,000 steps.")
            elif steps < 7000:
                recommendations.append(f"You've logged {steps} steps today. Good effort! Try to reach 10,000 for optimal cardiovascular health.")
            else:
                recommendations.append(f"Excellent! {steps} steps today shows great activity levels.")

        # Blood Pressure Analysis
        bp_systolic = request.metrics.get("bp_systolic")
        bp_diastolic = request.metrics.get("bp_diastolic")
        if bp_systolic is not None and bp_diastolic is not None:
            if bp_systolic >= 180 or bp_diastolic >= 120:
                risk_indicators.append({"metric": "blood_pressure", "value": f"{bp_systolic}/{bp_diastolic}", "status": "hypertensive_crisis", "severity": "critical"})
                recommendations.append("⚠️ Hypertensive crisis detected. Seek emergency medical care immediately.")
            elif bp_systolic >= 140 or bp_diastolic >= 90:
                risk_indicators.append({"metric": "blood_pressure", "value": f"{bp_systolic}/{bp_diastolic}", "status": "high", "severity": "high"})
                recommendations.append("Your blood pressure is in the high range (Stage 2 hypertension). Consult your healthcare provider.")
            elif bp_systolic >= 130 or bp_diastolic >= 80:
                risk_indicators.append({"metric": "blood_pressure", "value": f"{bp_systolic}/{bp_diastolic}", "status": "elevated", "severity": "moderate"})
                recommendations.append("Your blood pressure is elevated (Stage 1 hypertension). Consider lifestyle changes and consult your doctor.")
            elif bp_systolic < 90 or bp_diastolic < 60:
                risk_indicators.append({"metric": "blood_pressure", "value": f"{bp_systolic}/{bp_diastolic}", "status": "low", "severity": "moderate"})
                recommendations.append("Your blood pressure is lower than normal. Stay hydrated and consult your doctor if you feel dizzy.")
            else:
                recommendations.append(f"Blood pressure {bp_systolic}/{bp_diastolic} mmHg is within normal range.")

        # Temperature Analysis
        temperature = request.metrics.get("temperature")
        if temperature is not None:
            if temperature >= 39.0:
                risk_indicators.append({"metric": "temperature", "value": temperature, "status": "high_fever", "severity": "high"})
                recommendations.append("High fever detected. Monitor closely and seek medical attention if it persists.")
            elif temperature >= 37.5:
                risk_indicators.append({"metric": "temperature", "value": temperature, "status": "mild_fever", "severity": "moderate"})
                recommendations.append("Mild fever detected. Rest, stay hydrated, and monitor your temperature.")
            elif temperature < 35.0:
                risk_indicators.append({"metric": "temperature", "value": temperature, "status": "hypothermia", "severity": "high"})
                recommendations.append("Body temperature is below normal. Warm up gradually and seek medical attention if symptoms persist.")
            else:
                recommendations.append(f"Body temperature of {temperature}°C is normal.")

    if not recommendations:
        recommendations.append("Your vitals look normal. Keep up the healthy lifestyle!")

    masked_user = f"***{request.user_id[-4:]}" if len(request.user_id) > 4 else "****"
    return HealthAnalysisResponse(
        user_id=request.user_id,
        summary=f"Analyzed {len(request.metrics or {})} metrics",
        risk_indicators=risk_indicators,
        recommendations=recommendations,
        analyzed_at=datetime.utcnow().isoformat() + "Z",
    )
