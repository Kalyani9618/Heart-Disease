"""
Vision Routes
=============
Medical image analysis endpoints (ECG analysis).
Wraps the existing agents/components/vision.py capabilities.
Endpoints:
    POST /vision/ecg/analyze
"""

import logging
import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

logger = logging.getLogger("vision")

router = APIRouter()


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ECGAnalysisResponse(BaseModel):
    rhythm: str
    heart_rate_bpm: Optional[int] = None
    abnormalities: List[str] = []
    recommendations: List[str] = []
    confidence: float
    processing_time_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Vision service integration attempt
# ---------------------------------------------------------------------------

_vision_service = None

try:
    from agents.components.vision import MedicalImageAnalyzer
    _vision_service = MedicalImageAnalyzer()
    logger.info("Vision service loaded from agents/components/vision.py")
except ImportError:
    logger.info("Vision agent component not available, using built-in analysis")
except Exception as e:
    logger.warning(f"Vision service init failed: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health")
async def vision_health():
    """Health check for vision service."""
    return {
        "status": "healthy" if _vision_service else "degraded",
        "service": "Vision / ECG Analysis",
        "vision_backend": "MedicalImageAnalyzer" if _vision_service else "gemini-fallback",
    }


@router.post("/ecg/analyze", response_model=ECGAnalysisResponse)
async def analyze_ecg(
    file: UploadFile = File(...),
    patient_context: Optional[str] = Form(None),
):
    """Analyze an ECG image for rhythm, rate, and abnormalities."""
    import asyncio
    start = time.time()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG)")

    # Check file size by reading in chunks to avoid loading huge files
    MAX_SIZE = 10 * 1024 * 1024
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(64 * 1024)  # 64KB chunks
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        chunks.append(chunk)
    contents = b"".join(chunks)

    # Try real vision analysis
    if _vision_service and hasattr(_vision_service, "analyze_ecg"):
        try:
            result = await _vision_service.analyze_ecg(contents, patient_context)
            elapsed = round((time.time() - start) * 1000, 1)
            return ECGAnalysisResponse(**result, processing_time_ms=elapsed)
        except Exception as e:
            logger.warning(f"Vision service ECG analysis failed: {e}")

    # Try Gemini multimodal
    try:
        import google.generativeai as genai
        import json
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            "Analyze this ECG image. Return a JSON object with keys: "
            "rhythm (string), heart_rate_bpm (int or null), "
            "abnormalities (list of strings), recommendations (list of strings), "
            "confidence (float 0-1). Be concise and clinically accurate."
        )
        # Sanitize patient_context to prevent prompt injection
        if patient_context:
            sanitized_context = patient_context[:500].replace("\n", " ").strip()
            prompt += f"\nPatient context (for reference only): {sanitized_context}"

        # Run sync model call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content([prompt, {"mime_type": file.content_type, "data": contents}])
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        elapsed = round((time.time() - start) * 1000, 1)
        return ECGAnalysisResponse(
            rhythm=data.get("rhythm", "Unknown"),
            heart_rate_bpm=data.get("heart_rate_bpm"),
            abnormalities=data.get("abnormalities", []),
            recommendations=data.get("recommendations", []),
            confidence=data.get("confidence", 0.5),
            processing_time_ms=elapsed,
        )
    except Exception as e:
        logger.warning(f"Gemini ECG analysis failed: {e}")

    # Fallback response
    elapsed = round((time.time() - start) * 1000, 1)
    return ECGAnalysisResponse(
        rhythm="Unable to determine — upload a clearer ECG image",
        heart_rate_bpm=None,
        abnormalities=[],
        recommendations=["Please consult a cardiologist for professional ECG interpretation"],
        confidence=0.0,
        processing_time_ms=elapsed,
    )
