"""
Medical AI Routes
=================
NLP-powered medical entity extraction, patient summaries, and terminology expansion.
Leverages existing backend services: spacy_service, medical_phrase_matcher, MedGemma.
Endpoints:
    POST /medical-ai/extract-entities
    POST /medical-ai/patient-summary
    POST /medical-ai/terminology
"""

import logging
import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("medical-ai")

router = APIRouter()


# ---------------------------------------------------------------------------
# Try to load existing backend services
# ---------------------------------------------------------------------------

_spacy_service = None
_phrase_matcher = None
_medgemma = None

try:
    from core.services.spacy_service import SpaCyService
    _spacy_service = SpaCyService()
    logger.info("SpacyService loaded for medical entity extraction")
except Exception as e:
    logger.info(f"SpacyService not available: {e}")

try:
    from core.services.medical_phrase_matcher import MedicalPhraseMatcher
    if _spacy_service and hasattr(_spacy_service, '_nlp') and _spacy_service._nlp:
        _phrase_matcher = MedicalPhraseMatcher(_spacy_service._nlp)
        logger.info("MedicalPhraseMatcher loaded")
    else:
        logger.info("MedicalPhraseMatcher skipped: SpaCyService NLP pipeline not available")
except Exception as e:
    logger.info(f"MedicalPhraseMatcher not available: {e}")

try:
    from core.llm.medgemma_service import MedGemmaService
    _medgemma = MedGemmaService.get_instance()
    logger.info("MedGemma service available for medical AI")
except Exception as e:
    logger.info(f"MedGemma service not available: {e}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EntityExtractionRequest(BaseModel):
    text: str


class ExtractedEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = None


class EntityExtractionResponse(BaseModel):
    entities: List[ExtractedEntity]
    processing_time_ms: float


class PatientSummaryRequest(BaseModel):
    user_id: str


class PatientSummaryResponse(BaseModel):
    user_id: str
    summary: str
    conditions: List[str] = []
    medications: List[str] = []
    risk_factors: List[str] = []
    last_updated: str


class TerminologyRequest(BaseModel):
    term: str


class TerminologyResponse(BaseModel):
    term: str
    definition: str
    synonyms: List[str] = []
    related_terms: List[str] = []
    category: Optional[str] = None


# ---------------------------------------------------------------------------
# Basic medical terminology database
# ---------------------------------------------------------------------------

MEDICAL_TERMS = {
    "hypertension": {
        "definition": "A condition in which the blood pressure in the arteries is persistently elevated (≥130/80 mmHg). Also known as high blood pressure.",
        "synonyms": ["high blood pressure", "HTN", "arterial hypertension"],
        "related_terms": ["systolic pressure", "diastolic pressure", "antihypertensive", "prehypertension"],
        "category": "cardiovascular",
    },
    "tachycardia": {
        "definition": "A heart rate exceeding 100 beats per minute at rest. Can be a sign of underlying cardiac or systemic conditions.",
        "synonyms": ["rapid heart rate", "fast heartbeat"],
        "related_terms": ["bradycardia", "arrhythmia", "SVT", "ventricular tachycardia"],
        "category": "cardiovascular",
    },
    "bradycardia": {
        "definition": "A heart rate below 60 beats per minute at rest. May be normal in athletes or indicate cardiac conduction issues.",
        "synonyms": ["slow heart rate"],
        "related_terms": ["tachycardia", "heart block", "pacemaker", "sinus bradycardia"],
        "category": "cardiovascular",
    },
    "angina": {
        "definition": "Chest pain caused by reduced blood flow to the heart muscle, often a symptom of coronary artery disease.",
        "synonyms": ["angina pectoris", "chest pain"],
        "related_terms": ["myocardial ischemia", "coronary artery disease", "nitroglycerin", "stable angina", "unstable angina"],
        "category": "cardiovascular",
    },
    "myocardial infarction": {
        "definition": "Death of heart muscle tissue due to prolonged ischemia. Commonly known as a heart attack.",
        "synonyms": ["heart attack", "MI", "STEMI", "NSTEMI"],
        "related_terms": ["troponin", "coronary artery", "thrombolysis", "angioplasty", "cardiac arrest"],
        "category": "cardiovascular",
    },
    "arrhythmia": {
        "definition": "An abnormal heart rhythm. The heart may beat too fast, too slow, or irregularly.",
        "synonyms": ["dysrhythmia", "irregular heartbeat"],
        "related_terms": ["atrial fibrillation", "ventricular fibrillation", "ECG", "antiarrhythmic"],
        "category": "cardiovascular",
    },
    "cholesterol": {
        "definition": "A waxy substance found in blood. High levels of LDL cholesterol increase risk of cardiovascular disease.",
        "synonyms": ["blood lipids"],
        "related_terms": ["LDL", "HDL", "triglycerides", "statin", "lipid panel", "hyperlipidemia"],
        "category": "metabolic",
    },
    "diabetes": {
        "definition": "A metabolic disease characterized by high blood sugar levels over a prolonged period.",
        "synonyms": ["diabetes mellitus", "DM"],
        "related_terms": ["insulin", "HbA1c", "type 1", "type 2", "metformin", "hyperglycemia", "hypoglycemia"],
        "category": "metabolic",
    },
    "ecg": {
        "definition": "Electrocardiogram — a test that measures the electrical activity of the heart to detect abnormalities.",
        "synonyms": ["electrocardiogram", "EKG", "12-lead ECG"],
        "related_terms": ["P wave", "QRS complex", "ST segment", "T wave", "arrhythmia"],
        "category": "diagnostic",
    },
    "stent": {
        "definition": "A small mesh tube inserted into a narrowed coronary artery to keep it open and restore blood flow.",
        "synonyms": ["coronary stent", "drug-eluting stent", "DES"],
        "related_terms": ["angioplasty", "PCI", "restenosis", "antiplatelet therapy"],
        "category": "interventional",
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """Extract medical entities from text using NLP."""
    start = time.time()

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    entities = []

    # Try spaCy NER
    if _spacy_service:
        try:
            result = _spacy_service.extract_entities(request.text)
            if isinstance(result, list):
                for ent in result:
                    entities.append(ExtractedEntity(
                        text=ent.get("text", ""),
                        label=ent.get("label", "UNKNOWN"),
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                        confidence=ent.get("confidence"),
                    ))
        except Exception as e:
            logger.warning(f"SpaCy extraction failed: {e}")

    # Fallback: basic keyword matching against our medical terms
    if not entities:
        text_lower = request.text.lower()
        for term, info in MEDICAL_TERMS.items():
            idx = text_lower.find(term)
            if idx >= 0:
                entities.append(ExtractedEntity(
                    text=request.text[idx:idx + len(term)],
                    label=info.get("category", "MEDICAL").upper(),
                    start=idx,
                    end=idx + len(term),
                    confidence=0.8,
                ))
            # Also check synonyms
            for syn in info.get("synonyms", []):
                idx = text_lower.find(syn.lower())
                if idx >= 0:
                    entities.append(ExtractedEntity(
                        text=request.text[idx:idx + len(syn)],
                        label=info.get("category", "MEDICAL").upper(),
                        start=idx,
                        end=idx + len(syn),
                        confidence=0.7,
                    ))

    elapsed = round((time.time() - start) * 1000, 1)
    return EntityExtractionResponse(entities=entities, processing_time_ms=elapsed)


@router.post("/patient-summary", response_model=PatientSummaryResponse)
async def get_patient_summary(request: PatientSummaryRequest):
    """Generate a patient summary from stored health data."""
    from datetime import datetime
    from core.database.postgres_db import get_database

    db = await get_database()

    # Query real data from DB
    conditions = await db.get_user_conditions(request.user_id)
    medications = await db.get_user_medications_list(request.user_id)

    # Build risk factors from recent vitals
    risk_factors = []
    vitals_summary = await db.get_weekly_vitals_summary(request.user_id)
    if vitals_summary:
        avg_hr = vitals_summary.get("avg_heart_rate")
        if avg_hr and avg_hr > 100:
            risk_factors.append("Elevated resting heart rate")
        avg_sys = vitals_summary.get("avg_systolic")
        if avg_sys and avg_sys >= 130:
            risk_factors.append("Elevated blood pressure")

    summary_parts = []
    if conditions:
        summary_parts.append(f"Known conditions: {', '.join(conditions)}.")
    if medications:
        summary_parts.append(f"Current medications: {', '.join(medications)}.")
    if risk_factors:
        summary_parts.append(f"Risk factors: {', '.join(risk_factors)}.")
    summary_text = " ".join(summary_parts) if summary_parts else "No health records found for this patient."

    return PatientSummaryResponse(
        user_id=request.user_id,
        summary=summary_text,
        conditions=conditions,
        medications=medications,
        risk_factors=risk_factors,
        last_updated=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/terminology", response_model=TerminologyResponse)
async def expand_terminology(request: TerminologyRequest):
    """Look up medical terminology with definitions, synonyms, and related terms."""
    if not request.term or not request.term.strip():
        raise HTTPException(status_code=400, detail="Term is required")

    term_lower = request.term.lower().strip()

    # Direct lookup
    if term_lower in MEDICAL_TERMS:
        info = MEDICAL_TERMS[term_lower]
        return TerminologyResponse(
            term=request.term,
            definition=info["definition"],
            synonyms=info.get("synonyms", []),
            related_terms=info.get("related_terms", []),
            category=info.get("category"),
        )

    # Check if the term is a synonym of any known term
    for canonical, info in MEDICAL_TERMS.items():
        if term_lower in [s.lower() for s in info.get("synonyms", [])]:
            return TerminologyResponse(
                term=request.term,
                definition=info["definition"],
                synonyms=[canonical] + [s for s in info.get("synonyms", []) if s.lower() != term_lower],
                related_terms=info.get("related_terms", []),
                category=info.get("category"),
            )

    # Try LLM if available
    if _medgemma:
        try:
            response = await _medgemma.generate(
                f"Define the medical term '{request.term}' in one concise paragraph. Include common synonyms."
            )
            return TerminologyResponse(
                term=request.term,
                definition=response if isinstance(response, str) else str(response),
                synonyms=[],
                related_terms=[],
                category=None,
            )
        except Exception as e:
            logger.warning(f"MedGemma terminology lookup failed: {e}")

    # Not found
    return TerminologyResponse(
        term=request.term,
        definition=f"No definition found for '{request.term}'. Please consult a medical dictionary or your healthcare provider.",
        synonyms=[],
        related_terms=[],
        category=None,
    )
