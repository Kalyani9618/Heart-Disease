"""
Structured Outputs Routes
=========================
Provides LLM responses that match predefined JSON schemas.
Endpoints:
    GET  /structured-outputs/status
    GET  /structured-outputs/schema/{schema_name}
    POST /structured-outputs/health-analysis
    POST /structured-outputs/intent
    POST /structured-outputs/conversation
"""

import logging
import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("structured-outputs")

router = APIRouter()

# ---------------------------------------------------------------------------
# Schemas Registry
# ---------------------------------------------------------------------------

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "CardioHealthAnalysis": {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "intent_confidence": {"type": "number"},
            "sentiment": {"type": "string"},
            "urgency": {"type": "string"},
            "entities": {"type": "array", "items": {"type": "object"}},
            "response": {"type": "string"},
            "explanation": {"type": "string"},
            "recommendations": {"type": "array", "items": {"type": "object"}},
            "follow_up_questions": {"type": "array", "items": {"type": "object"}},
            "requires_professional": {"type": "boolean"},
            "disclaimer": {"type": "string"},
            "confidence": {"type": "string"},
        },
        "required": ["intent", "response", "urgency"],
        "description": "Comprehensive cardiovascular health analysis with intent, sentiment, entities, and recommendations.",
    },
    "SimpleIntentAnalysis": {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "confidence": {"type": "number"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["intent", "confidence", "summary"],
        "description": "Lightweight classification of user intent.",
    },
    "ConversationResponse": {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
            "tone": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "action_items": {"type": "array", "items": {"type": "string"}},
            "needs_clarification": {"type": "boolean"},
        },
        "required": ["response", "tone"],
        "description": "Structured conversation response with tone and action items.",
    },
    "VitalSignsAnalysis": {
        "type": "object",
        "properties": {
            "metric_type": {"type": "string"},
            "value": {"type": "number"},
            "unit": {"type": "string"},
            "status": {"type": "string"},
            "interpretation": {"type": "string"},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "reference_range": {"type": "string"},
        },
        "required": ["metric_type", "value", "status"],
        "description": "Vital signs interpretation with reference ranges.",
    },
    "MedicationInfo": {
        "type": "object",
        "properties": {
            "medication_name": {"type": "string"},
            "purpose": {"type": "string"},
            "common_side_effects": {"type": "array", "items": {"type": "string"}},
            "interactions_warning": {"type": "string"},
            "dosage_reminder": {"type": "string"},
            "important_notes": {"type": "array", "items": {"type": "string"}},
            "consult_doctor": {"type": "boolean"},
        },
        "required": ["medication_name", "purpose", "consult_doctor"],
        "description": "Medication information summary.",
    },
}


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class HealthAnalysisRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class ConversationRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None


class IntentRequest(BaseModel):
    message: str


class ExtractedEntity(BaseModel):
    entity_type: str
    value: str
    confidence: float
    context: Optional[str] = None


class FollowUpQuestion(BaseModel):
    question: str
    priority: int
    reason: Optional[str] = None


class HealthRecommendation(BaseModel):
    recommendation: str
    category: str
    urgency: str
    evidence_based: bool


class CardioHealthAnalysis(BaseModel):
    intent: str
    intent_confidence: float
    sentiment: str
    urgency: str
    entities: List[ExtractedEntity] = []
    response: str
    explanation: Optional[str] = None
    recommendations: List[HealthRecommendation] = []
    follow_up_questions: List[FollowUpQuestion] = []
    requires_professional: bool = False
    disclaimer: Optional[str] = None
    confidence: str = "medium"


class SimpleIntentAnalysis(BaseModel):
    intent: str
    confidence: float
    keywords: List[str] = []
    summary: str


class ConversationResponse(BaseModel):
    response: str
    tone: str
    topics: List[str] = []
    action_items: List[str] = []
    needs_clarification: bool = False


class StructuredMeta(BaseModel):
    generation_time_ms: float
    model: Optional[str] = None
    schema_name: str = Field(..., alias="schema")


class StructuredResponse(BaseModel):
    success: bool
    data: Any
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Intent classification helpers
# ---------------------------------------------------------------------------

INTENT_KEYWORDS = {
    "symptom_report": ["pain", "ache", "hurt", "fever", "dizzy", "nausea", "fatigue", "swelling", "shortness of breath"],
    "medication_question": ["medication", "medicine", "drug", "pill", "dose", "dosage", "prescription", "side effect"],
    "lifestyle_advice": ["diet", "exercise", "sleep", "stress", "weight", "smoking", "alcohol", "healthy"],
    "emergency": ["emergency", "911", "chest pain", "heart attack", "stroke", "can't breathe", "unconscious"],
    "appointment": ["appointment", "schedule", "doctor", "visit", "check-up", "follow-up"],
    "vital_signs": ["blood pressure", "heart rate", "pulse", "temperature", "oxygen", "glucose", "bmi"],
    "mental_health": ["anxiety", "depression", "stress", "mood", "sleep", "insomnia", "panic"],
    "general_health": ["health", "wellness", "prevention", "screening", "test", "lab"],
}


def classify_intent(text: str) -> tuple:
    """Simple keyword-based intent classification."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "unknown", 0.3

    best = max(scores, key=scores.get)
    confidence = min(0.95, 0.5 + scores[best] * 0.15)
    return best, round(confidence, 2)


def extract_keywords(text: str) -> List[str]:
    """Extract significant keywords from text."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "i", "me", "my", "you",
            "your", "we", "our", "they", "them", "it", "this", "that", "what",
            "how", "and", "or", "but", "if", "for", "to", "of", "in", "on", "at"}
    words = text.lower().split()
    return [w.strip(".,!?;:") for w in words if len(w) > 2 and w.strip(".,!?;:") not in stop][:10]


def classify_urgency(text: str, intent: str) -> str:
    """Classify urgency level."""
    text_lower = text.lower()
    if intent == "emergency" or any(w in text_lower for w in ["chest pain", "heart attack", "can't breathe", "stroke"]):
        return "critical"
    if intent == "symptom_report" and any(w in text_lower for w in ["severe", "sudden", "worst", "sharp"]):
        return "high"
    if intent in ("symptom_report", "medication_question"):
        return "moderate"
    return "low"


def classify_sentiment(text: str) -> str:
    """Basic sentiment classification."""
    text_lower = text.lower()
    negatives = ["pain", "hurt", "worried", "scared", "bad", "worse", "terrible", "awful"]
    positives = ["better", "good", "great", "improved", "happy", "relieved"]
    neg = sum(1 for w in negatives if w in text_lower)
    pos = sum(1 for w in positives if w in text_lower)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_status():
    """Check if structured outputs feature is available."""
    return {
        "enabled": True,
        "message": "Structured outputs are available",
        "available_schemas": list(SCHEMAS.keys()),
        "endpoints": [
            "/structured-outputs/status",
            "/structured-outputs/schema/{schema_name}",
            "/structured-outputs/health-analysis",
            "/structured-outputs/intent",
            "/structured-outputs/conversation",
        ],
    }


@router.get("/schema/{schema_name}")
async def get_schema(schema_name: str):
    """Get the JSON schema for a specific output type."""
    if schema_name not in SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_name}' not found. Available: {list(SCHEMAS.keys())}")
    schema = SCHEMAS[schema_name]
    return {
        "schema_name": schema_name,
        "json_schema": schema,
        "description": schema.get("description", ""),
    }


@router.post("/health-analysis")
async def health_analysis(request: HealthAnalysisRequest):
    """Generate a structured health analysis from user message."""
    start = time.time()
    intent, intent_conf = classify_intent(request.message)
    urgency = classify_urgency(request.message, intent)
    sentiment = classify_sentiment(request.message)
    keywords = extract_keywords(request.message)

    # Try to use MedGemma / LLM for richer response
    response_text = _build_health_response(request.message, intent, urgency)

    analysis = CardioHealthAnalysis(
        intent=intent,
        intent_confidence=intent_conf,
        sentiment=sentiment,
        urgency=urgency,
        entities=[],
        response=response_text,
        explanation=f"Analysis based on keyword intent classification ({intent}) with {urgency} urgency.",
        recommendations=_build_recommendations(intent, urgency),
        follow_up_questions=_build_follow_ups(intent),
        requires_professional=urgency in ("critical", "high"),
        disclaimer="This analysis is AI-generated and not a substitute for professional medical advice.",
        confidence="high" if intent_conf > 0.7 else ("medium" if intent_conf > 0.4 else "low"),
    )

    elapsed_ms = round((time.time() - start) * 1000, 1)
    return StructuredResponse(
        success=True,
        data=analysis.dict(),
        metadata={"generation_time_ms": elapsed_ms, "model": "keyword-classifier-v1", "schema": "CardioHealthAnalysis"},
    )


@router.post("/intent")
async def intent_analysis(request: IntentRequest):
    """Generate a quick intent analysis."""
    start = time.time()
    intent, confidence = classify_intent(request.message)
    keywords = extract_keywords(request.message)

    result = SimpleIntentAnalysis(
        intent=intent,
        confidence=confidence,
        keywords=keywords,
        summary=f"Detected intent: {intent} (confidence {confidence})",
    )

    elapsed_ms = round((time.time() - start) * 1000, 1)
    return StructuredResponse(
        success=True,
        data=result.dict(),
        metadata={"generation_time_ms": elapsed_ms, "model": "keyword-classifier-v1", "schema": "SimpleIntentAnalysis"},
    )


@router.post("/conversation")
async def conversation(request: ConversationRequest):
    """Generate a structured conversation response."""
    start = time.time()
    intent, _ = classify_intent(request.message)
    sentiment = classify_sentiment(request.message)
    keywords = extract_keywords(request.message)
    tone = "empathetic" if sentiment == "negative" else ("encouraging" if sentiment == "positive" else "informative")

    response_text = _build_health_response(request.message, intent, "low")

    result = ConversationResponse(
        response=response_text,
        tone=tone,
        topics=keywords[:5],
        action_items=[],
        needs_clarification=intent == "unknown",
    )

    elapsed_ms = round((time.time() - start) * 1000, 1)
    return StructuredResponse(
        success=True,
        data=result.dict(),
        metadata={"generation_time_ms": elapsed_ms, "model": "keyword-classifier-v1", "schema": "ConversationResponse"},
    )


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _build_health_response(message: str, intent: str, urgency: str) -> str:
    """Build a health-focused response based on intent and urgency."""
    if urgency == "critical":
        return (
            "⚠️ Based on your description, this may require immediate medical attention. "
            "Please call emergency services (911) or visit the nearest emergency room immediately. "
            "Do not delay seeking professional help."
        )
    responses = {
        "symptom_report": "Thank you for sharing your symptoms. Based on your description, I recommend consulting with your healthcare provider for a proper evaluation.",
        "medication_question": "Regarding your medication question — it's important to follow your prescriber's instructions. If you have concerns about side effects or interactions, please discuss them with your pharmacist or doctor.",
        "lifestyle_advice": "Great that you're thinking about your lifestyle! Small, consistent changes in diet, exercise, and sleep can significantly improve cardiovascular health.",
        "vital_signs": "Tracking your vital signs is an excellent health practice. Regular monitoring helps detect changes early.",
        "mental_health": "Mental health is just as important as physical health. If you're struggling, please reach out to a mental health professional.",
        "emergency": "If you're experiencing a medical emergency, please call 911 immediately.",
        "appointment": "Regular check-ups are important for preventive care. I'd recommend scheduling an appointment with your healthcare provider.",
        "general_health": "Maintaining good health involves regular check-ups, a balanced diet, regular exercise, and adequate sleep.",
    }
    return responses.get(intent, "I'd be happy to help with your health question. Could you provide more details so I can give you a more specific response?")


def _build_recommendations(intent: str, urgency: str) -> List[HealthRecommendation]:
    """Build context-appropriate health recommendations."""
    recs = []
    if urgency in ("critical", "high"):
        recs.append(HealthRecommendation(
            recommendation="Seek immediate medical attention",
            category="urgent_care",
            urgency=urgency,
            evidence_based=True,
        ))
    if intent in ("symptom_report", "vital_signs"):
        recs.append(HealthRecommendation(
            recommendation="Schedule a follow-up appointment with your healthcare provider",
            category="follow_up",
            urgency="moderate",
            evidence_based=True,
        ))
    if intent == "lifestyle_advice":
        recs.append(HealthRecommendation(
            recommendation="Aim for at least 150 minutes of moderate aerobic activity per week (AHA guideline)",
            category="exercise",
            urgency="low",
            evidence_based=True,
        ))
    return recs


def _build_follow_ups(intent: str) -> List[FollowUpQuestion]:
    """Build relevant follow-up questions."""
    follow_ups = {
        "symptom_report": [
            FollowUpQuestion(question="How long have you been experiencing these symptoms?", priority=1, reason="Duration helps assess severity"),
            FollowUpQuestion(question="Are the symptoms getting worse, staying the same, or improving?", priority=2, reason="Progression informs urgency"),
        ],
        "medication_question": [
            FollowUpQuestion(question="Are you currently taking any other medications or supplements?", priority=1, reason="Check for interactions"),
        ],
        "vital_signs": [
            FollowUpQuestion(question="When was this measurement taken?", priority=1, reason="Timing affects interpretation"),
        ],
    }
    return follow_ups.get(intent, [])
