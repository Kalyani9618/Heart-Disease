"""
Data models for NLP Service

OPTIMIZATION (PHASE 1 TASK 1.4): Enhanced input validation with:
- HTML sanitization
- Control character removal
- Length limits (prevent oversized requests)
- Pattern validation for IDs
- Security checks
"""


from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from html import escape
import re
from enum import Enum


class IntentEnum(str, Enum):
    """Intent types"""

    GREETING = "greeting"
    UNKNOWN = "unknown"


class SentimentEnum(str, Enum):
    """Sentiment types"""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    DISTRESSED = "distressed"
    URGENT = "urgent"


class Entity(BaseModel):
    """Extracted entity"""

    type: str = Field(..., description="Entity type")
    value: str = Field(..., description="Entity value")
    start_index: int = Field(..., description="Start index in original text")
    end_index: int = Field(..., description="End index in original text")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")


class IntentResult(BaseModel):
    """Intent recognition result"""

    intent: IntentEnum = Field(..., description="Identified intent")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    keywords_matched: List[str] = Field(
        default_factory=list, description="Matched keywords"
    )


class SentimentResult(BaseModel):
    """Sentiment analysis result"""

    sentiment: SentimentEnum = Field(..., description="Detected sentiment")
    score: float = Field(..., ge=-1, le=1, description="Sentiment score")
    intensity: str = Field(..., description="Intensity level (mild, moderate, severe)")


class NLPProcessRequest(BaseModel):
    """
    Request for NLP processing with comprehensive input validation.

    Security Features:
    - HTML escaping to prevent injection
    - Control character removal
    - Maximum length limits
    - Pattern validation for IDs
    """

    message: str = Field(
        ...,
        description="User message to process",
        min_length=1,
        max_length=10000,  # Prevent oversized requests
    )
    session_id: Optional[str] = Field(
        None,
        description="Chat session ID for context",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",  # Alphanumeric, underscore, hyphen only
    )
    user_id: Optional[str] = Field(
        None,
        description="User ID for personalization",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",  # Alphanumeric, underscore, hyphen only
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context"
    )
    model: Optional[str] = Field(
        default="gemini",
        description="AI model to use for response generation (gemini or ollama)",
        pattern=r"^(gemini|ollama)$",  # Only allow gemini or ollama
    )
    use_rag: Optional[bool] = Field(
        default=True,
        description="Use RAG (Retrieval-Augmented Generation) for context-enhanced responses",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Hello world",
                "session_id": "sess_12345abc",
                "user_id": "user_98765xyz",
                "context": {"location": "home", "time_of_day": "evening"},
            }
        }
    )

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """
        Sanitize input message to prevent injection attacks.

        Removes:
        - Null bytes (\x00)
        - Control characters (except newlines/tabs)
        - Excessive whitespace

        Escapes HTML entities to prevent XSS.
        """
        if not v:
            raise ValueError("Message cannot be empty")

        # Remove null bytes (null injection attack)
        v = v.replace("\x00", "")

        # Remove control characters (except newlines and tabs)
        v = "".join(c for c in v if ord(c) >= 32 or c in "\n\t\r")

        # HTML escape to prevent injection
        v = escape(v)

        # Normalize whitespace (multiple spaces → single space)
        v = " ".join(v.split())

        # Remove leading/trailing whitespace
        v = v.strip()

        if not v:
            raise ValueError("Message cannot be empty after sanitization")

        return v

    @field_validator("session_id", "user_id")
    @classmethod
    def validate_id_format(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate ID format.

        Allowed: Alphanumeric characters, underscores, hyphens
        Prevents: SQL injection, special characters that might break URLs
        """
        if v is None:
            return v

        # Pattern check already done by Field(pattern=...), but explicit validation helps
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError(
                f"ID must contain only alphanumeric characters, underscores, and hyphens. Got: {v}"
            )

        return v


class NLPProcessResponse(BaseModel):
    """Response from NLP processing"""

    intent: IntentEnum = Field(..., description="Identified intent")
    intent_confidence: float = Field(..., description="Intent confidence score")

    sentiment: SentimentEnum = Field(..., description="Detected sentiment")
    sentiment_score: float = Field(..., description="Sentiment analysis score")

    entities: List[Entity] = Field(
        default_factory=list, description="Extracted entities"
    )

    keywords_matched: List[str] = Field(
        default_factory=list, description="Keywords that matched"
    )

    suggested_response: str = Field(..., description="Suggested response template")

    context_updates: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Suggested context updates"
    )

    requires_escalation: bool = Field(
        False, description="Whether this requires human escalation"
    )

    confidence_overall: float = Field(
        ..., description="Overall confidence of the analysis"
    )


class EntityExtractionRequest(BaseModel):
    """Request for entity extraction"""

    text: str = Field(..., description="Text to extract entities from")
    entity_types: Optional[List[str]] = Field(
        None, description="Specific entity types to extract"
    )


class EntityExtractionResponse(BaseModel):
    """Response from entity extraction"""

    entities: List[Entity] = Field(
        default_factory=list, description="Extracted entities"
    )
    text_chunks: Optional[List[str]] = Field(
        None, description="Text chunks with annotations"
    )


class HealthCheckResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Timestamp of health check")
    models_loaded: Dict[str, bool] = Field(
        default_factory=dict, description="Status of loaded models"
    )


class OllamaResponseRequest(BaseModel):
    """Request for Ollama response generation with input validation"""

    message: str = Field(
        ...,
        description="User message to generate response for",
        min_length=1,
        max_length=10000,  # Prevent oversized requests
    )
    model: str = Field(
        default="gemma3:1b",
        description="Ollama model to use (gemma3:1b or gemma3:4b)",
        pattern=r"^[a-zA-Z0-9:_\-\.]+$",  # Model name format
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description="Previous messages in format [{'role': 'user|assistant', 'content': 'text'}, ...]",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Optional system prompt for context",
        max_length=5000,
    )
    stream: bool = Field(default=False, description="Whether to stream response")
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (higher = more creative)",
    )
    max_tokens: Optional[int] = Field(
        None, description="DEPRECATED: Ollama doesn't support max_tokens. Ignored."
    )
    user_id: Optional[str] = Field(
        None,
        description="User ID for tracking",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for context",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",
    )

    @field_validator("message", "system_prompt")
    @classmethod
    def sanitize_text_fields(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize text fields like NLPProcessRequest"""
        if v is None:
            return v

        # Remove null bytes
        v = v.replace("\x00", "")

        # Remove control characters
        v = "".join(c for c in v if ord(c) >= 32 or c in "\n\t\r")

        # HTML escape
        v = escape(v)

        # Normalize whitespace
        v = " ".join(v.split()).strip()

        return v


class OllamaResponseResponse(BaseModel):
    """Response from Ollama generation"""

    response: str = Field(..., description="Generated response text")
    model: str = Field(..., description="Model used for generation")
    generation_time_ms: float = Field(
        ..., description="Time taken to generate response"
    )
    tokens_generated: int = Field(default=0, description="Number of tokens generated")
    success: bool = Field(default=True, description="Whether generation was successful")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class OllamaHealthCheckResponse(BaseModel):
    """Health check response for Ollama"""

    status: str = Field(..., description="Connection status (healthy/unhealthy)")
    model: str = Field(..., description="Ollama model name")
    ollama_host: str = Field(..., description="Ollama server URL")
    available: bool = Field(..., description="Whether model is available")
    timestamp: str = Field(..., description="Check timestamp")


# NEW: Pydantic models for agent requests
class AgentRequest(BaseModel):
    """Request for agent processing"""

    query: str = Field(
        ...,
        description="User query to process",
        min_length=1,
        max_length=10000,
    )
    user_id: Optional[str] = Field(
        None,
        description="User ID for personalization",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for context",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-]+$",
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context for processing"
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize input query to prevent injection attacks."""
        if not v:
            raise ValueError("Query cannot be empty")

        # Remove null bytes (null injection attack)
        v = v.replace("\x00", "")

        # Remove control characters (except newlines and tabs)
        v = "".join(c for c in v if ord(c) >= 32 or c in "\n\t\r")

        # HTML escape to prevent injection
        v = escape(v)

        # Normalize whitespace (multiple spaces → single space)
        v = " ".join(v.split())

        # Remove leading/trailing whitespace
        v = v.strip()

        if not v:
            raise ValueError("Query cannot be empty after sanitization")

        return v


class AgentResponse(BaseModel):
    """Response from agent processing"""

    status: str = Field(..., description="Processing status (success/error)")
    agent: Optional[str] = Field(None, description="Agent name that handled request")
    action: Optional[str] = Field(None, description="Action taken")
    response: str = Field(..., description="Agent's response")
    data: Optional[Dict[str, Any]] = Field(
        None, description="Additional data from processing"
    )
    timestamp: str = Field(..., description="Timestamp of response")
