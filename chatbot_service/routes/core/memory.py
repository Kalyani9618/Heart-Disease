"""
Memory Management API Routes.

Provides REST endpoints for:
- User preferences CRUD
- Session management
- Context retrieval preview
- GDPR compliance (export, delete)
- Semantic memory search (EmbeddingSearchEngine)
- Conscious memory agent management (ConsciouscAgent)
- Memory statistics and metrics

Author: AI Memory System Implementation
Version: 2.0.0 (Enhanced with Memori Integration)
"""


from fastapi import APIRouter, HTTPException, Depends, Query, Request, status, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from core.security import get_current_user
from core.user.user_preferences import (
    UserPreferencesManager,
    get_preferences_manager,
)
from memori.memory_observability import (
    metrics_endpoint,
    health_endpoint,
    detailed_metrics_endpoint,
)


from enum import Enum

class ContextType(str, Enum):
    RECENT_INTERACTIONS = "recent_interactions"
    USER_PROFILE = "user_profile"
    PREFERENCES = "preferences"
    CHAT_HISTORY = "chat_history"
    USER_PREFERENCES = "user_preferences"
    RECENT_CONVERSATIONS = "recent_conversations"
    MEDICATIONS = "medications"


class ContextRetriever:
    async def retrieve_for_query(self, *args, **kwargs): return []

default_context_retriever = ContextRetriever()

class ChatHistoryManager:
    def get_user_sessions(self, *args, **kwargs): return []
    def get_history(self, *args, **kwargs): return []
    def get_session_info(self, *args, **kwargs): return {}
    def clear(self, *args, **kwargs): pass

chat_history_manager = ChatHistoryManager()


class AIQueryResponse:
    """Response object for AI query."""
    def __init__(
        self,
        response: str = "",
        session_id: str = "",
        success: bool = True,
        context_used: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        audit: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.response = response
        self.session_id = session_id
        self.success = success
        self.context_used = context_used or []
        self.metadata = metadata or {}
        self.audit = audit or {}
        self.error = error


class IntegratedHealthAIService:
    """
    Integrated Health AI Service for processing queries with memory context.
    
    Provides unified interface for AI-powered health queries with:
    - Memory retrieval and context assembly
    - Integration with LLM providers (Gemini, Ollama)
    - Audit logging and HIPAA compliance
    """
    
    def __init__(self):
        self.memory_manager = None
        self.llm_gateway = None
        self._initialized = False
        
        # Try to get services from DIContainer
        try:
            from core.dependencies import DIContainer
            container = DIContainer.get_instance()
            self.memory_manager = container.memory_manager
            self.llm_gateway = container.llm_gateway
            
            # Get optimizers
            self.rag_optimizer = getattr(container, 'rag_optimizer', None)
            self.memory_optimizer = getattr(container, 'memory_optimizer', None)
            
            self._initialized = True
        except Exception as e:
            logger.warning(f"IntegratedHealthAIService: Could not initialize from DIContainer: {e}")
    
    async def process_query(
        self,
        user_id: str,
        session_id: str,
        query: str,
        patient_name: Optional[str] = None,
        patient_age: Optional[int] = None,
        is_emergency: bool = False,
        ai_provider: Optional[str] = None,
    ) -> AIQueryResponse:
        """
        Process a health query with memory context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User's question
            patient_name: Optional patient name for context
            patient_age: Optional patient age for context
            is_emergency: Emergency flag for prioritization
            ai_provider: AI provider preference (gemini/ollama)
            
        Returns:
            AIQueryResponse with response and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Build context from memory if available
            context_used = []
            
            # 1. Try Optimized RAG Search first (Fastest + Cached)
            if self.rag_optimizer:
                try:
                    rag_results = await self.rag_optimizer.search(
                        query=query,
                        top_k=3,
                        use_cache=True
                    )
                    if rag_results:
                        context_used.extend([
                            {
                                "type": "memory", 
                                "source": "rag_optimized", 
                                "data": r.get("content", "")[:300],
                                "relevance": r.get("score", 0)
                            }
                            for r in rag_results
                        ])
                except Exception as e:
                    logger.warning(f"Optimized RAG search failed: {e}")

            # 2. Fallback/Augment with Memory Manager (if not enough results)
            if len(context_used) < 3 and self.memory_manager and hasattr(self.memory_manager, 'search_memory'):
                try:
                    memory_results = await self.memory_manager.search_memory(
                        patient_id=user_id,
                        query=query,
                        limit=5
                    )
                    if memory_results:
                        # Deduplicate
                        existing_content = {c.get("data") for c in context_used}
                        for r in memory_results[:3]:
                            content = str(r)[:200]
                            if content not in existing_content:
                                context_used.append({
                                    "type": "memory", 
                                    "source": "memori", 
                                    "data": content
                                })
                except Exception as e:
                    logger.warning(f"Memory search failed: {e}")
            
            # Generate response using LLM
            response_text = ""
            
            if self.llm_gateway:
                try:
                    # Build prompt with context
                    context_str = "\n".join([str(c.get("data", "")) for c in context_used])
                    prompt = f"""You are a helpful health assistant. Answer the following query.

Context from previous interactions:
{context_str if context_str else "No previous context available."}

Patient Info: {patient_name or 'Unknown'}, Age: {patient_age or 'Unknown'}
Emergency: {'Yes' if is_emergency else 'No'}

Query: {query}

Provide a helpful, accurate response. If this is an emergency, advise seeking immediate medical attention."""

                    result = await self.llm_gateway.generate(
                        prompt=prompt
                    )
                    response_text = result if isinstance(result, str) else (result.get("text", "") if isinstance(result, dict) else str(result))
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    response_text = f"I apologize, but I encountered an error processing your query. Please try again or consult a healthcare provider directly."
            else:
                # Fallback response when LLM not available
                response_text = (
                    f"I received your query: '{query[:100]}...'. "
                    "However, the AI service is currently unavailable. "
                    "Please try again later or consult a healthcare provider."
                )
            
            elapsed_time = time.time() - start_time
            
            return AIQueryResponse(
                response=response_text,
                session_id=session_id,
                success=True,
                context_used=context_used,
                metadata={
                    "ai_provider": ai_provider or "default",
                    "is_emergency": is_emergency,
                    "processing_time_ms": round(elapsed_time * 1000, 2),
                },
                audit={
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "query_length": len(query),
                    "response_length": len(response_text),
                },
                error=None,
            )
            
        except Exception as e:
            logger.error(f"IntegratedHealthAIService.process_query error: {e}")
            return AIQueryResponse(
                response="",
                session_id=session_id,
                success=False,
                context_used=[],
                metadata={},
                audit={"error": str(e)},
                error=str(e),
            )

def get_integrated_ai_service():
    return IntegratedHealthAIService()

# Import Memori components for enhanced memory features
try:
    from memori.agents.retrieval_agent import EmbeddingSearchEngine, MemorySearchEngine
    from memori.agents.conscious_agent import ConsciouscAgent
    from memori.utils.rate_limiter import RateLimiter
    from memori.utils.input_validator import InputValidator

    MEMORI_AVAILABLE = True
except ImportError:
    MEMORI_AVAILABLE = False
    EmbeddingSearchEngine = None
    MemorySearchEngine = None
    ConsciouscAgent = None
    RateLimiter = None
    InputValidator = None

# Import memory manager for enhanced operations
try:
    from memori.memory_manager import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False
    MemoryManager = None

# Import MemoriRAGBridge for hybrid search (semantic + keyword)
try:
    from rag.memory.memori_integration import MemoriRAGBridge, create_memori_rag_bridge
    MEMORI_RAG_BRIDGE_AVAILABLE = True
except ImportError:
    MEMORI_RAG_BRIDGE_AVAILABLE = False
    MemoriRAGBridge = None
    create_memori_rag_bridge = None

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["memory"])


# ============================================================================
# Pydantic Models
# ============================================================================


class PreferenceRequest(BaseModel):
    """Request model for setting a preference."""

    key: str = Field(..., description="Preference key")
    value: Any = Field(..., description="Preference value (any JSON-serializable type)")
    is_sensitive: bool = Field(False, description="Whether this is sensitive/PHI data")
    category: str = Field("general", description="Preference category")

    class Config:
        json_schema_extra = {
            "example": {
                "key": "communication_style",
                "value": "detailed",
                "is_sensitive": False,
                "category": "general",
            }
        }


class BulkPreferenceRequest(BaseModel):
    """Request model for setting multiple preferences."""

    preferences: Dict[str, Any] = Field(
        ..., description="Dictionary of key-value pairs"
    )
    category: str = Field("general", description="Category for all preferences")


class ContextSearchRequest(BaseModel):
    """Request model for context retrieval."""

    query: str = Field(..., description="Query to find relevant context for")
    context_types: Optional[List[str]] = Field(
        None, description="Specific context types to retrieve"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What does my blood pressure mean?",
                "context_types": ["recent_vitals", "medications"],
            }
        }


class AIQueryRequest(BaseModel):
    """Request model for integrated AI query."""

    query: str = Field(..., description="User's question")
    patient_name: Optional[str] = Field(None, description="Patient name")
    patient_age: Optional[int] = Field(None, description="Patient age")
    is_emergency: bool = Field(False, description="Emergency query flag")
    ai_provider: Optional[str] = Field(None, description="AI provider (gemini/ollama)")


class PreferenceResponse(BaseModel):
    """Response model for preferences."""

    user_id: str
    key: str
    value: Any
    updated_at: Optional[str] = None


class SessionInfo(BaseModel):
    """Session information response."""

    session_id: str
    user_id: Optional[str]
    created_at: Optional[str]
    last_activity: Optional[str]
    message_count: int


class ContextPreview(BaseModel):
    """Context preview response."""

    type: str
    source: str
    relevance_score: float
    data_preview: Dict[str, Any]


class GDPRExportResponse(BaseModel):
    """GDPR data export response."""

    user_id: str
    export_timestamp: str
    preferences: List[Dict[str, Any]]
    total_count: int


# ============================================================================
# NEW: Semantic Search Request/Response Models
# ============================================================================


class SemanticSearchRequest(BaseModel):
    """Request model for semantic memory search."""

    query: str = Field(..., description="Natural language query for semantic search")
    session_id: str = Field("default", description="Session identifier")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    memory_types: Optional[List[str]] = Field(
        None, description="Filter by memory types"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What were my recent blood pressure readings?",
                "limit": 10,
                "similarity_threshold": 0.6,
                "memory_types": ["health_data", "vitals"],
            }
        }


class SemanticSearchResult(BaseModel):
    """Single result from semantic search."""

    memory_id: str
    content: str
    memory_type: str
    similarity_score: float
    timestamp: str
    metadata: Dict[str, Any] = {}


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search."""

    query: str
    results: List[SemanticSearchResult]
    total_found: int
    search_time_ms: float
    embedding_model: str


class ConsciousMemoryRequest(BaseModel):
    """Request model for conscious memory operations."""

    action: str = Field(..., description="Action: ingest, initialize, status")
    limit: int = Field(10, ge=1, le=100, description="Limit for initialize action")

    class Config:
        json_schema_extra = {"example": {"action": "initialize", "limit": 10}}


class ConsciousMemoryResponse(BaseModel):
    """Response model for conscious memory operations."""

    action: str
    success: bool
    memories_processed: int
    context_initialized: bool
    message: str


class MemoryMetricsResponse(BaseModel):
    """Response model for memory system metrics."""

    enabled: bool
    initialized: bool
    cache_size: int
    cache_max_size: int
    searches: Dict[str, Any]
    stores: Dict[str, Any]
    cache: Dict[str, Any]
    circuit_breaker: Dict[str, Any]
    memori_available: bool


# ============================================================================
# Dependencies
# ============================================================================


def get_client_ip(request: Request) -> str:
    """Extract client IP for audit logging."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_preferences() -> UserPreferencesManager:
    """Dependency to get preferences manager."""
    return get_preferences_manager()


def get_context_retriever() -> ContextRetriever:
    """Dependency to get context retriever."""
    return default_context_retriever


def get_ai_service() -> IntegratedHealthAIService:
    """Dependency to get integrated AI service."""
    return get_integrated_ai_service()


# ============================================================================
# Preference Endpoints
# ============================================================================


@router.get("/preferences/{user_id}")
async def get_user_preferences(
    user_id: str,
    include_sensitive: bool = Query(False, description="Include sensitive preferences"),
    category: Optional[str] = Query(None, description="Filter by category"),
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get all preferences for a user.

    Args:
        user_id: User identifier (must match authenticated user)
        include_sensitive: Include PHI/sensitive preferences
        category: Filter by category (optional)
        current_user: Authenticated user from JWT token

    Returns:
        Dictionary of all preferences
    """
    # ✅ CRITICAL: Enforce ownership - users can only access their own preferences
    # Convert both to strings to handle int/str type mismatch
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only access your own preferences",
        )

    try:
        prefs = preferences.get_all_preferences(
            user_id=user_id, include_sensitive=include_sensitive, category=category
        )

        return {
            "user_id": user_id,
            "preferences": prefs,
            "count": len(prefs),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting preferences for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get preferences: {str(e)}",
        )


@router.get("/preferences/{user_id}/{key}")
async def get_single_preference(
    user_id: str,
    key: str,
    default: Optional[str] = Query(None, description="Default value if not found"),
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> PreferenceResponse:
    """
    Get a single preference value.

    Args:
        user_id: User identifier (must match authenticated user)
        key: Preference key
        default: Default value if not found
        current_user: Authenticated user from JWT token

    Returns:
        Preference value
    """
    # ✅ CRITICAL: Enforce ownership - users can only access their own preferences
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only access your own preferences",
        )

    value = preferences.get_preference(user_id=user_id, key=key, default=default)

    return PreferenceResponse(user_id=user_id, key=key, value=value)


@router.put("/preferences/{user_id}")
async def set_user_preference(
    user_id: str,
    preference: PreferenceRequest,
    request: Request,
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> PreferenceResponse:
    """
    Set a user preference.

    Args:
        user_id: User identifier (must match authenticated user)
        preference: Preference data
        current_user: Authenticated user from JWT token

    Returns:
        Confirmation with updated value
    """
    # ✅ CRITICAL: Enforce ownership - users can only set their own preferences
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only set your own preferences",
        )

    try:
        ip_address = get_client_ip(request)

        preferences.set_preference(
            user_id=user_id,
            key=preference.key,
            value=preference.value,
            is_sensitive=preference.is_sensitive,
            category=preference.category,
            ip_address=ip_address,
        )

        return PreferenceResponse(
            user_id=user_id,
            key=preference.key,
            value=preference.value,
            updated_at=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error setting preference for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set preference: {str(e)}",
        )


@router.put("/preferences/{user_id}/bulk")
async def set_bulk_preferences(
    user_id: str,
    bulk_request: BulkPreferenceRequest,
    request: Request,
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Set multiple preferences at once.

    Args:
        user_id: User identifier (must match authenticated user)
        bulk_request: Multiple preferences to set
        current_user: Authenticated user from JWT token

    Returns:
        Count of preferences set
    """
    # ✅ CRITICAL: Enforce ownership - users can only set their own preferences
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only set your own preferences",
        )

    try:
        ip_address = get_client_ip(request)

        count = preferences.set_preferences(
            user_id=user_id,
            preferences=bulk_request.preferences,
            category=bulk_request.category,
            ip_address=ip_address,
        )

        return {
            "user_id": user_id,
            "preferences_set": count,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error setting bulk preferences for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set preferences: {str(e)}",
        )


@router.delete("/preferences/{user_id}/{key}")
async def delete_user_preference(
    user_id: str,
    key: str,
    request: Request,
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Delete a specific preference.

    Args:
        user_id: User identifier (must match authenticated user)
        key: Preference key to delete
        current_user: Authenticated user from JWT token

    Returns:
        Confirmation of deletion
    """
    # ✅ CRITICAL: Enforce ownership - users can only delete their own preferences
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only delete your own preferences",
        )

    try:
        ip_address = get_client_ip(request)

        deleted = preferences.delete_preference(
            user_id=user_id, key=key, ip_address=ip_address
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preference '{key}' not found for user '{user_id}'",
            )

        return {
            "user_id": user_id,
            "key": key,
            "deleted": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting preference for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preference: {str(e)}",
        )


# ============================================================================
# Session Endpoints
# ============================================================================


@router.get("/sessions/{user_id}")
async def list_user_sessions(
    user_id: str,
    limit: int = Query(50, ge=1, le=200),
    include_expired: bool = Query(False),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all chat sessions for a user.

    Args:
        user_id: User identifier (must match authenticated user)
        limit: Maximum sessions to return
        include_expired: Include expired sessions
        current_user: Authenticated user from JWT token

    Returns:
        List of session information
    """
    # ✅ CRITICAL: Enforce ownership - users can only list their own sessions
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only list your own sessions",
        )

    try:
        # Get sessions from chat history
        sessions = (
            chat_history_manager.get_user_sessions(user_id=user_id, limit=limit)
            if hasattr(chat_history_manager, "get_user_sessions")
            else []
        )

        return {
            "user_id": user_id,
            "sessions": sessions,
            "count": len(sessions),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error listing sessions for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}",
        )


@router.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get full history for a session.

    Args:
        session_id: Session identifier
        limit: Maximum messages to return
        current_user: Authenticated user from JWT token

    Returns:
        Session history with messages
    """
    # ✅ CRITICAL: Validate session ownership before returning data (HIPAA compliance)
    try:
        session_info = chat_history_manager.get_session_info(session_id)
        if session_info and str(session_info.get("user_id")) != str(current_user.get("user_id")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only access your own sessions"
            )

        history = chat_history_manager.get_history(session_id=session_id, limit=limit)

        # session_info already retrieved above for ownership check

        return {
            "session_id": session_id,
            "session_info": session_info,
            "messages": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting history for {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session history: {str(e)}",
        )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a session and its history.

    Args:
        session_id: Session identifier
        current_user: Authenticated user from JWT token

    Returns:
        Confirmation of deletion
    """
    # ✅ CRITICAL: Validate session ownership before deletion (HIPAA compliance)
    try:
        session_info = chat_history_manager.get_session_info(session_id)
        if session_info and str(session_info.get("user_id")) != str(current_user.get("user_id")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only delete your own sessions"
            )

        chat_history_manager.clear(session_id)

        return {
            "session_id": session_id,
            "deleted": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        )


# ============================================================================
# Context Retrieval Endpoints
# ============================================================================


@router.post("/context/retrieve")
async def retrieve_context(
    user_id: str = Query(..., description="User identifier"),
    session_id: str = Query(..., description="Session identifier"),
    request_body: ContextSearchRequest = ...,
    retriever: ContextRetriever = Depends(get_context_retriever),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Retrieve relevant context for a query.

    Useful for debugging what context will be used for AI calls.

    Args:
        user_id: User identifier (must match authenticated user)
        session_id: Session identifier
        request_body: Query and optional context types
        current_user: Authenticated user from JWT token

    Returns:
        List of relevant context items
    """
    # ✅ CRITICAL: Enforce ownership - users can only retrieve context for their own user_id
    # Compare as strings to handle int/str type mismatch
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only retrieve context for your own user ID",
        )

    try:
        # Parse context types if provided
        context_types = None
        if request_body.context_types:
            try:
                context_types = [ContextType(ct) for ct in request_body.context_types]
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid context type: {e}",
                )

        # Retrieve context
        contexts = await retriever.retrieve_for_query(
            user_id=user_id,
            session_id=session_id,
            query=request_body.query,
            context_types=context_types,
        )

        # Format response
        return {
            "user_id": user_id,
            "session_id": session_id,
            "query": request_body.query,
            "contexts": [
                {
                    "type": ctx.context_type.value,
                    "source": ctx.source,
                    "relevance_score": ctx.relevance_score,
                    "token_estimate": ctx.token_estimate,
                    "data_preview": _truncate_data(ctx.data, max_len=200),
                }
                for ctx in contexts
            ],
            "count": len(contexts),
            "total_tokens": sum(ctx.token_estimate for ctx in contexts),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve context: {str(e)}",
        )


@router.get("/context/types")
async def list_context_types() -> Dict[str, Any]:
    """
    List all available context types.

    Returns:
        List of context type names and descriptions
    """
    return {
        "context_types": [
            {"name": ct.value, "description": _get_context_type_description(ct)}
            for ct in ContextType
        ],
        "count": len(ContextType),
    }


# ============================================================================
# Integrated AI Query Endpoint
# ============================================================================


@router.post("/ai/query")
async def ai_query(
    user_id: str = Query(..., description="User identifier"),
    session_id: str = Query(..., description="Session identifier"),
    request_body: AIQueryRequest = ...,
    ai_service: IntegratedHealthAIService = Depends(get_ai_service),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Process query through integrated AI service.

    Full flow: store → retrieve context → build prompt → AI call → store response.

    Args:
        user_id: User identifier (must match authenticated user)
        session_id: Session identifier
        request_body: Query parameters
        current_user: Authenticated user from JWT token

    Returns:
        AI response with metadata and audit trail
    """
    # ✅ CRITICAL: Enforce ownership - users can only query with their own user_id
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only query with your own user ID",
        )

    try:
        response = await ai_service.process_query(
            user_id=user_id,
            session_id=session_id,
            query=request_body.query,
            patient_name=request_body.patient_name,
            patient_age=request_body.patient_age,
            is_emergency=request_body.is_emergency,
            ai_provider=request_body.ai_provider,
        )

        return {
            "response": response.response,
            "session_id": response.session_id,
            "success": response.success,
            "context_used": response.context_used,
            "metadata": response.metadata,
            "audit": response.audit,
            "error": response.error,
        }
    except Exception as e:
        logger.error(f"Error in AI query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI query failed: {str(e)}",
        )


# ============================================================================
# GDPR Compliance Endpoints
# ============================================================================


@router.post("/gdpr/export/{user_id}")
async def export_user_data(
    user_id: str,
    format: str = Query("json", enum=["json"]),
    request: Request = None,
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> GDPRExportResponse:
    """
    Export all user data (GDPR compliance).

    Args:
        user_id: User identifier (must match authenticated user)
        format: Export format (currently only JSON)
        current_user: Authenticated user from JWT token

    Returns:
        Complete export of user preferences and data
    """
    # ✅ CRITICAL: Enforce ownership - users can only export their own data
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only export your own data",
        )

    try:
        ip_address = get_client_ip(request) if request else None

        export_data = preferences.export_user_data(
            user_id=user_id, format=format, ip_address=ip_address
        )

        return GDPRExportResponse(**export_data)
    except Exception as e:
        logger.error(f"Error exporting data for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export data: {str(e)}",
        )


@router.delete("/gdpr/delete/{user_id}")
async def delete_user_data(
    user_id: str,
    confirm: bool = Query(False, description="Confirm deletion"),
    request: Request = None,
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Delete all user data (GDPR right to erasure).

    Args:
        user_id: User identifier (must match authenticated user)
        confirm: Must be true to confirm deletion
        current_user: Authenticated user from JWT token

    Returns:
        Confirmation of deletion with counts
    """
    # ✅ CRITICAL: Enforce ownership - users can only delete their own data
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only delete your own data",
        )

    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must set confirm=true to delete all user data",
        )

    try:
        ip_address = get_client_ip(request) if request else None

        # Delete preferences
        pref_result = preferences.delete_user_data(
            user_id=user_id, ip_address=ip_address
        )

        # Delete chat history sessions
        chat_deleted = 0
        try:
            if hasattr(chat_history_manager, "delete_user_sessions"):
                chat_deleted = chat_history_manager.delete_user_sessions(user_id)
        except Exception as e:
            logger.warning(f"Failed to delete chat sessions: {e}")

        return {
            "user_id": user_id,
            "deleted": True,
            "preferences_deleted": pref_result.get("preferences_deleted", 0),
            "sessions_deleted": chat_deleted,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete data: {str(e)}",
        )


# ============================================================================
# Audit Endpoints
# ============================================================================


@router.get("/audit/{user_id}")
async def get_audit_log(
    user_id: str,
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = Query(None, description="Filter by action type"),
    preferences: UserPreferencesManager = Depends(get_preferences),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get audit log for a user.

    Args:
        user_id: User identifier (must match authenticated user)
        limit: Maximum records to return
        action: Filter by action type (set, delete, export, etc.)
        current_user: Authenticated user from JWT token

    Returns:
        List of audit log entries
    """
    # ✅ CRITICAL: Enforce ownership - users can only access their own audit logs
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only access your own audit logs",
        )

    try:
        logs = preferences.get_audit_log(user_id=user_id, limit=limit, action=action)

        return {
            "user_id": user_id,
            "audit_logs": logs,
            "count": len(logs),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting audit log for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit log: {str(e)}",
        )


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health")
async def memory_health_check(
    preferences: UserPreferencesManager = Depends(get_preferences),
    retriever: ContextRetriever = Depends(get_context_retriever),
) -> Dict[str, Any]:
    """
    Health check for memory management system.

    Returns:
        Health status of all memory components
    """
    health = {"status": "healthy", "components": {}}

    # Check preferences
    try:
        pref_health = preferences.health_check()
        health["components"]["preferences"] = pref_health
        if pref_health.get("status") != "healthy":
            health["status"] = "degraded"
    except Exception as e:
        health["components"]["preferences"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    # Check context retriever
    try:
        health["components"]["context_retriever"] = retriever.get_stats()
    except Exception as e:
        health["components"]["context_retriever"] = {"status": "error", "error": str(e)}

    # Check chat history
    try:
        chat_history_manager.get_history("health_check", limit=1)
        health["components"]["chat_history"] = {"status": "ok"}
    except Exception as e:
        health["components"]["chat_history"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    health["timestamp"] = datetime.utcnow().isoformat()
    return health


# ============================================================================
# NEW: Semantic Memory Search Endpoints (EmbeddingSearchEngine)
# ============================================================================

# ✅ FIXED: Embedding search engine is now initialized during app startup
# See: app_lifespan.py - startup_event()
# Eliminates 3-5 second delay on first semantic search request


def get_embedding_search_engine():
    """
    Get embedding search engine instance.
    
    **IMPORTANT:** Must be called AFTER app startup.
    Embedding search engine is initialized in app_lifespan.startup_event()
    
    This prevents the first semantic search request from hanging while
    loading the sentence-transformers model (~3-5 seconds).
    """
    from app_lifespan import get_embedding_search_engine as _get_engine
    return _get_engine()


@router.post("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_memory_search(
    request_body: SemanticSearchRequest,
    user_id: str = Query(..., description="User identifier"),
    current_user: dict = Depends(get_current_user),
) -> SemanticSearchResponse:
    """
    Perform semantic search across user memories using embedding similarity.

    Uses EmbeddingSearchEngine from Memori for intelligent retrieval:
    - Local sentence-transformers for fast embedding (all-MiniLM-L6-v2)
    - Cosine similarity ranking
    - Embedding cache for performance
    - **✅ Model pre-loaded during startup (no delay)**

    Args:
        request_body: Search parameters
        user_id: User identifier (must match authenticated user)
        current_user: Authenticated user from JWT token

    Returns:
        Semantically relevant memories ranked by similarity
    """
    # ✅ CRITICAL: Enforce ownership - users can only search their own memories
    # Compare as strings to handle int/str type mismatch
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only search your own memories",
        )

    import time

    start_time = time.time()

    if not MEMORI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memori semantic search not available. Install memori package.",
        )

    # Validate input using Memori's InputValidator
    try:
        if InputValidator:
            validated_query = InputValidator.validate_and_sanitize_query(
                request_body.query, max_length=5000
            )
        else:
            validated_query = request_body.query
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid query: {str(e)}"
        )

    search_engine = get_embedding_search_engine()
    if not search_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding search engine not initialized",
        )

    # Optimized RAG Search (Preferred)
    rag_optimizer = getattr(container, 'rag_optimizer', None) if "container" in locals() or "container" in globals() else None
    
    # Try to get container if not available
    if not rag_optimizer:
        try:
            from core.dependencies import DIContainer
            container = DIContainer.get_instance()
            rag_optimizer = getattr(container, 'rag_optimizer', None)
        except Exception:
            pass

    if rag_optimizer:
        try:
            rag_results = await rag_optimizer.search(
                query=validated_query,
                top_k=request_body.limit,
                use_cache=True,
                filters={"similarity_threshold": request_body.similarity_threshold}
            )
            
            results = [
                SemanticSearchResult(
                    memory_id=r.get("id", f"rag_{i}"),
                    content=r.get("content", ""),
                    memory_type=r.get("metadata", {}).get("type", "rag_memory"),
                    similarity_score=r.get("score", 0.0),
                    timestamp=r.get("metadata", {}).get("timestamp", datetime.utcnow().isoformat()),
                    metadata=r.get("metadata", {}),
                )
                for i, r in enumerate(rag_results)
                if r.get("score", 0) >= request_body.similarity_threshold
            ]
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return SemanticSearchResponse(
                query=validated_query,
                results=results,
                total_found=len(results),
                search_time_ms=round(elapsed_ms, 2),
                embedding_model="optimized-rag-model",
            )
        except Exception as e:
            logger.warning(f"Optimized RAG semantic search failed, falling back: {e}")

    try:
        # Use MemoryManager for search if available
        if MEMORY_MANAGER_AVAILABLE:
            memory_mgr = MemoryManager.get_instance()
            search_results = await memory_mgr.search_memory(
                patient_id=user_id,
                query=validated_query,
                session_id=request_body.session_id,
                limit=request_body.limit,
            )

            # Convert to response format
            results = [
                SemanticSearchResult(
                    memory_id=r.id,
                    content=r.content,
                    memory_type=r.memory_type,
                    similarity_score=r.relevance_score,
                    timestamp=r.timestamp,
                    metadata=r.metadata,
                )
                for r in search_results
                if r.relevance_score >= request_body.similarity_threshold
            ]
        else:
            # Fallback to basic search
            results = []

        elapsed_ms = (time.time() - start_time) * 1000

        return SemanticSearchResponse(
            query=validated_query,
            results=results,
            total_found=len(results),
            search_time_ms=round(elapsed_ms, 2),
            embedding_model="MedCPT-Query-Encoder" if search_engine.use_local else "openai",
        )

    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )


@router.get("/search/semantic/status")
async def semantic_search_status() -> Dict[str, Any]:
    """
    Get status of semantic search capabilities.

    Returns:
        Status information about embedding search engine
    """
    search_engine = get_embedding_search_engine()

    return {
        "memori_available": MEMORI_AVAILABLE,
        "embedding_engine_initialized": search_engine is not None,
        "embedding_model": (
            {
                "local": search_engine.use_local if search_engine else False,
                "model_name": (
                    search_engine.DEFAULT_LOCAL_MODEL if search_engine else None
                ),
                "openai_fallback": (
                    search_engine.openai_client is not None if search_engine else False
                ),
            }
            if search_engine
            else None
        ),
        "cache_size": len(search_engine._embedding_cache) if search_engine else 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# NEW: Conscious Memory Agent Endpoints
# ============================================================================

# Initialize conscious agent (singleton)
_conscious_agent = None


def get_conscious_agent():
    """Get or create conscious agent singleton."""
    global _conscious_agent
    if _conscious_agent is None and MEMORI_AVAILABLE:
        try:
            _conscious_agent = ConsciouscAgent()
            logger.info("ConsciouscAgent initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ConsciouscAgent: {e}")
            _conscious_agent = None
    return _conscious_agent


@router.post("/conscious", response_model=ConsciousMemoryResponse)
async def manage_conscious_memory(
    user_id: str = Query(..., description="User identifier"),
    request_body: ConsciousMemoryRequest = ...,
    current_user: dict = Depends(get_current_user),
) -> ConsciousMemoryResponse:
    """
    Manage conscious memory context using Memori's ConsciouscAgent.

    The ConsciouscAgent copies 'conscious-info' labeled memories from
    long-term memory to short-term memory for immediate context availability.

    Actions:
    - ingest: Run conscious context ingestion (copies conscious-info memories)
    - initialize: Initialize existing conscious memories
    - status: Get current conscious memory status

    Args:
        user_id: User identifier (must match authenticated user)
        request_body: Action and parameters
        current_user: Authenticated user from JWT token

    Returns:
        Result of conscious memory operation
    """
    # ✅ CRITICAL: Enforce ownership - users can only manage their own memories
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only manage your own memories",
        )

    if not MEMORI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memori conscious agent not available",
        )

    conscious_agent = get_conscious_agent()
    if not conscious_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conscious agent not initialized",
        )

    action = request_body.action.lower()

    try:
        if action == "status":
            return ConsciousMemoryResponse(
                action=action,
                success=True,
                memories_processed=0,
                context_initialized=conscious_agent.context_initialized,
                message="Conscious agent status retrieved",
            )

        elif action in ["ingest", "initialize"]:
            # Get database manager from MemoryManager if available
            if MEMORY_MANAGER_AVAILABLE:
                memory_mgr = MemoryManager.get_instance()

                # Get patient memory to access db_manager
                try:
                    patient_memory = await memory_mgr.get_patient_memory(user_id)
                    db_manager = (
                        patient_memory.memori.db_manager
                        if hasattr(patient_memory.memori, "db_manager")
                        else None
                    )
                except Exception as e:
                    logger.warning(f"Could not get db_manager: {e}")
                    db_manager = None

                if db_manager:
                    if action == "ingest":
                        success = await conscious_agent.run_conscious_ingest(
                            db_manager=db_manager, user_id=user_id
                        )
                    else:  # initialize
                        success = await conscious_agent.initialize_existing_conscious_memories(
                            db_manager=db_manager,
                            user_id=user_id,
                            limit=request_body.limit,
                        )

                    return ConsciousMemoryResponse(
                        action=action,
                        success=success,
                        memories_processed=request_body.limit if success else 0,
                        context_initialized=conscious_agent.context_initialized,
                        message=f"Conscious memory {action} completed",
                    )
                else:
                    return ConsciousMemoryResponse(
                        action=action,
                        success=False,
                        memories_processed=0,
                        context_initialized=False,
                        message="Database manager not available",
                    )
            else:
                return ConsciousMemoryResponse(
                    action=action,
                    success=False,
                    memories_processed=0,
                    context_initialized=False,
                    message="Memory manager not available",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {action}. Use 'ingest', 'initialize', or 'status'",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conscious memory error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conscious memory operation failed: {str(e)}",
        )



# ============================================================================
# NEW: Observability Endpoints
# ============================================================================

@router.get("/metrics/prometheus", response_class=Response)
async def get_prometheus_metrics():
    """Get metrics in Prometheus text format."""
    return Response(content=await metrics_endpoint(), media_type="text/plain")


@router.get("/health/detailed", response_model=Dict[str, Any])
async def get_detailed_health():
    """Get comprehensive health check for Memori."""
    return await health_endpoint()


@router.get("/metrics/detailed", response_model=Dict[str, Any])
async def get_detailed_metrics_json():
    """Get detailed metrics in JSON format."""
    return await detailed_metrics_endpoint()


# ============================================================================
# NEW: Memory System Metrics Endpoint
# ============================================================================


@router.get("/metrics", response_model=MemoryMetricsResponse)
async def get_memory_metrics() -> MemoryMetricsResponse:
    """
    Get comprehensive metrics for the memory management system.

    Includes:
    - Cache statistics (hits, misses, hit rate)
    - Search/store operation counts and latencies
    - Circuit breaker status
    - Memori availability status

    Returns:
        Detailed memory system metrics
    """
    if not MEMORY_MANAGER_AVAILABLE:
        return MemoryMetricsResponse(
            enabled=False,
            initialized=False,
            cache_size=0,
            cache_max_size=0,
            searches={},
            stores={},
            cache={},
            circuit_breaker={},
            memori_available=MEMORI_AVAILABLE,
        )

    try:
        memory_mgr = MemoryManager.get_instance()
        metrics = memory_mgr.get_metrics()

        return MemoryMetricsResponse(
            enabled=metrics.get("enabled", False),
            initialized=metrics.get("initialized", False),
            cache_size=metrics.get("cache_size", 0),
            cache_max_size=metrics.get("cache_max_size", 0),
            searches=metrics.get("metrics", {}).get("searches", {}),
            stores=metrics.get("metrics", {}).get("stores", {}),
            cache=metrics.get("metrics", {}).get("cache", {}),
            circuit_breaker=metrics.get("metrics", {}).get("circuit_breaker", {}),
            memori_available=MEMORI_AVAILABLE,
        )
    except Exception as e:
        logger.error(f"Error getting memory metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory metrics: {str(e)}",
        )


# ============================================================================
# NEW: Rate Limiter Status Endpoint
# ============================================================================


@router.get("/rate-limit/status")
async def get_rate_limit_status(
    user_id: str = Query(..., description="User identifier"),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get rate limit status for a user.

    Shows current request counts and quotas from Memori's RateLimiter.

    Args:
        user_id: User identifier (must match authenticated user)
        current_user: Authenticated user from JWT token

    Returns:
        Rate limit status and quotas
    """
    # ✅ CRITICAL: Enforce ownership - users can only check their own rate limits
    if str(user_id) != str(current_user.get("user_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only check your own rate limits",
        )

    if not MEMORI_AVAILABLE or not RateLimiter:
        return {
            "available": False,
            "message": "Rate limiter not available",
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        # Check if rate limiter is available
        _ = RateLimiter()

        return {
            "available": True,
            "user_id": user_id,
            "limits": {
                "search_requests_per_minute": 60,
                "store_requests_per_minute": 100,
                "api_calls_per_day": 1000,
            },
            "message": "Rate limiter is active",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        return {
            "available": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# Helper Functions
# ============================================================================


def _truncate_data(data: Dict[str, Any], max_len: int = 200) -> Dict[str, Any]:
    """Truncate data values for preview."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_len:
            result[key] = value[:max_len] + "..."
        elif isinstance(value, (list, dict)):
            str_val = str(value)
            if len(str_val) > max_len:
                result[key] = str_val[:max_len] + "..."
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def _get_context_type_description(ct: ContextType) -> str:
    """Get human-readable description for context type."""
    descriptions = {
        ContextType.RECENT_INTERACTIONS: "Recent user interactions and activities",
        ContextType.USER_PROFILE: "User profile and personal information",
        ContextType.MEDICATIONS: "Current medications list",
        ContextType.RECENT_CONVERSATIONS: "Recent chat messages in session",
        ContextType.USER_PREFERENCES: "User settings and preferences",
        ContextType.CHAT_HISTORY: "Chat history and conversation logs",
        ContextType.PREFERENCES: "User preferences and settings",
    }
    return descriptions.get(ct, ct.value)
