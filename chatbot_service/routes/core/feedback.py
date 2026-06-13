"""
Feedback Routes - User Feedback Collection for RAG Quality Improvement

This module provides API endpoints for collecting user feedback on
RAG responses, enabling continuous improvement of retrieval quality.

Endpoints:
    POST /feedback - Submit feedback for a response
    GET /feedback/stats - Get feedback statistics
"""


from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    feedback_id: str = Field(..., description="Response ID (from chat response)")
    rating: int = Field(..., ge=-1, le=1, description="1=positive, 0=neutral, -1=negative")
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Generated response")
    citations: List[Dict[str, Any]] = Field(default=[], description="Response citations")
    user_id: Optional[str] = Field(None, description="User identifier")
    comment: Optional[str] = Field(None, description="Optional user comment")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    message: str
    feedback_id: str


class FeedbackStats(BaseModel):
    """Feedback statistics model."""
    total_feedback: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_rating: float
    recent_negative: List[Dict[str, Any]] = []


# Store reference (initialized during app startup via lifespan)
# ✅ FIXED: Removed global _feedback_store initialization
# Feedback store is now initialized via FastAPI lifespan in app_lifespan.py
# This ensures every worker has its own store instance


def get_feedback_store():
    """
    Get feedback store instance.
    
    **IMPORTANT:** Must be called AFTER app startup.
    Feedback store is initialized in app_lifespan.startup_event()
    """
    from app_lifespan import get_feedback_store as _get_feedback_store
    return _get_feedback_store()


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> FeedbackResponse:
    """
    Submit user feedback for a RAG response.
    
    This enables:
    - Quality tracking of RAG responses
    - Identification of problematic queries
    - Continuous improvement via RLHF signals
    """
    feedback_store = get_feedback_store()
    if not feedback_store:
        raise HTTPException(
            status_code=503,
            detail="Feedback service not initialized"
        )
    
    try:
        success = await feedback_store.record_feedback(
            feedback_id=request.feedback_id,
            rating=request.rating,
            query=request.query,
            response=request.response,
            citations=request.citations,
            user_id=str(current_user.get('user_id', request.user_id)),
            comment=request.comment
        )
        
        if success:
            rating_label = "positive" if request.rating == 1 else (
                "negative" if request.rating == -1 else "neutral"
            )
            logger.info(f"Feedback received: {request.feedback_id} ({rating_label})")
            
            return FeedbackResponse(
                success=True,
                message=f"Thank you for your {rating_label} feedback!",
                feedback_id=request.feedback_id
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to record feedback"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission error: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Error recording feedback"
        )


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> FeedbackStats:
    """
    Get feedback statistics for monitoring.
    
    Returns aggregate stats and recent negative feedback
    for quality improvement analysis.
    """
    feedback_store = get_feedback_store()
    if not feedback_store:
        raise HTTPException(
            status_code=503,
            detail="Feedback service not initialized"
        )
    
    try:
        # Get negative feedback for analysis
        negative = await feedback_store.get_negative_feedback(limit=10)
        
        # Get counts (if available on store)
        stats = {
            "total_feedback": 0,
            "positive_count": 0,
            "negative_count": len(negative),
            "neutral_count": 0,
            "average_rating": 0.0,
            "recent_negative": negative
        }
        
        # Try to get full stats if method exists
        if hasattr(feedback_store, 'get_stats'):
            full_stats = await feedback_store.get_stats()
            stats.update(full_stats)
        
        return FeedbackStats(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback stats: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving stats"
        )


@router.get("/health")
async def feedback_health() -> Dict[str, Any]:
    """Health check for feedback service."""
    feedback_store = get_feedback_store()
    return {
        "status": "ok" if feedback_store else "not_initialized",
        "store_type": type(feedback_store).__name__ if feedback_store else None
    }
