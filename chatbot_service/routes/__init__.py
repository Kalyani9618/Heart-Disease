"""
Routes module for Cardio AI NLP Service.

Organized into 3 sub-packages:
- core/     - Authentication, Chat, Memory, Documents, Users, Feedback, SSE, WebSocket, Speech
- health/   - Heart Prediction, Smartwatch, Vision (ECG), Medical AI, Tools, Calendar, Notifications
- admin/    - Database Health, RAG Health, NLP Debug, Job Management, Models, Evaluation
"""

from fastapi import APIRouter

# Import individual routers from sub-packages
from .core.auth_routes import router as auth_router
from .core.memory import router as memory_router
from .core.orchestrated_chat import router as orchestrated_chat_router
from .core.documents import router as document_router

# Create a placeholder router for backward compatibility
router = APIRouter()

__all__ = [
    "router",
    "auth_router",
    "memory_router",
    "orchestrated_chat_router",
    "document_router",
]
