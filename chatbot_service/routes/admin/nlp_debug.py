"""
NLP Debug Routes
================
Endpoints for visualizing and debugging the NLP pipeline.
"""


from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except Exception as e:
    spacy = None  # type: ignore
    displacy = None  # type: ignore
    SPACY_AVAILABLE = False
import logging

from core.services.spacy_service import get_spacy_service
from core.security import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def nlp_health():
    """Health check for NLP pipeline."""
    service_ok = False
    model_name = None
    pipeline = []
    try:
        service = get_spacy_service()
        if service and service.nlp:
            service_ok = True
            model_name = service.nlp.meta.get("name")
            pipeline = service.nlp.pipe_names
    except Exception:
        pass
    return {
        "status": "healthy" if service_ok else "degraded",
        "service": "NLP Pipeline",
        "spacy_available": SPACY_AVAILABLE,
        "model": model_name,
        "pipeline": pipeline,
    }


class VisualizeRequest(BaseModel):
    text: str
    style: str = "ent"  # "ent" or "dep"
    minify: bool = True

@router.post("/visualize", response_class=HTMLResponse)
async def visualize_text(
    request: VisualizeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Render spaCy visualization for text.
    
    Args:
        text: Text to analyze
        style: Visualization style ("ent" for entities, "dep" for dependencies)
    """
    if not SPACY_AVAILABLE:
        raise HTTPException(status_code=503, detail="spaCy is not available")
    try:
        service = get_spacy_service()
        doc = service.process(request.text)
        
        options = {}
        if request.style == "ent":
            # Custom colors for medical entities
            colors = {
                "DRUG": "#f08080",      # Light Coral
                "MEDICATION": "#f08080",
                "DOSAGE": "#add8e6",    # Light Blue
                "DISEASE": "#90ee90",   # Light Green
                "CONDITION": "#90ee90",
                "SYMPTOM": "#ffb6c1",   # Light Pink
                "PROBLEM": "#ffb6c1",
            }
            options = {"colors": colors}
            
        html = displacy.render(
            doc, 
            style=request.style, 
            page=True, 
            minify=request.minify,
            options=options
        )
        return html
        
    except Exception as e:
        logger.error(f"Visualization failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Visualization processing failed")

@router.get("/pipeline/info")
async def get_pipeline_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get information about the active NLP pipeline."""
    if not SPACY_AVAILABLE:
        raise HTTPException(status_code=503, detail="spaCy is not available")
    try:
        service = get_spacy_service()
        nlp = service.nlp
    except Exception as e:
        logger.error(f"Pipeline info failed: {type(e).__name__}")
        raise HTTPException(status_code=503, detail="NLP pipeline not available")
    
    return {
        "model": nlp.meta["name"],
        "version": nlp.meta["version"],
        "pipeline": nlp.pipe_names,
        "tokenizer": str(type(nlp.tokenizer)),
    }

@router.post("/tokenize")
async def inspect_tokens(
    request: VisualizeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Inspect how text is tokenized."""
    service = get_spacy_service()
    doc = service.process(request.text)
    
    tokens = []
    for token in doc:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop,
        })
        
    return {"tokens": tokens}
