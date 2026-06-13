"""
Model Management Routes
=======================
Endpoints for tracking ML model versions, history, and availability.
Endpoints:
    GET /models/versions
    GET /models/history/{model_name}
    GET /models/list
"""

import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("models-management")

router = APIRouter()


# ---------------------------------------------------------------------------
# Auto-discover models from disk
# ---------------------------------------------------------------------------

def _get_model_dir() -> str:
    """Get the models directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def _discover_models() -> Dict[str, Dict]:
    """Discover available models from the file system."""
    model_dir = _get_model_dir()
    models = {}

    if not os.path.exists(model_dir):
        return models

    for subdir in os.listdir(model_dir):
        subdir_path = os.path.join(model_dir, subdir)
        if os.path.isdir(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith(('.joblib', '.onnx', '.bin', '.pt', '.h5'))]
            if files:
                models[subdir] = {
                    "name": subdir,
                    "type": _infer_model_type(subdir),
                    "status": "active",
                    "description": _get_model_description(subdir),
                    "files": files,
                    "directory": subdir_path,
                    "versions": _extract_versions(files),
                    "current_version": _get_latest_version(files),
                    "last_updated": datetime.utcfromtimestamp(
                        max(os.path.getmtime(os.path.join(subdir_path, f)) for f in files)
                    ).isoformat() + "Z",
                }

    return models


def _infer_model_type(name: str) -> str:
    """Infer model type from directory name."""
    mapping = {
        "heart_disease": "classification",
        "medgemma": "llm",
        "cross-encoder_model": "reranker",
        "onnx_models": "inference",
        "spacy_models": "ner",
    }
    return mapping.get(name, "unknown")


def _get_model_description(name: str) -> str:
    """Get human-readable description of a model."""
    mapping = {
        "heart_disease": "Stacking ensemble model for heart disease prediction (8 base models + meta-learner)",
        "medgemma": "MedGemma medical LLM for clinical interpretation",
        "cross-encoder_model": "Cross-encoder model for semantic reranking",
        "onnx_models": "ONNX runtime optimized models for fast inference",
        "spacy_models": "spaCy NLP models for medical entity recognition",
    }
    return mapping.get(name, f"Model: {name}")


def _extract_versions(files: List[str]) -> List[str]:
    """Extract version identifiers from filenames."""
    versions = []
    for f in files:
        # Look for version patterns like v3, v4, etc.
        import re
        match = re.search(r'v(\d+)', f)
        if match:
            versions.append(f"v{match.group(1)}")
    return sorted(set(versions), key=lambda v: int(v[1:])) if versions else ["v1"]


def _get_latest_version(files: List[str]) -> str:
    """Get the latest version from filenames."""
    versions = _extract_versions(files)
    return versions[-1] if versions else "v1"


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    description: Optional[str] = None


class ModelVersion(BaseModel):
    name: str
    current_version: str
    versions: List[str]
    last_updated: str


class ModelHistoryEntry(BaseModel):
    version: str
    deployed_at: str
    metrics: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ModelHistoryResponse(BaseModel):
    model_name: str
    history: List[ModelHistoryEntry]


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]


class ModelVersionsResponse(BaseModel):
    models: List[ModelVersion]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/versions", response_model=ModelVersionsResponse)
async def get_model_versions():
    """Get all model versions."""
    discovered = _discover_models()
    models = []
    for name, info in discovered.items():
        models.append(ModelVersion(
            name=name,
            current_version=info["current_version"],
            versions=info["versions"],
            last_updated=info["last_updated"],
        ))

    return ModelVersionsResponse(models=models)


@router.get("/history/{model_name}", response_model=ModelHistoryResponse)
async def get_model_history(model_name: str):
    """Get version history for a specific model."""
    discovered = _discover_models()

    if model_name not in discovered:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available: {list(discovered.keys())}")

    info = discovered[model_name]
    subdir_path = info["directory"]
    history = []
    for version in info["versions"]:
        # Find files matching this version to get per-version timestamp
        version_files = [f for f in info["files"] if version in f]
        if version_files:
            deployed_ts = datetime.utcfromtimestamp(
                max(os.path.getmtime(os.path.join(subdir_path, f)) for f in version_files)
            ).isoformat() + "Z"
        else:
            deployed_ts = info["last_updated"]
        history.append(ModelHistoryEntry(
            version=version,
            deployed_at=deployed_ts,
            metrics={"files": len(version_files) if version_files else len(info["files"])},
            notes=f"Model files: {', '.join(info['files'][:5])}{'...' if len(info['files']) > 5 else ''}",
        ))

    return ModelHistoryResponse(model_name=model_name, history=history)


@router.get("/list", response_model=ModelsListResponse)
async def list_models():
    """List all available models."""
    discovered = _discover_models()
    models = [
        ModelInfo(
            name=name,
            type=info["type"],
            status=info["status"],
            description=info["description"],
        )
        for name, info in discovered.items()
    ]
    return ModelsListResponse(models=models)
